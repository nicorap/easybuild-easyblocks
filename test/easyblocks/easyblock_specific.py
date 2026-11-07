##
# Copyright 2019-2026 Ghent University
#
# This file is part of EasyBuild,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://www.vscentrum.be),
# Flemish Research Foundation (FWO) (http://www.fwo.be/en)
# and the Department of Economy, Science and Innovation (EWI) (http://www.ewi-vlaanderen.be/en).
#
# https://github.com/easybuilders/easybuild
#
# EasyBuild is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation v2.
#
# EasyBuild is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EasyBuild.  If not, see <http://www.gnu.org/licenses/>.
##
"""
Unit tests for specific easyblocks.

@author: Kenneth Hoste (Ghent University)
"""
import base64
import copy
import gzip
import hashlib
import http.server
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
from pathlib import Path
from unittest import TestLoader, TextTestRunner
from test.easyblocks.module import cleanup

import easybuild.tools.options as eboptions
import easybuild.tools.tomllib as tomllib
import easybuild.easyblocks.generic.pythonpackage as pythonpackage
import easybuild.easyblocks.generic.cargo as cargo
import easybuild.easyblocks.generic.containerstack as containerstack
import easybuild.easyblocks.l.lammps as lammps
import easybuild.easyblocks.p.python as python
import easybuild.easyblocks.p.pytorch as pytorch
from easybuild.base.testing import TestCase
from easybuild.easyblocks.generic.cmakemake import det_cmake_version
from easybuild.easyblocks.generic.toolchain import Toolchain
from easybuild.easyblocks.tensorflow import det_binutils_bin_path
from easybuild.framework.easyblock import EasyBlock, get_easyblock_instance
from easybuild.framework.easyconfig import easyconfig
from easybuild.framework.easyconfig.easyconfig import process_easyconfig
from easybuild.tools import config
from easybuild.tools.build_log import EasyBuildError
from easybuild.tools.config import GENERAL_CLASS, build_option, get_module_syntax, update_build_option
from easybuild.tools.environment import modify_env
from easybuild.tools.filetools import adjust_permissions, mkdir, move_file, read_file, remove_dir, symlink, write_file
from easybuild.tools.modules import modules_tool
from easybuild.tools.options import set_tmpdir
from easybuild.tools.run import RunShellCmdResult
from easybuild.tools.toolchain.toolchain import RPATH_WRAPPERS_SUBDIR


CS_OCI_INDEX = 'application/vnd.oci.image.index.v1+json'
CS_OCI_MANIFEST = 'application/vnd.oci.image.manifest.v1+json'
# enough of enroot 4.2.0 for the wrappers: 'list', 'create', and 'start' printing the container's environment, made
# as runtime::_do_environ, common::envfmt and enroot-switchroot make it (with enroot's set -euo pipefail)
CS_FAKE_ENROOT = r'''#!/bin/bash
set -euo pipefail
case "$1" in list | create) exit 0 ;; start) shift ;; *) exit 1 ;; esac
# enroot's own settings join its environment (enroot.in)
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run}" ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-/data}"
conf= envs=()
while [ $# -gt 0 ]; do
    case "$1" in
        --conf) conf="$2"; shift 2 ;;
        --env) envs+=("$2"); shift 2 ;;
        --mount) shift 2 ;;
        *) break ;;
    esac
done
export ENROOT_ENVIRON="@DIR@/environment"
(
    [ -z "$conf" ] || source "$conf"
    # the image's /etc/environment, then the --conf file's environ(), then --env
    cat "@DIR@/image-environment" > "$ENROOT_ENVIRON"
    if declare -F environ > /dev/null; then
        environ >> "$ENROOT_ENVIRON"
    fi
    for env in ${envs[@]+"${envs[@]}"}; do
        awk '{ print /=/ ? $0 : $0"="ENVIRON[$0] }' <<< "$env"
    done >> "$ENROOT_ENVIRON"
)
# envfmt drops ill-formed and ENROOT_ lines; switchroot clears the environment and sets each line (a later one wins)
sed -e '/^[[:alpha:]_][[:alnum:]_]*=/!d' -e '/^ENROOT_/d' "$ENROOT_ENVIRON" |
    awk '{ i = index($0, "="); v[substr($0, 1, i - 1)] = substr($0, i + 1) }
         END { for (k in v) print "ENV " k "=" v[k] }'
'''


class _CSFakeRegistry:
    """Minimal OCI distribution API on 127.0.0.1 (bearer token, manifests, blobs) for ContainerStack tests."""

    def __init__(self, repo='team/tiny'):
        self.repo = repo
        self.manifests = {}
        self.blobs = {}
        self.requests = []
        self.token_auth = []
        self.tamper_manifest = False
        self.tamper_blobs = False
        self.token = 'T0K'
        self.reject_tokens = False
        self.redirect_blobs = False
        self.cdn_auth = []
        registry = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                registry.requests.append(self.path)
                if self.path.startswith('/token'):
                    registry.token_auth.append(self.headers.get('Authorization'))
                    return self._send(200, json.dumps({'token': registry.token}).encode(), 'application/json')
                if self.path.startswith('/cdn/'):
                    registry.cdn_auth.append(self.headers.get('Authorization'))
                    if self.headers.get('Authorization'):
                        return self._send(400, b'{}', 'application/json')
                    return self._send(200, registry.blobs[self.path[len('/cdn/'):]], 'application/octet-stream')
                if registry.reject_tokens or self.headers.get('Authorization') != 'Bearer ' + registry.token:
                    challenge = 'Bearer realm="http://127.0.0.1:%d/token",service="fake",scope="repository:%s:pull"'
                    self.send_response(401)
                    self.send_header('WWW-Authenticate', challenge % (registry.port, registry.repo))
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return None
                prefix = '/v2/%s/' % registry.repo
                kind, ref = '', ''
                if self.path.startswith(prefix):
                    kind, _, ref = self.path[len(prefix):].partition('/')
                if kind == 'manifests' and ref in registry.manifests:
                    data, ctype = registry.manifests[ref]
                    return self._send(200, data + (b' ' if registry.tamper_manifest else b''), ctype)
                if kind == 'blobs' and ref in registry.blobs and registry.redirect_blobs:
                    self.send_response(307)
                    self.send_header('Location', 'http://127.0.0.1:%d/cdn/%s' % (registry.port, ref))
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return None
                if kind == 'blobs' and ref in registry.blobs:
                    data = registry.blobs[ref] + (b'x' if registry.tamper_blobs else b'')
                    return self._send(200, data, 'application/octet-stream')
                return self._send(404, b'{}', 'application/json')

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.port = self.server.server_address[1]
        self.address = '127.0.0.1:%d' % self.port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def add_blob(self, data):
        digest = 'sha256:' + hashlib.sha256(data).hexdigest()
        self.blobs[digest] = data
        return digest

    def add_manifest(self, doc, ctype, tag=None):
        data = json.dumps(doc).encode()
        digest = 'sha256:' + hashlib.sha256(data).hexdigest()
        self.manifests[digest] = (data, ctype)
        if tag:
            self.manifests[tag] = (data, ctype)
        return digest

    def blob_requests(self):
        return [r for r in self.requests if '/blobs/' in r]


def _cs_tiny_image(registry, tag='1.0', workdir='/work'):
    """Push a one-layer image (file etc/greeting) and a multi-arch index to the fake registry; return digests."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name, data in (('etc/greeting', b'hello\n'), ('./etc/hosts', b'127.0.0.1 localhost\n')):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o640
            tar.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo('var/empty')
        info.type, info.mode = tarfile.DIRTYPE, 0o555
        tar.addfile(info)
    layer = buf.getvalue()
    config = json.dumps({'architecture': 'amd64', 'os': 'linux', 'rootfs': {'type': 'layers', 'diff_ids': []},
                         'config': {'Env': ['PATH=/usr/bin:/bin', 'GREETING=hello world'], 'Cmd': ['sh'],
                                    'WorkingDir': workdir}}).encode()
    layer_digest = registry.add_blob(layer)
    config_digest = registry.add_blob(config)
    manifest = {'schemaVersion': 2, 'mediaType': CS_OCI_MANIFEST,
                'config': {'mediaType': 'application/vnd.oci.image.config.v1+json', 'digest': config_digest,
                           'size': len(config)},
                'layers': [{'mediaType': 'application/vnd.oci.image.layer.v1.tar+gzip', 'digest': layer_digest,
                            'size': len(layer)}]}
    manifest_digest = registry.add_manifest(manifest, CS_OCI_MANIFEST)
    index = {'schemaVersion': 2, 'mediaType': CS_OCI_INDEX, 'manifests': [
        {'mediaType': CS_OCI_MANIFEST, 'digest': manifest_digest, 'size': len(json.dumps(manifest)),
         'platform': {'os': 'linux', 'architecture': 'amd64'}},
        {'mediaType': CS_OCI_MANIFEST, 'digest': 'sha256:' + 'f' * 64, 'size': 1,
         'platform': {'os': 'unknown', 'architecture': 'unknown'}}]}
    index_digest = registry.add_manifest(index, CS_OCI_INDEX, tag=tag)
    return {'manifest': manifest_digest, 'index': index_digest, 'config': config_digest, 'layer': layer_digest}


def _cs_member(name, data=b'', kind=tarfile.REGTYPE, mode=0o644, linkname=''):
    """(TarInfo, data) for a layer member: a file with 'data', or a directory, symlink or whiteout."""
    info = tarfile.TarInfo(name)
    info.type, info.mode, info.linkname = kind, mode, linkname
    if kind == tarfile.REGTYPE:
        info.size = len(data)
        return info, io.BytesIO(data)
    return info, None


def _cs_layer(blobs, members, zstd=False):
    """Write a layer blob (gzip, or zstd via the zstd command) of _cs_member()s to blobs; return its descriptor."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        for info, data in members:
            tar.addfile(info, data)
    if zstd:
        blob = subprocess.run(['zstd', '-q', '-c'], input=buf.getvalue(), stdout=subprocess.PIPE, check=True).stdout
    else:
        blob = gzip.compress(buf.getvalue(), mtime=0)
    digest = containerstack.sha256_digest(blob)
    path = containerstack.blob_path(blobs, digest)
    mkdir(os.path.dirname(path), parents=True)
    with open(path, 'wb') as fh:
        fh.write(blob)
    return {'mediaType': 'application/vnd.oci.image.layer.v1.tar+%s' % ('zstd' if zstd else 'gzip'),
            'digest': digest, 'size': len(blob)}


class EasyBlockSpecificTest(TestCase):
    """ Baseclass for easyblock testcases """

    # initialize configuration (required for e.g. default modules_tool setting)
    eb_go = eboptions.parse_options(args=[])
    config.init(eb_go.options, eb_go.get_options_by_section('config'))
    build_options = {
        'suffix_modules_path': GENERAL_CLASS,
        'valid_module_classes': config.module_classes(),
        'valid_stops': [x[0] for x in EasyBlock.get_steps()],
    }
    config.init_build_options(build_options=build_options)
    set_tmpdir()
    del eb_go

    def setUp(self):
        """Test setup."""
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

        self.orig_environ = copy.deepcopy(os.environ)
        self.orig_pythonpackage_run_shell_cmd = pythonpackage.run_shell_cmd
        self.cs_orig = (containerstack.run_shell_cmd, containerstack.source_paths, containerstack.host_arch_key,
                        containerstack.disk_free, EasyBlock.sanity_check_step, EasyBlock.make_module_step)
        self.cs_orig_opts = {opt: build_option(opt) for opt in ('module_only', 'sanity_check_only', 'force')}
        # test.easyblocks.suite runs this after module.py, which leaves --module-only --force set
        for opt in self.cs_orig_opts:
            update_build_option(opt, False)
        self.cs_fail = {}
        self.cs_calls = []
        self.cs_squashed = {}
        os.environ['DOCKER_CONFIG'] = os.path.join(self.tmpdir, 'docker')
        for var in ('CONTAINERSTACK_REGISTRY_USER', 'CONTAINERSTACK_REGISTRY_PASSWORD', 'CS_GPU_DEVICE',
                    'ENROOT_SQUASH_OPTIONS', 'ENROOT_SYSCONF_PATH'):
            os.environ.pop(var, None)

    def tearDown(self):
        """Test cleanup."""
        (containerstack.run_shell_cmd, containerstack.source_paths, containerstack.host_arch_key,
         containerstack.disk_free, EasyBlock.sanity_check_step, EasyBlock.make_module_step) = self.cs_orig
        for opt, value in self.cs_orig_opts.items():
            update_build_option(opt, value)
        if os.path.isdir(self.tmpdir):
            adjust_permissions(self.tmpdir, stat.S_IWUSR, add=True, recursive=True)
        remove_dir(self.tmpdir)

        pythonpackage.run_shell_cmd = self.orig_pythonpackage_run_shell_cmd

        # restore original environment
        modify_env(os.environ, self.orig_environ, verbose=False)

        super().tearDown()

    def test_toolchain_external_modules(self):
        """Test use of Toolchain easyblock with external modules."""

        external_modules = ['gcc/8.3.0', 'openmpi/4.0.2', 'openblas/0.3.7', 'fftw/3.3.8', 'scalapack/2.0.2']
        external_modules_metadata = {
            # all metadata for gcc/8.3.0
            'gcc/8.3.0': {
                'name': ['GCC'],
                'version': ['8.3.0'],
                'prefix': '/software/gcc/8.3.0',
            },
            # only name/version for openmpi/4.0.2
            'openmpi/4.0.2': {
                'name': ['OpenMPI'],
                'version': ['4.0.2'],
            },
            # only name/prefix for openblas/0.3.7
            'openblas/0.3.7': {
                'name': ['OpenBLAS'],
                'prefix': '/software/openblas/0.3.7',
            },
            # only version/prefix for fftw/3.3.8 (no name)
            'fftw/3.3.8': {
                'version': ['3.3.8'],
                'prefix': '/software/fftw/3.3.8',
            },
            # no metadata for scalapack/2.0.2
        }

        # initialize configuration
        cleanup()
        eb_go = eboptions.parse_options(args=['--installpath=%s' % self.tmpdir])
        config.init(eb_go.options, eb_go.get_options_by_section('config'))
        build_options = {
            'external_modules_metadata': external_modules_metadata,
            'valid_module_classes': config.module_classes(),
        }
        config.init_build_options(build_options=build_options)
        set_tmpdir()
        del eb_go

        modtool = modules_tool()

        # make sure no $EBROOT* or $EBVERSION* environment variables are set in current environment
        for key in os.environ:
            if any(key.startswith(x) for x in ['EBROOT', 'EBVERSION']):
                del os.environ[key]

        # create dummy module file for each of the external modules
        test_mod_path = os.path.join(self.tmpdir, 'modules', 'all')
        for mod in external_modules:
            write_file(os.path.join(test_mod_path, mod), "#%Module")

        modtool.use(test_mod_path)

        # test easyconfig file to install toolchain that uses external modules,
        # and enables set_env_external_modules
        test_ec_path = os.path.join(self.tmpdir, 'test.eb')
        test_ec_txt = '\n'.join([
            "easyblock = 'Toolchain'",
            "name = 'test-toolchain'",
            "version = '1.2.3'",
            "homepage = 'https://example.com'",
            "description = 'just a test'",
            "toolchain = SYSTEM",
            "dependencies = [",
            "   ('gcc/8.3.0', EXTERNAL_MODULE),",
            "   ('openmpi/4.0.2', EXTERNAL_MODULE),",
            "   ('openblas/0.3.7', EXTERNAL_MODULE),",
            "   ('fftw/3.3.8', EXTERNAL_MODULE),",
            "   ('scalapack/2.0.2', EXTERNAL_MODULE),",
            "]",
            "set_env_external_modules = True",
            "moduleclass = 'toolchain'",
        ])
        write_file(test_ec_path, test_ec_txt)
        test_ec = process_easyconfig(test_ec_path)[0]

        # create easyblock & install module via run_all_steps
        tc_inst = get_easyblock_instance(test_ec)
        self.assertTrue(isinstance(tc_inst, Toolchain))
        self.mock_stdout(True)
        tc_inst.run_all_steps(False)
        self.mock_stdout(False)

        # make sure expected module file exists
        test_mod = os.path.join(test_mod_path, 'test-toolchain', '1.2.3')
        if get_module_syntax() == 'Lua':
            test_mod += '.lua'
        self.assertTrue(os.path.exists(test_mod))

        # load test-toolchain/1.2.3 module to get environment variable to check for defined
        modtool.load(['test-toolchain/1.2.3'])

        # check whether expected environment variables are defined
        self.assertEqual(os.environ.pop('EBROOTGCC'), '/software/gcc/8.3.0')
        self.assertEqual(os.environ.pop('EBVERSIONGCC'), '8.3.0')
        self.assertEqual(os.environ.pop('EBVERSIONOPENMPI'), '4.0.2')
        self.assertEqual(os.environ.pop('EBROOTOPENBLAS'), '/software/openblas/0.3.7')
        undefined_env_vars = [
            'EBROOTOPENMPI',  # no prefix in metadata
            'EBVERSIONOPENBLAS'  # no version in metadata
            'EBROOTFFTW', 'EBVERSIONFFTW',  # no name in metadata
            'EBROOTSCALAPACK', 'EBVERSIONSCALAPACK',  # no metadata
        ]
        for env_var in undefined_env_vars:
            self.assertTrue(os.getenv(env_var) is None)

        # make sure no unexpected $EBROOT* or $EBVERSION* environment variables were defined
        del os.environ['EBROOTTESTMINTOOLCHAIN']
        del os.environ['EBVERSIONTESTMINTOOLCHAIN']
        extra_eb_env_vars = []
        for key in os.environ:
            if any(key.startswith(x) for x in ['EBROOT', 'EBVERSION']):
                extra_eb_env_vars.append(key)
        self.assertEqual(extra_eb_env_vars, [])

    def test_det_cmake_version(self):
        """Tests for det_cmake_version function provided along with CMakeMake generic easyblock."""

        # set up fake 'cmake' command
        cmake_cmd = os.path.join(self.tmpdir, 'cmake')
        write_file(cmake_cmd, '#!/bin/bash\nexit 1')
        adjust_permissions(cmake_cmd, stat.S_IXUSR)

        os.environ['PATH'] = '%s:%s' % (self.tmpdir, os.getenv('PATH'))

        self.assertRaisesRegex(EasyBuildError, "Failed to determine CMake version", det_cmake_version)

        # if $EBVERSIONCMAKE is defined (by loaded CMake module), that's picked up
        os.environ['EBVERSIONCMAKE'] = '1.2.3'
        self.assertEqual(det_cmake_version(), '1.2.3')

        del os.environ['EBVERSIONCMAKE']

        # output of 'cmake --version' as produced by CMake 2.x < 2.4.0
        write_file(cmake_cmd, textwrap.dedent("""
        #!/bin/bash
        echo "CMake version 2.3.0"
        """))
        self.assertEqual(det_cmake_version(), '2.3.0')

        # output of 'cmake --version' as produced by CMake 2.x >= 2.4.0
        write_file(cmake_cmd, textwrap.dedent("""
        #!/bin/bash
        echo "cmake version 2.4.1"
        """))
        self.assertEqual(det_cmake_version(), '2.4.1')

        # output of 'cmake --version' as produced by CMake 3.x
        write_file(cmake_cmd, textwrap.dedent("""
        #!/bin/bash
        echo "cmake version 3.15.3"
        echo ""
        echo "CMake suite maintained and supported by Kitware (kitware.com/cmake)."
        """))
        self.assertEqual(det_cmake_version(), '3.15.3')

        # also consider release candidate versions
        write_file(cmake_cmd, textwrap.dedent("""
        #!/bin/bash
        echo "cmake version 1.2.3-rc4"
        """))
        self.assertEqual(det_cmake_version(), '1.2.3-rc4')

    def test_det_installed_python_packages(self):
        """
        Test det_installed_python_packages function providyed by PythonPackage easyblock
        """
        pkg1 = None
        res = python.det_installed_python_packages(python_cmd=sys.executable)
        # we can't make too much assumptions on which installed Python packages are found
        self.assertTrue(isinstance(res, list))
        if res:
            pkg1_name = res[0]
            self.assertTrue(isinstance(pkg1_name, str))

        res_detailed = python.det_installed_python_packages(python_cmd=sys.executable, names_only=False)
        self.assertTrue(isinstance(res_detailed, list))
        if res_detailed:
            pkg1 = res_detailed[0]
            self.assertTrue(isinstance(pkg1, dict))
            self.assertTrue(sorted(pkg1.keys()), ['name', 'version'])
            self.assertEqual(pkg1['name'], pkg1_name)
            regex = re.compile('^[0-9].*')
            ver = pkg1['version']
            self.assertTrue(regex.match(ver), f"Pattern {regex.pattern} matches for pkg version: {ver}")

        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            stderr = None
            if "pip list" in cmd:
                output = '[{"name": "example", "version": "1.2.3"}]'
                stderr = "DEPRECATION: Python 2.7 reached the end of its life on January 1st, 2020"
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=0, output=output, stderr=stderr, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)
        python.run_shell_cmd = mocked_run_shell_cmd_pip
        res = python.det_installed_python_packages(python_cmd=sys.executable)
        self.assertEqual(res, ['example'])

    def test_det_py_install_scheme(self):
        """Test det_py_install_scheme function provided by PythonPackage easyblock."""
        res = pythonpackage.det_py_install_scheme(sys.executable)
        self.assertTrue(isinstance(res, str))

        # symlink currently used python command to 'python', so we can also test det_py_install_scheme with default;
        # this is required because 'python' command may not be available
        symlink(sys.executable, os.path.join(self.tmpdir, 'python'))
        os.environ['PATH'] = '%s:%s' % (self.tmpdir, os.getenv('PATH'))

        res = pythonpackage.det_py_install_scheme()
        self.assertTrue(isinstance(res, str))

    def test_cargo_get_workspace_members(self):
        """Test get_workspace_members in the Cargo easyblock"""
        # Simple crate
        toml_text = textwrap.dedent("""
            [package]
            name = 'my_crate'
            version = "0.1.0"
            edition = "2021"
            description = 'desc'
            documentation = "url"
            license = "MIT"
        """)
        members = cargo._get_workspace_members(tomllib.loads(toml_text))
        self.assertIsNone(members)

        # Virtual manifest
        toml_text = textwrap.dedent("""
            [workspace]
            members = [
                "reqwest-middleware",
                "reqwest-tracing",
                "reqwest-retry",
            ]
        """)
        members = cargo._get_workspace_members(tomllib.loads(toml_text))
        self.assertEqual(members, ["reqwest-middleware", "reqwest-tracing", "reqwest-retry"])

        # Workspace (root is a package too)
        toml_text = textwrap.dedent("""
            [package]
            name = "nothing-linux-ui"
            version = "0.0.2"
            edition = "2021"
            authors = ["sn99"]

            [workspace]
            members = ["nothing", "src-tauri"]

            [dependencies]
            leptos = { version = "0.6", features = ["csr"] }
        """)
        members = cargo._get_workspace_members(tomllib.loads(toml_text))
        self.assertEqual(members, ["nothing", "src-tauri"])

    def test_cargo_merge_sub_crate(self):
        """Test merge_sub_crate in the Cargo easyblock"""
        crate_dir = Path(tempfile.mkdtemp())
        cargo_toml = crate_dir / 'Cargo.toml'
        ws_parsed = tomllib.loads("""
            [workspace]
            members = ["bar"]

            [workspace.package]
            version = "1.2.3"
            authors = ["Nice Folks"]
            description = "A short description of my package"
            documentation = "https://example.com/bar"

            [workspace.dependencies]
            regex = { version = "1.6.0", default-features = false, features = ["std"] }
            cc = "1.0.73"
            rand = "0.8.5"

            [workspace.lints.rust]
            unsafe_code = "forbid"
        """)
        cargo_toml.write_text("""
            [package]
            name = "bar"
            version.workspace = true
            authors.workspace = true
            description.workspace = true
            documentation.workspace = true

            # Unrelated line that looks like a workspace key
            dummy = "Uses regex=123 and regex = 456 and not foo.workspace = true"

            [dependencies]
            foo = { version = "42" }
            # Overwrite 'features' value
            regex = { workspace = true, features = ["unicode"] }

            [build-dependencies]
            cc.workspace = true

            [dev-dependencies]
            rand = { workspace = true }

            [lints]
            workspace = true
        """)
        cargo._merge_sub_crate(cargo_toml, ws_parsed)
        self.assertEqual(tomllib.loads(cargo_toml.read_text()), tomllib.loads("""
            [package]
            name = "bar"
            version = "1.2.3"
            authors = ["Nice Folks"]
            description = "A short description of my package"
            documentation = "https://example.com/bar"

            dummy = "Uses regex=123 and regex = 456 and not foo.workspace = true"

            [dependencies]
            foo = { version = "42" }
            regex = { version = "1.6.0", default-features = false, features = ["unicode"] }

            [build-dependencies]
            cc = "1.0.73"

            [dev-dependencies]
            rand = "0.8.5"

            [lints.rust]
            unsafe_code = "forbid"
        """))

        # Only dict-style workspace dependency
        cargo_toml.write_text("""
            [package]
            name = "bar"

            [dependencies]
            regex = { workspace = true }
        """)
        cargo._merge_sub_crate(cargo_toml, ws_parsed)
        self.assertEqual(tomllib.loads(cargo_toml.read_text()), tomllib.loads("""
            [package]
            name = "bar"

            [dependencies]
            regex = { version = "1.6.0", default-features = false, features = ["std"] }
        """))

    def test_handle_local_py_install_scheme(self):
        """Test handle_local_py_install_scheme function provided by PythonPackage easyblock."""

        # test with empty dir, should be fine
        pythonpackage.handle_local_py_install_scheme(self.tmpdir)
        self.assertEqual(os.listdir(self.tmpdir), [])

        # create normal structure (no 'local' subdir), shouldn't cause trouble
        bindir = os.path.join(self.tmpdir, 'bin')
        mkdir(bindir)
        write_file(os.path.join(bindir, 'test'), 'test')
        libdir = os.path.join(self.tmpdir, 'lib')
        pyshortver = '.'.join(str(x) for x in sys.version_info[:2])
        pylibdir = os.path.join(libdir, 'python' + pyshortver, 'site-packages')
        mkdir(pylibdir, parents=True)
        write_file(os.path.join(pylibdir, 'test.py'), "import os")

        pythonpackage.handle_local_py_install_scheme(self.tmpdir)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ['bin', 'lib'])

        # move bin + lib into local/, check whether expected symlinks are created
        local_subdir = os.path.join(self.tmpdir, 'local')
        mkdir(local_subdir)
        for subdir in (bindir, libdir):
            move_file(subdir, os.path.join(local_subdir, os.path.basename(subdir)))
        self.assertEqual(os.listdir(self.tmpdir), ['local'])

        pythonpackage.handle_local_py_install_scheme(self.tmpdir)
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ['bin', 'lib', 'local'])
        self.assertTrue(os.path.islink(bindir) and os.path.samefile(bindir, os.path.join(local_subdir, 'bin')))
        self.assertTrue(os.path.islink(libdir) and os.path.samefile(libdir, os.path.join(local_subdir, 'lib')))
        self.assertTrue(os.path.exists(os.path.join(bindir, 'test')))
        local_test_py = os.path.join(libdir, 'python' + pyshortver, 'site-packages', 'test.py')
        self.assertTrue(os.path.exists(local_test_py))

    def test_partial_normalize_pip(self):
        """Test partial_normalize_pip function provided by EB_Python easyblock."""

        self.assertEqual(python.partial_normalize_pip('friendly-bard'), 'friendly-bard')  # normalized form
        self.assertEqual(python.partial_normalize_pip('Friendly-Bard'), 'friendly-bard')  # uppercase -> lowercase
        self.assertEqual(python.partial_normalize_pip('friendly_bard'), 'friendly-bard')  # underscore -> hyphen
        self.assertEqual(python.partial_normalize_pip('friendly.bard'), 'friendly.bard')  # dots are not normalized
        # multiple consecutive hyphens are merged into a single hyphen
        self.assertEqual(python.partial_normalize_pip('friendly--bard'), 'friendly-bard')
        # multiple consecutive underscores are merged and converted into a single hyphen
        self.assertEqual(python.partial_normalize_pip('friendly__bard'), 'friendly-bard')
        # multiple consecutive underscores and hyphens are merged and converted into a single hyphen
        self.assertEqual(python.partial_normalize_pip('FrIeNdLy--__--bArD'), 'friendly-bard')

    def test_run_pip_check(self):
        """Test run_pip_check function provided by EB_Python easyblock."""

        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip check" in cmd:
                output = "No broken requirements found."
            elif "pip --version" in cmd:
                output = "pip 20.0"
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=0, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        with self.mocked_stdout_stderr():
            python.run_pip_check(python_cmd=sys.executable)

        # inject all possible errors
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip check" in cmd:
                output = "foo-1.2.3 requires bar-4.5.6, which is not installed."
                exit_code = 1
            elif "pip --version" in cmd:
                output = "pip 20.0"
                exit_code = 0
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=exit_code, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        error_pattern = '\n'.join([
            "pip check.*failed.*",
            "foo.*requires.*bar.*not installed.*",
        ])
        with self.mocked_stdout_stderr():
            self.assertRaisesRegex(EasyBuildError, error_pattern, python.run_pip_check,
                                   python_cmd=sys.executable)

        # invalid pip version
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            return RunShellCmdResult(cmd=cmd, exit_code=0, output="1.2.3", stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        error_pattern = "Failed to determine pip version!"
        self.assertRaisesRegex(EasyBuildError, error_pattern, python.run_pip_check, python_cmd=sys.executable)

    def test_run_pip_list(self):
        """Test run_pip_list function provided by EB_Python easyblock."""

        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip list" in cmd:
                output = '[{"name": "example", "version": "1.2.3"}]'
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=0, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        with self.mocked_stdout_stderr():
            python.run_pip_list([], python_cmd=sys.executable)

        # test ignored unversioned Python packages
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip list" in cmd:
                output = '[{"name": "zero", "version": "0.0.0"}, {"name": "example-pkg", "version": "1.2.3"}]'
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=0, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        with self.mocked_stdout_stderr():
            python.run_pip_list([('example_pkg', '1.2.3')], python_cmd=sys.executable, unversioned_packages=('zero', ))

        with self.mocked_stdout_stderr():
            python.run_pip_list([('example.pkg', '1.2.3')], python_cmd=sys.executable, unversioned_packages={'zero'})

        # inject all possible errors with unversioned packages
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip list" in cmd:
                output = '[{"name": "example", "version": "1.2.3"}, {"name": "wrong", "version": "0.0.0"}]'
                exit_code = 0
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=exit_code, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        error_pattern = '\n'.join([
            r"Package 'example'.*version of 1\.2\.3 which is valid.*",
            "Package 'nosuchpkg' in unversioned_packages was not found in the installed packages.*",
            r".*not installed correctly.*version of '0\.0\.0':",
            "wrong",
        ])
        with self.mocked_stdout_stderr():
            self.assertRaisesRegex(EasyBuildError, error_pattern, python.run_pip_list, [],
                                   python_cmd=sys.executable, unversioned_packages=['example', 'nosuchpkg'])

        # inject errors with mismatched packages name or version
        def mocked_run_shell_cmd_pip(cmd, **kwargs):
            if "pip list" in cmd:
                output = '[{"name": "example", "version": "1.2.3"}, {"name": "wrong-version", "version": "1.1.1"}]'
                exit_code = 0
            else:
                # unexpected command
                return None

            return RunShellCmdResult(cmd=cmd, exit_code=exit_code, output=output, stderr=None, work_dir=None,
                                     out_file=None, err_file=None, cmd_sh=None, thread_id=None, task_id=None)

        python.run_shell_cmd = mocked_run_shell_cmd_pip
        error_pattern = '\n'.join([
            r"The following Python packages were likely specified with a wrong name because they are missing.*",
            r"wrong-name.*",
            r"The following Python packages were likely specified with a wrong version.*",
            r"wrong-version 5.6.7.*",
        ])
        with self.mocked_stdout_stderr():
            self.assertRaisesRegex(EasyBuildError, error_pattern, python.run_pip_list,
                                   [('wrong_name', '1.2.3'), ('wrong_version', '5.6.7')],
                                   python_cmd=sys.executable, strict_check=True)

    def test_symlink_dist_site_packages(self):
        """Test symlink_dist_site_packages provided by PythonPackage easyblock."""
        pyshortver = '.'.join(str(x) for x in sys.version_info[:2])
        pylibdir_lib_dist = os.path.join('lib', 'python' + pyshortver, 'dist-packages')
        pylibdir_lib64_site = os.path.join('lib64', 'python' + pyshortver, 'site-packages')
        pylibdirs = [pylibdir_lib_dist, pylibdir_lib64_site]

        # first test on empty dir
        pythonpackage.symlink_dist_site_packages(self.tmpdir, pylibdirs)
        self.assertEqual(os.listdir(self.tmpdir), [])

        # check intended usage: dist-packages exists, site-packages doesn't => symlink created
        lib64_site_path = os.path.join(self.tmpdir, pylibdir_lib_dist)
        mkdir(os.path.join(self.tmpdir, pylibdir_lib_dist), parents=True)
        # also create (empty) site-packages directory, which should get replaced by a symlink to dist-packages
        mkdir(os.path.join(self.tmpdir, os.path.dirname(pylibdir_lib_dist), 'site-packages'), parents=True)

        lib64_site_path = os.path.join(self.tmpdir, pylibdir_lib64_site)
        mkdir(lib64_site_path, parents=True)
        # populate site-packages under lib64, because it'll get removed if empty
        write_file(os.path.join(lib64_site_path, 'test.py'), "import os")

        pythonpackage.symlink_dist_site_packages(self.tmpdir, pylibdirs)

        # check for expected directory structure
        self.assertEqual(sorted(os.listdir(self.tmpdir)), ['lib', 'lib64'])
        path = os.path.join(self.tmpdir, 'lib')
        self.assertEqual(os.listdir(path), ['python' + pyshortver])
        path = os.path.join(path, 'python' + pyshortver)
        self.assertEqual(sorted(os.listdir(path)), ['dist-packages', 'site-packages'])

        # check for site-packages -> dist-packages symlink
        dist_pkgs = os.path.join(path, 'dist-packages')
        self.assertTrue(os.path.isdir(dist_pkgs))
        self.assertFalse(os.path.islink(dist_pkgs))
        site_pkgs = os.path.join(path, 'site-packages')
        self.assertTrue(os.path.isdir(site_pkgs))
        self.assertTrue(os.path.islink(site_pkgs))

        # if (non-empty) site-packages dir was there, no changes were made
        lib64_path = os.path.dirname(lib64_site_path)
        self.assertEqual(sorted(os.listdir(lib64_path)), ['site-packages'])
        self.assertTrue(os.path.isdir(lib64_site_path))
        self.assertFalse(os.path.islink(lib64_site_path))

    def test_det_binutils_bin_path(self):
        """Test det_binutils_bin_path provided by TensorFlow easyblock."""
        binutils_bin_path = det_binutils_bin_path()
        self.assertTrue(os.path.join(binutils_bin_path, 'ld'))
        self.assertFalse(RPATH_WRAPPERS_SUBDIR in binutils_bin_path)

        wrappers_dir = os.path.join(self.tmpdir, 'fake_wrappers', RPATH_WRAPPERS_SUBDIR)

        # put fake wrappers in place for a couple of binutils command
        binutils_cmds = ('as', 'ld', 'nm', 'objdump')
        for cmd in binutils_cmds:
            wrapper_dir = os.path.join(wrappers_dir, '%s.wrapper' % cmd)
            os.environ['PATH'] = wrapper_dir + ':' + os.getenv('PATH')
            fake_wrapper = os.path.join(wrapper_dir, cmd)
            # fake contents for RPATH wrapper script, enough to fool Toolchain.is_rpath_wrapper
            fake_wrapper_txt = '"$RPATH_ARGS_PY" "$CMD"'
            write_file(fake_wrapper, fake_wrapper_txt)
            adjust_permissions(fake_wrapper, stat.S_IXUSR)

        # if $EBROOTBINUTILS is set, binutils commands to consider is determined by contents of $EBROOTBINUTILS/bin
        binutils_root = os.path.join(self.tmpdir, 'binutils_root')
        for cmd in binutils_cmds[:2]:
            cmd_path = os.path.join(binutils_root, 'bin', cmd)
            write_file(cmd_path, '#!/bin/bash\necho %s' % cmd)
            adjust_permissions(cmd_path, stat.S_IXUSR)
        os.environ['EBROOTBINUTILS'] = binutils_root

        binutils_bin_path = det_binutils_bin_path()
        self.assertEqual(os.path.basename(binutils_bin_path), RPATH_WRAPPERS_SUBDIR)
        self.assertEqual(sorted(os.listdir(binutils_bin_path)), ['as', 'ld'])
        for cmd in binutils_cmds[:2]:
            cmd_path = os.path.join(binutils_bin_path, cmd)
            self.assertTrue(os.path.islink(cmd_path))
            expected_target = os.path.join(wrappers_dir, '%s.wrapper' % cmd, cmd)
            self.assertEqual(os.path.realpath(cmd_path), os.path.realpath(expected_target))

        del os.environ['EBROOTBINUTILS']

        # if $EBROOTBINUTILS is not set, a pre-defined list of known binutils commands is used (KNOWN_BINUTILS constant)
        binutils_bin_path = det_binutils_bin_path()
        self.assertEqual(os.path.basename(binutils_bin_path), RPATH_WRAPPERS_SUBDIR)
        found_cmds = os.listdir(binutils_bin_path)
        self.assertTrue(all(x in found_cmds for x in binutils_cmds))
        for cmd in binutils_cmds:
            cmd_path = os.path.join(binutils_bin_path, cmd)
            self.assertTrue(os.path.islink(cmd_path))
            expected_target = os.path.join(wrappers_dir, '%s.wrapper' % cmd, cmd)
            self.assertEqual(os.path.realpath(cmd_path), os.path.realpath(expected_target))

    def test_translate_lammps_version(self):
        """Test translate_lammps_version function from LAMMPS easyblock"""
        lammps_versions = {
            '23Jun2022': '2022.06.23',
            '2Aug2023_update2': '2023.08.02.2',
            '29Aug2024': '2024.08.29',
            '29Aug2024_update2': '2024.08.29.2',
            '28Oct2024': '2024.10.28',
        }
        for key, expected_version in lammps_versions.items():
            self.assertEqual(lammps.translate_lammps_version(key), expected_version)

        version_file = os.path.join(self.tmpdir, 'src', 'version.h')
        version_txt = '\n'.join([
            '#define LAMMPS_VERSION "2 Apr 2025"',
            '#define LAMMPS_UPDATE "Development"',
        ])
        write_file(version_file, version_txt)

        self.assertEqual(lammps.translate_lammps_version('d3adb33f', path=self.tmpdir), '2025.04.02')
        self.assertEqual(lammps.translate_lammps_version('devel', path=self.tmpdir), '2025.04.02')

        version_file = os.path.join(self.tmpdir, 'src', 'version.h')
        version_txt = '\n'.join([
            '#define LAMMPS_VERSION "2 Apr 2025"',
            '#define LAMMPS_UPDATE "Update 3"',
        ])
        write_file(version_file, version_txt)

        self.assertEqual(lammps.translate_lammps_version('d3adb33f', path=self.tmpdir), '2025.04.02.3')

    def test_pytorch_test_log_parsing(self):
        """Verify parsing of XML files produced by PyTorch tests."""
        TestState = pytorch.TestState

        test_log_dir = Path(__file__).parent.parent / 'pytorch_test_logs'

        results = pytorch.get_test_results(test_log_dir / 'test-reports')
        results2 = pytorch.get_test_results(test_log_dir)
        self.assertEqual(results.keys(), results2.keys())
        for name, suite in results.items():
            self.assertEqual((name, suite.summary), (name, results2[name].summary))
        del results2

        self.assertEqual(len(results), 16)

        # 2 small test suites used as a smoke test using a most features
        self.assertIn('backends/xeon/test_launch', results)
        suite = results['backends/xeon/test_launch']
        self.assertEqual((suite.errors, suite.failures, suite.num_tests, suite.skipped), (1, 2, 8, 3))
        # Failure in one file, success in the other --> Success
        self.assertEqual(suite['TestTorchrun.test_cpu_info'].state, TestState.SUCCESS)
        # New in 2nd file
        self.assertEqual(suite['TestTorchrun.test_multi_threads'].state, TestState.SUCCESS)
        self.assertEqual(suite['TestTorchrun.test_reshape_cpu_float64'].state, TestState.FAILURE)
        self.assertEqual(suite['TestTorchrun.test_foo'].state, TestState.SKIPPED)
        self.assertEqual(suite['TestTorchrun.test_bar'].state, TestState.ERROR)
        self.assertEqual(suite.get_errored_tests(), ['TestTorchrun.test_bar'])
        self.assertEqual(suite.get_failed_tests(), ['TestTorchrun.test_reshape_cpu_float64', 'TestTorchrun.test_baz'])
        self.assertIn('test_autoload', results)
        suite = results['test_autoload']
        self.assertEqual((suite.errors, suite.failures, suite.num_tests, suite.skipped), (0, 0, 2, 1))
        self.assertEqual(suite['TestBackendAutoload.test_autoload'].state, TestState.SUCCESS)
        self.assertEqual(suite['TestBackendAutoload.test_unload'].state, TestState.SKIPPED)

        # Verify summaries which should be enough to catch most issues
        report = '\n'.join(sorted(f'{suite.name}: {suite.summary}' for suite in results.values()))
        self.assertEqual(report, textwrap.dedent("""
            backends/xeon/test_launch: 2 failed, 2 passed, 3 skipped, 1 errors
            dist-gloo-init-env/distr/algorithms/quantization/test_quantization: 0 failed, 1 passed, 0 skipped, 0 errors
            dist-gloo-init-file/distr/algorithms/quantization/test_quantization: 0 failed, 1 passed, 0 skipped, 0 errors
            dist-nccl-init-env/distr/algorithms/quantization/test_quantization: 0 failed, 1 passed, 0 skipped, 0 errors
            dist-nccl-init-file/distr/algorithms/quantization/test_quantization: 0 failed, 1 passed, 0 skipped, 0 errors
            dist/foo/bar: 0 failed, 4 passed, 0 skipped, 0 errors
            distributed/_composable/test_composability/test_pp_composability: 0 failed, 2 passed, 0 skipped, 0 errors
            distributed/tensor/test_dtensor_ops: 0 failed, 2 passed, 2 skipped, 0 errors
            dynamo/test_dynamic_shapes: 3 failed, 14 passed, 0 skipped, 0 errors
            dynamo/test_misc: 3 failed, 11 passed, 0 skipped, 0 errors
            inductor/test_aot_inductor_arrayref: 2 failed, 0 passed, 0 skipped, 0 errors
            inductor/test_cudagraph_trees: 1 failed, 0 passed, 0 skipped, 0 errors
            jit/test_builtins: 0 failed, 1 passed, 0 skipped, 0 errors
            test_autoload: 0 failed, 1 passed, 1 skipped, 0 errors
            test_nestedtensor: 3 failed, 2 passed, 3 skipped, 1 errors
            test_quantization: 0 failed, 12 passed, 5 skipped, 0 errors
        """).strip())
        tests = '\n'.join(sorted(f'{test.name}: {test.state.value}'
                                 for suite in results.values()
                                 for test in suite.get_tests()))
        self.assertEqual(tests, textwrap.dedent("""
            AOTInductorTestABICompatibleCpuWithStackAllocation.test_fail_and_skip: failure
            AOTInductorTestABICompatibleCpuWithStackAllocation.test_skip_and_fail: failure
            ComposabilityTest.test_pass_on_rerun_different_classname: success
            ComposabilityTest.test_pp_and_dcp: success
            CudaGraphTreeTests.test_workspace_allocation_error: failure
            DistQuantizationTests.test_all_gather_fp16: success
            DistQuantizationTests.test_all_gather_fp16: success
            DistQuantizationTests.test_all_gather_fp16: success
            DistQuantizationTests.test_all_gather_fp16: success
            DynamicShapesCtxManagerTests.test_autograd_profiler_dynamic_shapes: success
            DynamicShapesCtxManagerTests.test_generic_context_manager_with_graph_break_dynamic_shapes: success
            DynamicShapesCtxManagerTests.test_generic_ctx_manager_with_graph_break_dynamic_shapes: success
            DynamicShapesMiscTests.test_outside_linear_module_free_dynamic_shapes: failure
            DynamicShapesMiscTests.test_packaging_version_parse_dynamic_shapes: success
            DynamicShapesMiscTests.test_pair_dynamic_shapes: success
            DynamicShapesMiscTests.test_param_shape_binops_dynamic_shapes: success
            DynamicShapesMiscTests.test_parameter_free_dynamic_shapes: failure
            DynamicShapesMiscTests.test_patched_builtin_functions_dynamic_shapes: success
            DynamicShapesMiscTests.test_proxy_frozen_dataclass_dynamic_shapes: success
            DynamicShapesMiscTests.test_pt2_compliant_ops_are_allowed_dynamic_shapes: success
            DynamicShapesMiscTests.test_pt2_compliant_overload_dynamic_shapes: success
            DynamicShapesMiscTests.test_pure_python_accumulate_dynamic_shapes: success
            DynamicShapesMiscTests.test_py_guards_mark_dynamic_dynamic_shapes: success
            DynamicShapesMiscTests.test_python_slice_dynamic_shapes: success
            DynamicShapesMiscTests.test_pytree_tree_flatten_unflatten_dynamic_shapes: success
            DynamicShapesMiscTests.test_pytree_tree_leaves_dynamic_shapes: failure
            MiscTests.test_fail_then_skip2: failure
            MiscTests.test_fail_then_skip: failure
            MiscTests.test_fail_then_xfail2: success
            MiscTests.test_fail_then_xfail: success
            MiscTests.test_packaging_version_parse: success
            MiscTests.test_pair: success
            MiscTests.test_param_shape_binops: success
            MiscTests.test_parameter_free: failure
            MiscTests.test_pytree_tree_map: success
            MiscTests.test_shape_env_no_recording: success
            MiscTests.test_shape_env_recorded_function_fallback: success
            MiscTests.test_yield_from_in_a_loop: success
            TestBackendAutoload.test_autoload: success
            TestBackendAutoload.test_unload: skipped
            TestBuiltins.test_name: success
            TestCustomFunction.test_autograd_function_with_matmul_folding_at_output: success
            TestDTensorOpsCPU.test_dtensor_op_db_H_cpu_float16: success
            TestDTensorOpsCPU.test_dtensor_op_db_H_cpu_float32: success
            TestDTensorOpsCPU.test_dtensor_op_db_H_cpu_float64: skipped
            TestDTensorOpsCPU.test_dtensor_op_db_H_cpu_int8: skipped
            TestDynamicQuantizedOps.test_qrnncell: success
            TestFakeQuantizeOps.test_backward_per_channel: skipped
            TestFakeQuantizeOps.test_backward_per_channel_cachemask_cpu: success
            TestFakeQuantizeOps.test_backward_per_channel_cachemask_cuda: success
            TestName.test_bar: success
            TestNestedTensor.test_bmm_cuda_gpu_float16: failure
            TestNestedTensor.test_bmm_cuda_gpu_float32: failure
            TestNestedTensor.test_bmm_cuda_gpu_float64: error
            TestNestedTensor.test_cat: success
            TestNestedTensor.test_copy_: success
            TestNestedTensor.test_reshape_cpu_float16: skipped
            TestNestedTensor.test_reshape_cpu_float32: skipped
            TestNestedTensor.test_reshape_cpu_float64: failure
            TestNestedTensorSubclassCPU.test_linear_backward_memory_usage_cpu_float32: skipped
            TestNumericDebugger.test_quantize_pt2e_preserve_handle: success
            TestNumericDebugger.test_re_export_preserve_handle: success
            TestPadding.test_reflection_pad1d: success
            TestQuantizedConv.test_conv_reorder_issue_onednn: success
            TestQuantizedConv.test_conv_transpose_reorder_issue_onednn: success
            TestQuantizedFunctionalOps.test_relu_api: success
            TestQuantizedLinear.test_qlinear_cudnn: skipped
            TestQuantizedLinear.test_qlinear_gelu_pt2e: success
            TestQuantizedOps.test_adaptive_avg_pool2d_nhwc: success
            TestQuantizedOps.test_adaptive_avg_pool: skipped
            TestQuantizedOps.test_qadd_relu_cudnn: skipped
            TestQuantizedOps.test_qadd_relu_cudnn_nhwc: skipped
            TestQuantizedOps.test_qadd_relu_different_qparams: success
            TestTorchrun.test_bar: error
            TestTorchrun.test_baz: failure
            TestTorchrun.test_cpu_info: success
            TestTorchrun.test_foo2: skipped
            TestTorchrun.test_foo3: skipped
            TestTorchrun.test_foo: skipped
            TestTorchrun.test_multi_threads: success
            TestTorchrun.test_reshape_cpu_float64: failure
            TestTracer.test_jit_save: success
            test_2.test_func3: success
            test_foo.TestBar.test_func2: success
            test_foo.TestName.test_func1: success
        """).strip())

        #  Some error cases
        error_log_dir = test_log_dir / 'faulty-reports'

        self.assertRaisesRegex(ValueError, "<testsuites> or <testsuite>",
                               pytorch.get_test_results, error_log_dir / 'root')
        self.assertRaisesRegex(ValueError, "Failed to parse",
                               pytorch.get_test_results, error_log_dir / 'invalid_xml')
        self.assertRaisesRegex(ValueError, "multiple reported files",
                               pytorch.get_test_results, error_log_dir / 'multi_file')
        self.assertRaisesRegex(ValueError, "Path from folder and filename should be equal",
                               pytorch.get_test_results, error_log_dir / 'different_file_name')
        self.assertRaisesRegex(ValueError, "Unexpected file attribute",
                               pytorch.get_test_results, error_log_dir / 'file_attribute')
        self.assertRaisesRegex(ValueError, "Invalid state",
                               pytorch.get_test_results, error_log_dir / 'skip_and_failed')
        self.assertRaisesRegex(ValueError, "no test",
                               pytorch.get_test_results, error_log_dir / 'no_tests')
        self.assertRaisesRegex(ValueError, "Invalid test count",
                               pytorch.get_test_results, error_log_dir / 'consistency')
        self.assertRaisesRegex(ValueError, "Duplicate test",
                               pytorch.get_test_results, error_log_dir / 'duplicate')

    def test_containerstack_select_digest(self):
        """select_digest() returns the host-arch digest and explains every way a recipe can get it wrong."""
        good = 'sha256:' + 'a' * 64
        self.assertEqual(containerstack.select_digest({'x86_64': good}, 'x86_64'), good)
        self.assertRaisesRegex(EasyBuildError, "no digest for host architecture aarch64; recipe supports: x86_64",
                               containerstack.select_digest, {'x86_64': good}, 'aarch64')
        self.assertRaisesRegex(EasyBuildError, "recipe supports: none", containerstack.select_digest, {}, 'x86_64')
        self.assertRaisesRegex(EasyBuildError, "not of the form sha256:<64 hex>",
                               containerstack.select_digest, {'x86_64': 'sha256:abc'}, 'x86_64')
        self.assertRaisesRegex(EasyBuildError, "not of the form sha256:<64 hex>",
                               containerstack.select_digest, {'x86_64': '26.10-py3'}, 'x86_64')

    def test_containerstack_host_arch_key(self):
        """EasyBuild's architecture names map to the recipe's digest keys; others are refused."""
        orig = containerstack.get_cpu_architecture
        try:
            for arch, key in ((containerstack.X86_64, 'x86_64'), (containerstack.AARCH64, 'aarch64')):
                containerstack.get_cpu_architecture = lambda arch=arch: arch
                self.assertEqual(containerstack.host_arch_key(), key)
            containerstack.get_cpu_architecture = lambda: 'POWER'
            self.assertRaisesRegex(EasyBuildError, "unsupported host architecture POWER", containerstack.host_arch_key)
        finally:
            containerstack.get_cpu_architecture = orig

    def test_containerstack_split_image(self):
        """Image references split into registry and repository; tags and digests are refused."""
        split = containerstack.split_image
        self.assertEqual(split('nvcr.io/nvidia/pytorch'), ('nvcr.io', 'nvidia/pytorch'))
        self.assertEqual(split('python'), ('registry-1.docker.io', 'library/python'))
        self.assertEqual(split('docker.io/library/python'), ('registry-1.docker.io', 'library/python'))
        self.assertEqual(split('vllm/vllm-openai'), ('registry-1.docker.io', 'vllm/vllm-openai'))
        self.assertEqual(split('127.0.0.1:5000/team/tiny'), ('127.0.0.1:5000', 'team/tiny'))
        self.assertEqual(split('localhost/tiny'), ('localhost', 'tiny'))
        for bad in ('nvcr.io/nvidia/pytorch:26.10-py3', 'python@sha256:' + 'a' * 64):
            self.assertRaisesRegex(EasyBuildError, "must not contain a tag or digest", split, bad)
        self.assertEqual(containerstack.env_var_prefix('PyTorch-container'), 'PYTORCHCONTAINER')
        self.assertEqual(containerstack.env_var_prefix('vLLM-container'), 'VLLMCONTAINER')

    def _cs_registry(self, workdir='/work'):
        """Fake registry with the tiny image, and a client for it."""
        reg = _CSFakeRegistry()
        self.addCleanup(reg.close)
        img = _cs_tiny_image(reg, workdir=workdir)
        return reg, img, containerstack.RegistryClient(reg.address, reg.repo, scheme='http')

    def test_containerstack_registry_token_flow(self):
        """Anonymous bearer-token flow: 401 challenge, token request, manifest by digest; the token is reused."""
        reg, img, client = self._cs_registry()
        data, media_type = client.fetch_manifest(img['manifest'])
        self.assertEqual(media_type, CS_OCI_MANIFEST)
        self.assertEqual(json.loads(data)['layers'][0]['digest'], img['layer'])
        client.fetch_manifest(img['manifest'])
        self.assertEqual(len([r for r in reg.requests if r.startswith('/token')]), 1)
        self.assertEqual(reg.token_auth, [None])

    def test_containerstack_registry_rejects_index(self):
        """Pinning the multi-arch index digest instead of a per-arch manifest digest fails with advice."""
        _, img, client = self._cs_registry()
        self.assertRaisesRegex(EasyBuildError, "is a multi-arch index, not a per-architecture manifest",
                               client.fetch_manifest, img['index'])

    def test_containerstack_registry_rejects_wrong_manifest(self):
        """A registry that returns different bytes than the requested digest is refused."""
        reg, img, client = self._cs_registry()
        reg.tamper_manifest = True
        self.assertRaisesRegex(EasyBuildError, "not the requested %s" % img['manifest'],
                               client.fetch_manifest, img['manifest'])

    def test_containerstack_registry_errors(self):
        """HTTP errors and unreachable registries produce messages naming the URL or host."""
        _, _, client = self._cs_registry()
        self.assertRaisesRegex(EasyBuildError, "returned HTTP 404", client.fetch_manifest, 'sha256:' + '0' * 64)
        dead = containerstack.RegistryClient('127.0.0.1:9', 'x/y', scheme='http', timeout=5)
        self.assertRaisesRegex(EasyBuildError, "cannot reach", dead.fetch_manifest, 'sha256:' + '0' * 64)

    def test_containerstack_registry_credentials(self):
        """Credentials come from the environment or $DOCKER_CONFIG/config.json, never from the easyconfig."""
        reg, img, _ = self._cs_registry()
        self.assertEqual(containerstack.registry_credentials(reg.address), None)
        auth = base64.b64encode(b'alice:s3cret').decode()
        write_file(os.path.join(self.tmpdir, 'docker', 'config.json'),
                   json.dumps({'auths': {reg.address: {'auth': auth}}}))
        self.assertEqual(containerstack.registry_credentials(reg.address), ('alice', 's3cret'))
        os.environ['CONTAINERSTACK_REGISTRY_USER'] = 'bob'
        os.environ['CONTAINERSTACK_REGISTRY_PASSWORD'] = 'pw'
        client = containerstack.RegistryClient(reg.address, reg.repo, scheme='http')
        client.fetch_manifest(img['manifest'])
        self.assertEqual(reg.token_auth, ['Basic ' + base64.b64encode(b'bob:pw').decode()])

    def test_containerstack_platform_digests(self):
        """platform_digests() maps a tag's index to per-arch manifest digests (linux only)."""
        _, img, client = self._cs_registry()
        self.assertEqual(client.platform_digests('1.0'), {'x86_64': img['manifest']})
        self.assertRaisesRegex(EasyBuildError, "single-architecture image", client.platform_digests, img['manifest'])

    def test_containerstack_registry_token_refresh(self):
        """An expired token is dropped and renewed once; a registry that never accepts the token still fails."""
        reg, img, client = self._cs_registry()
        client.fetch_manifest(img['manifest'])
        self.assertEqual(len([r for r in reg.requests if r.startswith('/token')]), 1)
        reg.token = 'T1K'
        data, _ = client.fetch_manifest(img['manifest'])
        self.assertEqual(json.loads(data)['layers'][0]['digest'], img['layer'])
        self.assertEqual(len([r for r in reg.requests if r.startswith('/token')]), 2)
        self.assertEqual(client.token, 'T1K')
        reg.reject_tokens = True
        self.assertRaisesRegex(EasyBuildError, "returned HTTP 401", client.fetch_manifest, img['manifest'])
        self.assertEqual(len([r for r in reg.requests if r.startswith('/token')]), 3)
        fresh = containerstack.RegistryClient(reg.address, reg.repo, scheme='http')
        self.assertRaisesRegex(EasyBuildError, "returned HTTP 401", fresh.fetch_manifest, img['manifest'])
        self.assertEqual(len([r for r in reg.requests if r.startswith('/token')]), 4)

    def test_containerstack_registry_blob_redirect(self):
        """Blobs follow a registry redirect to a CDN, and the bearer token is not sent to the CDN."""
        reg, img, client = self._cs_registry()
        reg.redirect_blobs = True
        dest = os.path.join(self.tmpdir, 'layer.tar.gz')
        client.fetch_blob(img['layer'], dest)
        self.assertEqual(containerstack.file_sha256(dest), img['layer'])
        self.assertFalse(os.path.exists(dest + '.part'))
        self.assertEqual(reg.cdn_auth, [None])

    def test_containerstack_fetch_and_verify(self):
        """fetch_oci_blobs() stores manifest, config and layers by digest; verify_oci_blobs() re-checks them."""
        _, img, client = self._cs_registry()
        blobs = os.path.join(self.tmpdir, 'blobs')
        manifest = containerstack.fetch_oci_blobs(client, img['manifest'], blobs)
        self.assertEqual(manifest['config']['digest'], img['config'])
        expected = sorted(d.split(':')[1] for d in (img['manifest'], img['config'], img['layer']))
        self.assertEqual(sorted(os.listdir(blobs)), expected)
        self.assertEqual(containerstack.verify_oci_blobs(blobs, img['manifest']), manifest)

    def test_containerstack_fetch_reuses_cache(self):
        """A second fetch of a verified cache makes no registry requests at all."""
        reg, img, client = self._cs_registry()
        blobs = os.path.join(self.tmpdir, 'blobs')
        containerstack.fetch_oci_blobs(client, img['manifest'], blobs)
        before = len(reg.requests)
        containerstack.fetch_oci_blobs(client, img['manifest'], blobs)
        self.assertEqual(reg.requests[before:], [])

    def test_containerstack_corrupt_cache_refetched(self):
        """Corrupt, missing or half-downloaded cached blobs are detected and fetched again, never trusted."""
        reg, img, client = self._cs_registry()
        blobs = os.path.join(self.tmpdir, 'blobs')
        containerstack.fetch_oci_blobs(client, img['manifest'], blobs)
        layer = containerstack.blob_path(blobs, img['layer'])
        with open(layer, 'r+b') as fh:
            fh.seek(10)
            byte = fh.read(1)
            fh.seek(10)
            fh.write(bytes([byte[0] ^ 0xff]))
        self.assertRaisesRegex(EasyBuildError, "blob %s is corrupt" % img['layer'],
                               containerstack.verify_oci_blobs, blobs, img['manifest'])
        write_file(layer + '.part', 'junk from an interrupted download')
        self.mock_stdout(True)
        self.mock_stderr(True)
        containerstack.fetch_oci_blobs(client, img['manifest'], blobs)
        self.mock_stdout(False)
        self.mock_stderr(False)
        self.assertTrue(reg.blob_requests()[-1].endswith(img['layer']))
        containerstack.verify_oci_blobs(blobs, img['manifest'])
        self.assertFalse(os.path.exists(layer + '.part'))
        os.remove(layer)
        self.assertRaisesRegex(EasyBuildError, "blob %s is missing" % img['layer'],
                               containerstack.verify_oci_blobs, blobs, img['manifest'])

    def test_containerstack_tampered_download_refused(self):
        """A blob whose downloaded bytes do not match its digest is refused and leaves no file behind."""
        reg, img, client = self._cs_registry()
        blobs = os.path.join(self.tmpdir, 'blobs')
        reg.tamper_blobs = True
        self.assertRaisesRegex(EasyBuildError, "refusing it", containerstack.fetch_oci_blobs, client,
                               img['manifest'], blobs)
        self.assertEqual(os.listdir(blobs), [img['manifest'].split(':')[1]])

    def test_containerstack_bad_blob_digest(self):
        """Digests that are not sha256:<64 hex> are refused before being used as paths in the blob store."""
        for bad in ('sha256:/etc/x', 'sha256:../../x', 'nocolon', 'sha256:' + 'A' * 64):
            self.assertRaisesRegex(EasyBuildError, "invalid blob digest", containerstack.blob_path, self.tmpdir, bad)
        reg, img, client = self._cs_registry()
        manifest = {'schemaVersion': 2, 'mediaType': CS_OCI_MANIFEST,
                    'config': {'mediaType': 'application/vnd.oci.image.config.v1+json', 'digest': img['config'],
                               'size': len(reg.blobs[img['config']])},
                    'layers': [{'mediaType': 'application/vnd.oci.image.layer.v1.tar+gzip',
                                'digest': 'sha256:../../escape', 'size': 1}]}
        digest = reg.add_manifest(manifest, CS_OCI_MANIFEST)
        blobs = os.path.join(self.tmpdir, 'store', 'blobs')
        self.assertRaisesRegex(EasyBuildError, "invalid blob digest", containerstack.fetch_oci_blobs, client,
                               digest, blobs)
        escaped = [f for _, _, files in os.walk(self.tmpdir) for f in files if f in ('escape', 'escape.part')]
        self.assertEqual(escaped, [])

    def test_containerstack_oci_layout(self):
        """write_oci_layout() makes an OCI layout whose blobs are the shared store; rewriting is idempotent."""
        _, img, client = self._cs_registry()
        blobs = os.path.join(self.tmpdir, 'blobs')
        containerstack.fetch_oci_blobs(client, img['manifest'], blobs)
        layout = os.path.join(self.tmpdir, 'layout')
        for _ in range(2):
            containerstack.write_oci_layout(layout, blobs, img['manifest'], CS_OCI_MANIFEST)
        index = json.loads(read_file(os.path.join(layout, 'index.json')))
        self.assertEqual(index['manifests'][0]['digest'], img['manifest'])
        self.assertEqual(index['manifests'][0]['size'],
                         os.path.getsize(containerstack.blob_path(blobs, img['manifest'])))
        self.assertTrue(os.path.islink(os.path.join(layout, 'blobs', 'sha256')))
        self.assertTrue(os.path.isfile(os.path.join(layout, 'blobs', 'sha256', img['layer'].split(':')[1])))
        self.assertEqual(json.loads(read_file(os.path.join(layout, 'oci-layout'))), {'imageLayoutVersion': '1.0.0'})

    def test_containerstack_enroot_files(self):
        """enroot_files() reproduces enroot 4.2.0's /etc files exactly (fixtures captured from enroot import)."""
        topdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'containerstack_enroot')
        cases = sorted(d for d in os.listdir(topdir) if os.path.isdir(os.path.join(topdir, d)))
        self.assertEqual(cases, ['golang', 'nginx', 'postgres', 'python'])
        provenance = re.compile(r'^# enroot-provenance: .*\n')
        for case in cases:
            casedir = os.path.join(topdir, case)
            config = json.loads(read_file(os.path.join(casedir, 'config.json')))
            files = containerstack.enroot_files(config, 'containerstack://example.org/x@sha256:' + '0' * 64)
            for rel in ('etc/rc', 'etc/environment', 'etc/fstab', 'etc/rc.local'):
                expected = read_file(os.path.join(casedir, os.path.basename(rel)))
                got = files[rel]
                if rel == 'etc/rc':
                    self.assertTrue(provenance.match(got), got)
                    expected, got = provenance.sub('', expected), provenance.sub('', got)
                self.assertEqual(got, expected, "%s: %s differs from enroot's" % (case, rel))
        # no entrypoint and no cmd -> /bin/sh; arguments quoted like bash ${var@Q}
        files = containerstack.enroot_files({'config': {}}, 'p')
        self.assertIn("    exec  '/bin/sh'\n", files['etc/rc'])
        files = containerstack.enroot_files({'config': {'Entrypoint': ['/x y', "it's"]}}, 'p')
        self.assertIn("    exec '/x y' 'it'\\''s' \"$@\"\n", files['etc/rc'])
        self.assertIn("    exec '/x y' 'it'\\''s' \n", files['etc/rc'])

    def test_containerstack_squash_options(self):
        """mksquashfs options: easyconfig value, then $ENROOT_SQUASH_OPTIONS, enroot.conf, enroot's default."""
        self.assertEqual(containerstack.det_squash_options('-comp zstd'), '-comp zstd')
        os.environ['ENROOT_SYSCONF_PATH'] = os.path.join(self.tmpdir, 'enroot')
        self.assertEqual(containerstack.det_squash_options(), '-comp lzo -noD -exit-on-error')
        write_file(os.path.join(self.tmpdir, 'enroot', 'enroot.conf'),
                   "#ENROOT_SQUASH_OPTIONS -comp lzo\nENROOT_SQUASH_OPTIONS     -noI -noD -noF -no-duplicates  \n")
        self.assertEqual(containerstack.det_squash_options(), '-noI -noD -noF -no-duplicates')
        os.environ['ENROOT_SQUASH_OPTIONS'] = '-comp gzip'
        self.assertEqual(containerstack.det_squash_options(), '-comp gzip')

    def test_containerstack_wrapper(self):
        """Wrappers choose the runtime from context and pass arguments, mounts and site env through unchanged."""
        root = os.path.join(self.tmpdir, 'install')
        fakebin = os.path.join(self.tmpdir, 'fakebin')
        for tool in ('srun', 'apptainer', 'enroot'):
            path = os.path.join(fakebin, tool)
            # like Apptainer, the fake image sets OMP_NUM_THREADS itself: only APPTAINERENV_<NAME> overrides it
            env = '${APPTAINERENV_OMP_NUM_THREADS:-image}' if tool == 'apptainer' else '$OMP_NUM_THREADS'
            write_file(path, '#!/bin/bash\necho "TOOL $(basename "$0")"\nfor a in "$@"; do echo "ARG $a"; done\n'
                             'echo "ENV OMP_NUM_THREADS=%s"\n' % env)
            adjust_permissions(path, stat.S_IXUSR, add=True)
        for fmt in ('sqsh', 'sif'):
            write_file(os.path.join(root, 'image.' + fmt), '')
        write_file(os.path.join(root, 'etc', 'site.env'), '# site policy\n\nOMP_NUM_THREADS=1\n')
        wrapper = os.path.join(root, 'bin', 'greet')
        write_file(wrapper, containerstack.render_wrapper('tiny', '1.0', 'sha256:' + 'a' * 64,
                                                          "cat '/etc/my greeting'", 'auto', ['/data', '/x:/y']))
        adjust_permissions(wrapper, stat.S_IXUSR, add=True)
        real = os.path.realpath(root)          # the wrapper resolves symlinks (macOS /var -> /private/var)
        sif, sqsh = os.path.join(real, 'image.sif'), os.path.join(real, 'image.sqsh')

        def run(env=None, args=()):
            full = {'PATH': fakebin + ':/usr/bin:/bin', 'CS_GPU_DEVICE': os.path.join(self.tmpdir, 'no-gpu')}
            full.update(env or {})
            res = subprocess.run([wrapper] + list(args), env=full, capture_output=True, text=True)
            return res.returncode, res.stdout.splitlines(), res.stderr

        # outside Slurm: Apptainer, binds normalised to src:dst, command words and user args intact, site env set
        # (beating the image's own ENV); comment and blank lines in site.env are skipped silently
        code, out, err = run(args=['a b', "it's"])
        self.assertEqual(code, 0)
        self.assertEqual(err, '')
        self.assertEqual(out, ['TOOL apptainer', 'ARG exec', 'ARG --bind', 'ARG /data:/data,/x:/y', 'ARG ' + sif,
                               'ARG cat', 'ARG /etc/my greeting', 'ARG a b', "ARG it's", 'ENV OMP_NUM_THREADS=1'])
        # in an allocation but not in a step (batch script): pyxis via srun, one task like the plain command
        code, out, err = run({'SLURM_JOB_ID': '42'})
        self.assertEqual(err, '')
        self.assertEqual(out, ['TOOL srun', 'ARG --nodes=1', 'ARG --ntasks=1', 'ARG --container-image=' + sqsh,
                               'ARG --container-mounts=/data:/data,/x:/y', 'ARG --container-env=OMP_NUM_THREADS',
                               'ARG cat', 'ARG /etc/my greeting', 'ENV OMP_NUM_THREADS=1'])
        # CS_SRUN_ARGS comes after the defaults, so it wins (srun: the later option counts)
        code, out, _ = run({'SLURM_JOB_ID': '42', 'CS_SRUN_ARGS': '--ntasks-per-node=1 -N 2'})
        self.assertEqual(out[1:6], ['ARG --nodes=1', 'ARG --ntasks=1', 'ARG --ntasks-per-node=1', 'ARG -N', 'ARG 2'])
        # already inside a step: pyxis can no longer attach a container -> Apptainer, never a nested srun
        code, out, _ = run({'SLURM_JOB_ID': '42', 'SLURM_STEP_ID': '0'})
        self.assertEqual(out[0], 'TOOL apptainer')
        # a GPU device present -> --nv
        gpu = os.path.join(self.tmpdir, 'nvidia0')
        write_file(gpu, '')
        code, out, _ = run({'CS_GPU_DEVICE': gpu})
        self.assertEqual(out[:3], ['TOOL apptainer', 'ARG exec', 'ARG --nv'])
        # forced enroot (sqsh): create once, start with mounts and env names
        code, out, _ = run({'CS_RUNTIME': 'sqsh'})
        self.assertEqual(code, 0)
        self.assertEqual(out[:2], ['TOOL enroot', 'ARG start'])
        self.assertEqual(out[out.index('ARG --env') + 1], 'ARG OMP_NUM_THREADS')
        self.assertFalse([a for a in out if '#' in a])
        self.assertIn('ARG /x:/y', out)
        self.assertIn('ARG cs-tiny-1.0-aaaaaaaaaaaa', out)
        # debug output, and an unknown runtime
        _, _, err = run({'CS_DEBUG': '1'})
        self.assertIn('ContainerStack: runtime=apptainer', err)
        code, _, err = run({'CS_RUNTIME': 'bogus'})
        self.assertEqual(code, 2)
        self.assertIn("unknown CS_RUNTIME 'bogus'", err)
        # the generic .cs-run wrapper runs its arguments
        cs_run = os.path.join(root, 'bin', '.cs-run')
        write_file(cs_run, containerstack.render_wrapper('tiny', '1.0', 'sha256:' + 'a' * 64, '', 'auto', []))
        adjust_permissions(cs_run, stat.S_IXUSR, add=True)
        res = subprocess.run([cs_run, 'sh', '-c', 'true'], capture_output=True, text=True,
                             env={'PATH': fakebin + ':/usr/bin:/bin', 'CS_GPU_DEVICE': '/nonexistent'})
        self.assertEqual(res.stdout.splitlines()[:5], ['TOOL apptainer', 'ARG exec', 'ARG ' + sif, 'ARG sh', 'ARG -c'])

    def test_containerstack_apptainer_triton_libcuda(self):
        """With --nv, Apptainer gets TRITON_LIBCUDA_PATH=/.singularity.d/libs, unless the caller set it already."""
        root = os.path.join(self.tmpdir, 'install')
        fakebin = os.path.join(self.tmpdir, 'fakebin')
        for tool in ('srun', 'apptainer', 'enroot'):
            path = os.path.join(fakebin, tool)
            write_file(path, '#!/bin/bash\necho "TOOL $(basename "$0")"\nfor a in "$@"; do echo "ARG $a"; done\n'
                             'echo "ENV TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH-unset}"\n'
                             'echo "ENV APPTAINERENV_TRITON_LIBCUDA_PATH=${APPTAINERENV_TRITON_LIBCUDA_PATH-unset}"\n')
            adjust_permissions(path, stat.S_IXUSR, add=True)
        for fmt in ('sqsh', 'sif'):
            write_file(os.path.join(root, 'image.' + fmt), '')
        wrapper = os.path.join(root, 'bin', 'python')
        write_file(wrapper, containerstack.render_wrapper('tiny', '1.0', 'sha256:' + 'a' * 64, 'python', 'auto', []))
        adjust_permissions(wrapper, stat.S_IXUSR, add=True)
        gpu = os.path.join(self.tmpdir, 'nvidia0')
        write_file(gpu, '')

        def run(env):
            full = {'PATH': fakebin + ':/usr/bin:/bin', 'CS_GPU_DEVICE': gpu}
            full.update(env)
            res = subprocess.run([wrapper], env=full, capture_output=True, text=True)
            self.assertEqual((res.returncode, res.stderr), (0, ''))
            out = res.stdout.splitlines()
            return out[0], 'ARG --nv' in out, [line for line in out if line.startswith('ENV ')]

        # Apptainer with --nv: Triton looks for libcuda where --nv put the driver libraries
        self.assertEqual(run({}), ('TOOL apptainer', True,
                                   ['ENV TRITON_LIBCUDA_PATH=unset',
                                    'ENV APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs']))
        # the caller's own setting (either name, even empty) is left alone
        self.assertEqual(run({'TRITON_LIBCUDA_PATH': '/opt/drv'})[2], ['ENV TRITON_LIBCUDA_PATH=/opt/drv',
                                                                       'ENV APPTAINERENV_TRITON_LIBCUDA_PATH=unset'])
        self.assertEqual(run({'TRITON_LIBCUDA_PATH': ''})[2], ['ENV TRITON_LIBCUDA_PATH=',
                                                               'ENV APPTAINERENV_TRITON_LIBCUDA_PATH=unset'])
        self.assertEqual(run({'APPTAINERENV_TRITON_LIBCUDA_PATH': '/opt/drv'})[2],
                         ['ENV TRITON_LIBCUDA_PATH=unset', 'ENV APPTAINERENV_TRITON_LIBCUDA_PATH=/opt/drv'])
        # no GPU (no --nv), pyxis and enroot (the NVIDIA hook puts libcuda in the image's library path): not set
        unset = ['ENV TRITON_LIBCUDA_PATH=unset', 'ENV APPTAINERENV_TRITON_LIBCUDA_PATH=unset']
        self.assertEqual(run({'CS_GPU_DEVICE': '/nonexistent'}), ('TOOL apptainer', False, unset))
        self.assertEqual(run({'SLURM_JOB_ID': '42'}), ('TOOL srun', False, unset))
        self.assertEqual(run({'CS_RUNTIME': 'sqsh'})[::2], ('TOOL enroot', unset))

    def test_containerstack_enroot_skips_entrypoint(self):
        """Under enroot the wrappers run their command, not the image's entrypoint (as Apptainer and pyxis' default)."""
        eb, _ = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        conf = os.path.join(eb.installdir, 'etc', 'enroot-no-entrypoint.conf')
        self.assertTrue(os.path.isfile(conf))
        # the wrapper passes the hook to 'enroot start'
        fakebin = os.path.join(self.tmpdir, 'fakebin')
        write_file(os.path.join(fakebin, 'enroot'), '#!/bin/bash\nfor a in "$@"; do echo "ARG $a"; done\n')
        adjust_permissions(os.path.join(fakebin, 'enroot'), stat.S_IXUSR, add=True)
        res = subprocess.run([os.path.join(eb.installdir, 'bin', 'greet')], capture_output=True, text=True,
                             env={'PATH': fakebin + ':/usr/bin:/bin', 'CS_RUNTIME': 'enroot',
                                  'CS_GPU_DEVICE': '/nonexistent'})
        out = res.stdout.splitlines()
        self.assertEqual(out[:3], ['ARG start', 'ARG --conf', 'ARG ' + os.path.realpath(conf)])
        # what the hook does to enroot's /etc/rc: the command runs instead of the entrypoint; no command -> as before
        rootfs, workdir = os.path.join(self.tmpdir, 'rootfs'), os.path.join(self.tmpdir, 'workdir')
        files = containerstack.enroot_files({'config': {'Entrypoint': ['echo', 'ENTRY'], 'Cmd': ['default'],
                                                        'WorkingDir': workdir}}, 'p')
        write_file(os.path.join(rootfs, 'etc', 'rc.local'), files['etc/rc.local'])
        rc = os.path.join(rootfs, 'etc', 'rc')
        write_file(rc, files['etc/rc'].replace('/etc/rc.local', os.path.join(rootfs, 'etc', 'rc.local')))

        def start(*args):
            return subprocess.run(['bash', rc] + list(args), capture_output=True, text=True).stdout

        self.assertEqual(start('echo', 'hi'), 'ENTRY echo hi\n')    # plain enroot: entrypoint + command
        subprocess.run(['bash', '-c', 'source "$1" && hooks', '-', conf], check=True,
                       env={'PATH': '/usr/bin:/bin', 'ENROOT_ROOTFS': rootfs})
        self.assertEqual(start('echo', 'hi'), 'hi\n')
        self.assertEqual(start('pwd', '-P'), os.path.realpath(workdir) + '\n')    # still in the image's WorkingDir
        self.assertEqual(start(), 'ENTRY default\n')

    def test_containerstack_enroot_caller_env(self):
        """Under enroot the caller's environment reaches the container, as under pyxis and Apptainer."""
        root, fake = os.path.join(self.tmpdir, 'install'), os.path.join(self.tmpdir, 'fake-enroot')
        fakebin = os.path.join(self.tmpdir, 'fakebin')
        write_file(os.path.join(fakebin, 'enroot'), CS_FAKE_ENROOT.replace('@DIR@', fake))
        adjust_permissions(os.path.join(fakebin, 'enroot'), stat.S_IXUSR, add=True)
        write_file(os.path.join(fake, 'image-environment'),
                   'PATH=/image/bin\nSHARED=image\nIMAGE_ONLY=image\nOMP_NUM_THREADS=image\n')
        write_file(os.path.join(root, 'image.sqsh'), '')
        write_file(os.path.join(root, 'etc', 'site.env'), 'OMP_NUM_THREADS=1\n')
        write_file(os.path.join(root, containerstack.ENROOT_NO_ENTRYPOINT), containerstack.ENROOT_NO_ENTRYPOINT_CONF)
        wrapper = os.path.join(root, 'bin', 'python')
        write_file(wrapper, containerstack.render_wrapper('tiny', '1.0', 'sha256:' + 'a' * 64, 'python', 'auto', []))
        adjust_permissions(wrapper, stat.S_IXUSR, add=True)
        caller = {'PATH': fakebin + ':/usr/bin:/bin', 'CS_RUNTIME': 'sqsh', 'CS_GPU_DEVICE': '/nonexistent',
                  'HF_HUB_OFFLINE': '1', 'NCCL_DEBUG': 'INFO', 'SPACED': ' a  b ', 'EMPTY': '',
                  'SHARED': 'caller', 'OMP_NUM_THREADS': '8', 'TWO_LINES': 'a\nINJECTED=1',
                  'LD_LIBRARY_PATH': '/host/lib', 'LD_PRELOAD': '/host/lib/x.so', 'LANG': 'C.UTF-8', 'LC_ALL': 'C',
                  'HOME': '/home/u', 'USER': 'u', 'TERM': 'xterm'}
        res = subprocess.run([wrapper, '-c', 'pass'], env=caller, capture_output=True, text=True)
        self.assertEqual((res.returncode, res.stderr), (0, ''))
        env = dict(line[4:].split('=', 1) for line in res.stdout.splitlines())
        for name in ('PWD', 'SHLVL'):    # bash's own, passed on like any other
            env.pop(name, None)
        # the caller's variables go in as they are; the image's ENV wins over them (as under pyxis), site.env over
        # both; PATH, LD_*, locale and the user's identity are the image's or enroot's, never the caller's; a value
        # with a newline cannot be one line of enroot's environment file and stays out; enroot's own XDG_* too
        self.assertEqual(env, {'PATH': '/image/bin', 'IMAGE_ONLY': 'image', 'SHARED': 'image',
                               'OMP_NUM_THREADS': '1', 'APPTAINERENV_OMP_NUM_THREADS': '1',
                               'CS_RUNTIME': 'sqsh', 'CS_GPU_DEVICE': '/nonexistent',
                               'HF_HUB_OFFLINE': '1', 'NCCL_DEBUG': 'INFO', 'SPACED': ' a  b ', 'EMPTY': ''})

    def _cs_fake_run(self, cmd, **kwargs):
        """Stand-in for run_shell_cmd: records commands and fakes the files Apptainer/mksquashfs would make."""
        self.cs_calls.append(cmd)
        sandbox = re.search(r'apptainer build --force --sandbox (\S+) oci:', cmd)
        image = re.search(r'apptainer build --force (\S+\.sif) oci:', cmd)
        squash = re.search(r'^mksquashfs (\S+) (\S+) ', cmd)
        if sandbox:
            rootfs = sandbox.group(1)
            for sub in ('.singularity.d/env', 'etc', 'usr/bin'):
                mkdir(os.path.join(rootfs, sub), parents=True)
            for name in ('.exec', '.run', '.shell', '.test'):
                write_file(os.path.join(rootfs, name), '')
            os.symlink('.singularity.d/env', os.path.join(rootfs, 'environment'))
            os.symlink('.singularity.d', os.path.join(rootfs, 'singularity'))
            for name in ('etc/hosts', 'etc/resolv.conf'):    # Apptainer's empty bind targets
                write_file(os.path.join(rootfs, name), '')
            mkdir(os.path.join(rootfs, 'var', 'empty'), parents=True)
            os.chmod(os.path.join(rootfs, 'var', 'empty'), 0o555)
            os.chmod(rootfs, 0o2750)
        elif image:
            write_file(image.group(1), 'sif image')
        elif squash:
            rootfs = squash.group(1)
            self.cs_squashed = {'top': sorted(os.listdir(rootfs)),
                                'modes': dict((rel, stat.S_IMODE(os.lstat(os.path.join(rootfs, rel)).st_mode))
                                              for rel in ('', 'var/empty', 'etc/hosts')),
                                'hosts': read_file(os.path.join(rootfs, 'etc', 'hosts')),
                                'resolv': os.path.lexists(os.path.join(rootfs, 'etc', 'resolv.conf')),
                                'rc': read_file(os.path.join(rootfs, 'etc', 'rc')),
                                'environment': read_file(os.path.join(rootfs, 'etc', 'environment'))}
            write_file(squash.group(2), 'sqsh image')
        code, out = 0, 'fake 1.0\n'
        for pattern, exit_code in self.cs_fail.items():
            if pattern in cmd:
                code, out = exit_code, 'boom\n'
        return RunShellCmdResult(cmd=cmd, exit_code=code, output=out, stderr=None, work_dir=None, out_file=None,
                                 err_file=None, cmd_sh=None, thread_id=None, task_id=None)

    def _cs_easyblock(self, extra='', workdir='/work'):
        """ContainerStack instance for the tiny image in a fake registry, with mocked commands and paths."""
        reg, img, _ = self._cs_registry(workdir=workdir)
        ec_file = os.path.join(self.tmpdir, 'tiny-container-1.0.eb')
        write_file(ec_file, textwrap.dedent("""
            easyblock = 'ContainerStack'
            name = 'tiny-container'
            version = '1.0'
            homepage = 'https://example.org'
            description = 'ContainerStack test stack'
            toolchain = SYSTEM
            image = '%s/%s'
            registry_scheme = 'http'
            digests = {'x86_64': '%s'}
            launchers = {'greet': 'cat /etc/greeting'}
            site_env = {'OMP_NUM_THREADS': '1'}
            smoke_tests = [('cpu', 'test -f /etc/greeting'), ('gpu', 'nvidia-smi -L')]
            squash_options = '-noI -noD'
        """) % (reg.address, reg.repo, img['manifest']) + extra)
        easyconfig._easyconfigs_cache.clear()    # same path, new content: parse it again
        eb = get_easyblock_instance(process_easyconfig(ec_file)[0])
        eb.builddir = os.path.join(self.tmpdir, 'build')
        mkdir(eb.builddir, parents=True)
        eb.installdir = os.path.join(self.tmpdir, 'install')
        containerstack.host_arch_key = lambda: 'x86_64'
        containerstack.source_paths = lambda: [os.path.join(self.tmpdir, 'sources')]
        containerstack.run_shell_cmd = self._cs_fake_run
        return eb, img

    def _cs_gpu(self, present):
        """Point the GPU check (Python and wrappers) at an existing or a missing device file."""
        path = os.path.join(self.tmpdir, 'nvidia0' if present else 'no-gpu')
        if present:
            write_file(path, '')
        os.environ['CS_GPU_DEVICE'] = path

    def test_containerstack_validation(self):
        """Bad easyconfig values fail early with a message naming the parameter."""
        cases = [("formats = ['tar']\n", "unknown format"), ("formats = []\n", "unknown format"),
                 ("launch_policy = 'docker'\n", "launch_policy must be one of"),
                 ("launchers = {'../x': 'ls'}\n", "invalid launcher name"),
                 ("site_env = {'A': 1}\n", "site_env"), ("site_env = {'1A': 'x'}\n", "site_env"),
                 ("smoke_tests = [('tpu', 'true')]\n", "smoke test"),
                 ("image = 'nvcr.io/nvidia/pytorch:26.10'\n", "must not contain a tag"),
                 ("image = None\n", "'image' is required")]
        for extra, msg in cases:
            eb, _ = self._cs_easyblock(extra)
            self.assertRaisesRegex(EasyBuildError, msg, eb.fetch_step)

    def test_containerstack_build_step(self):
        """build_step makes the .sif from the verified layout and a clean enroot-style .sqsh, never appending."""
        eb, img = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        layout = os.path.join(eb.builddir, 'oci-layout')
        rootfs = os.path.join(eb.builddir, 'rootfs')
        sif = [c for c in self.cs_calls if 'apptainer build --force %s/image.sif oci:%s' % (eb.builddir, layout) in c]
        self.assertEqual(len(sif), 1)
        self.assertIn('APPTAINER_TMPDIR=%s/tmp' % eb.builddir, sif[0])
        squash = [c for c in self.cs_calls if c.startswith('mksquashfs ')]
        self.assertEqual(squash, ['mksquashfs %s %s/image.sqsh -noappend -all-root -no-progress -noI -noD'
                                  % (rootfs, eb.builddir)])
        self.assertEqual(self.cs_squashed['top'], ['etc', 'usr', 'var', 'work'])
        # as 'enroot import' makes it: the image's /etc/hosts (not Apptainer's empty one), no /etc/resolv.conf
        # the image lacks, user-writable directories and a 0755 root
        self.assertEqual(self.cs_squashed['hosts'], '127.0.0.1 localhost\n')
        self.assertFalse(self.cs_squashed['resolv'])
        self.assertEqual(self.cs_squashed['modes'], {'': 0o755, 'var/empty': 0o755, 'etc/hosts': 0o640})
        self.assertTrue(self.cs_squashed['rc'].startswith('# enroot-provenance: containerstack://%s@%s\n'
                                                          % (eb.cfg['image'], img['manifest'])))
        self.assertEqual(self.cs_squashed['environment'], 'PATH=/usr/bin:/bin\nGREETING=hello world\n')
        self.assertFalse(os.path.exists(rootfs))
        # only the requested formats are built
        self.cs_calls = []
        eb, _ = self._cs_easyblock("formats = ['sif']\n")
        eb.fetch_step()
        eb.build_step()
        self.assertEqual([c for c in self.cs_calls if 'mksquashfs' in c or '--sandbox' in c], [])

    def test_containerstack_image_entries(self):
        """image_entries applies layers in order, with OCI whiteouts and opaque directories."""
        blobs = os.path.join(self.tmpdir, 'blobs')
        layers = []
        for members in ([('etc/hosts', b'one\n'), ('etc/resolv.conf', b'ns\n'), ('etc/a/x', b'x\n')],
                        [('etc/.wh.resolv.conf', b''), ('etc/a/.wh..wh..opq', b''), ('./etc/hosts', b'two\n')]):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode='w:gz') as tar:
                for name, data in members:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            digest = containerstack.sha256_digest(buf.getvalue())
            path = containerstack.blob_path(blobs, digest)
            mkdir(os.path.dirname(path), parents=True)
            with open(path, 'wb') as fh:
                fh.write(buf.getvalue())
            layers.append({'digest': digest})
        found = containerstack.image_entries(blobs, layers, ('etc/hosts', 'etc/resolv.conf', 'etc/a/x', 'etc/no'))
        self.assertEqual(found['etc/hosts'][1], b'two\n')
        self.assertEqual([found[p] for p in ('etc/resolv.conf', 'etc/a/x', 'etc/no')], [None, None, None])
        self.assertEqual(containerstack.image_entries(blobs, layers[:1], ('etc/a/x',))['etc/a/x'][1], b'x\n')

    def test_containerstack_image_entries_root_opaque_and_replaced_dirs(self):
        """An opaque whiteout at the image root clears all lower layers; a non-directory hides what was beneath."""
        blobs = os.path.join(self.tmpdir, 'blobs')
        lower = _cs_layer(blobs, [_cs_member('etc/hosts', b'one\n'), _cs_member('usr/lib/x', b'x\n'),
                                  _cs_member('opt/y', b'y\n')])
        root_opq = _cs_layer(blobs, [_cs_member('./.wh..wh..opq'), _cs_member('opt/y', b'new y\n')])
        found = containerstack.image_entries(blobs, [lower, root_opq], ('etc/hosts', 'usr/lib/x', 'opt/y'))
        self.assertEqual(found['etc/hosts'], None)
        self.assertEqual(found['usr/lib/x'], None)
        self.assertEqual(found['opt/y'][1], b'new y\n')    # same-layer entries survive the opaque marker
        replaced = _cs_layer(blobs, [_cs_member('etc', kind=tarfile.SYMTYPE, linkname='private/etc'),
                                     _cs_member('usr/lib', b'now a file\n')])
        found = containerstack.image_entries(blobs, [lower, replaced], ('etc/hosts', 'usr/lib/x', 'opt/y', 'etc'))
        self.assertEqual([found[p] for p in ('etc/hosts', 'usr/lib/x')], [None, None])
        self.assertEqual(found['opt/y'][1], b'y\n')
        self.assertTrue(found['etc'][0].issym())
        # a later layer that has 'etc' as a directory again keeps the lower layer's files
        etc_dir = _cs_layer(blobs, [_cs_member('etc', kind=tarfile.DIRTYPE, mode=0o755)])
        self.assertEqual(containerstack.image_entries(blobs, [lower, etc_dir], ('etc/hosts',))['etc/hosts'][1],
                         b'one\n')

    def _cs_without_zstd_tarfile(self):
        """Hide this Python's tarfile zstd support (as before Python 3.14) until the test ends."""
        orig = tarfile.TarFile.OPEN_METH
        tarfile.TarFile.OPEN_METH = dict((k, v) for k, v in orig.items() if k != 'zst')
        self.addCleanup(setattr, tarfile.TarFile, 'OPEN_METH', orig)

    def test_containerstack_zstd_layer_no_reader(self):
        """A zstd layer with neither tarfile zstd support nor a zstd command: an error naming layer and tool."""
        blobs = os.path.join(self.tmpdir, 'blobs')
        members = [_cs_member('etc/hosts', b'zstd hosts\n')]
        if shutil.which('zstd'):
            layer = _cs_layer(blobs, members, zstd=True)
        else:    # the content is never read: the missing reader is reported first
            layer = dict(_cs_layer(blobs, members), mediaType='application/vnd.oci.image.layer.v1.tar+zstd')
        self._cs_without_zstd_tarfile()
        os.environ['PATH'] = os.path.join(self.tmpdir, 'empty-path')
        self.assertRaisesRegex(EasyBuildError, "layer %s is zstd-compressed.*no 'zstd' command" % layer['digest'],
                               containerstack.image_entries, blobs, [layer], ('etc/hosts',))

    def test_containerstack_zstd_layer_via_command(self):
        """Without tarfile zstd support, a zstd layer is read through the zstd command."""
        if not shutil.which('zstd'):
            self.skipTest("no 'zstd' command in $PATH")
        blobs = os.path.join(self.tmpdir, 'blobs')
        layers = [_cs_layer(blobs, [_cs_member('etc/hosts', b'one\n'), _cs_member('etc/resolv.conf', b'ns\n')]),
                  _cs_layer(blobs, [_cs_member('etc/hosts', b'zstd hosts\n', mode=0o600),
                                    _cs_member('etc/.wh.resolv.conf')], zstd=True)]
        if containerstack.tarfile_reads_zstd():    # Python >= 3.14: tarfile itself
            self.assertEqual(containerstack.image_entries(blobs, layers, ('etc/hosts',))['etc/hosts'][1],
                             b'zstd hosts\n')
        self._cs_without_zstd_tarfile()
        self.assertFalse(containerstack.tarfile_reads_zstd())
        found = containerstack.image_entries(blobs, layers, ('etc/hosts', 'etc/resolv.conf'))
        self.assertEqual((found['etc/hosts'][1], found['etc/hosts'][0].mode), (b'zstd hosts\n', 0o600))
        self.assertEqual(found['etc/resolv.conf'], None)

    def test_containerstack_fixperms(self):
        """enroot's fixperms: chmod -R u+rX, then u+w on directories as deep as enroot's find reaches them."""
        root = os.path.join(self.tmpdir, 'rootfs')
        outside = os.path.join(self.tmpdir, 'outside')
        deep = [os.path.join(root, *['n%d' % i for i in range(1, depth + 1)]) for depth in range(1, 17)]
        mkdir(deep[-1], parents=True)
        mkdir(os.path.join(root, 'locked'))
        mkdir(os.path.join(outside, 'dir'), parents=True)
        files = {'f0000': 0o000, 'f0644': 0o644, 'f0700': 0o700, 'f0055': 0o055, 'locked/inner': 0o000,
                 os.path.join(outside, 'target'): 0o000, os.path.join(outside, 'dir', 'g'): 0o000}
        for rel, mode in files.items():
            write_file(os.path.join(root, rel), 'x')
        os.symlink(os.path.join(outside, 'target'), os.path.join(root, 'link'))
        os.symlink(os.path.join(outside, 'dir'), os.path.join(root, 'dirlink'))
        if hasattr(os, 'lchmod'):    # macOS: a symlink has a mode of its own; a chmod of the link must not happen
            os.lchmod(os.path.join(root, 'link'), 0o000)
        dirs = {'locked': 0o000, 'n1/n2': 0o555, deep[13]: 0o555, deep[14]: 0o555, deep[15]: 0o555,
                os.path.join(outside, 'dir'): 0o555}
        for rel, mode in list(files.items()) + sorted(dirs.items(), key=lambda kv: -len(kv[0])):
            os.chmod(os.path.join(root, rel), mode)
        try:
            containerstack.enroot_fixperms(root)
            modes = dict((rel, stat.S_IMODE(os.lstat(os.path.join(root, rel)).st_mode))
                         for rel in list(files) + list(dirs))
        finally:    # tearDown only adds u+w
            os.chmod(os.path.join(root, 'locked'), 0o700)

        def mode(rel):
            return modes[rel]
        # u+r everywhere; u+x on directories and on files that anyone may execute (X); never through a symlink
        self.assertEqual(dict((rel, mode(rel)) for rel in files),
                         {'f0000': 0o400, 'f0644': 0o644, 'f0700': 0o700, 'f0055': 0o555, 'locked/inner': 0o400,
                          os.path.join(outside, 'target'): 0o000, os.path.join(outside, 'dir', 'g'): 0o000})
        self.assertEqual(mode(os.path.join(outside, 'dir')), 0o555)
        self.assertTrue(os.path.islink(os.path.join(root, 'link')) and os.path.islink(os.path.join(root, 'dirlink')))
        # u+w on directories: enroot runs 'find . -maxdepth 15' in the directory holding one directory per layer,
        # so an image directory n levels below the rootfs is at find depth n + 1
        self.assertEqual((mode('locked'), mode('n1/n2')), (0o700, 0o755))
        self.assertEqual([mode(d) for d in deep[13:]], [0o755, 0o555, 0o555])

    def test_containerstack_sqsh_symlink_bind_target(self):
        """An image whose /etc/hosts is a symlink gets that symlink back in the .sqsh, not Apptainer's file."""
        seen = {}

        def run(cmd, **kwargs):
            sandbox = re.search(r'--sandbox (\S+) ', cmd)
            if cmd.startswith('mksquashfs '):
                hosts = os.path.join(cmd.split()[1], 'etc', 'hosts')
                seen['hosts'] = os.readlink(hosts) if os.path.islink(hosts) else None
            res = self._cs_fake_run(cmd, **kwargs)
            if sandbox:
                write_file(os.path.join(sandbox.group(1), 'etc', 'hosts.image'), 'via the image link\n')
            return res

        info = _cs_member('etc/hosts', kind=tarfile.SYMTYPE, linkname='hosts.image')[0]
        orig = containerstack.image_entries
        containerstack.image_entries = lambda *args: {'etc/hosts': (info, None), 'etc/resolv.conf': None}
        self.addCleanup(setattr, containerstack, 'image_entries', orig)
        eb, _ = self._cs_easyblock("formats = ['sqsh']\n")
        eb.fetch_step()
        containerstack.run_shell_cmd = run
        eb.build_step()
        self.assertEqual(seen['hosts'], 'hosts.image')
        self.assertEqual(self.cs_squashed['hosts'], 'via the image link\n')

    def test_containerstack_sqsh_root_mode_from_umask(self):
        """The .sqsh root directory has the mode a plain mkdir gives it, as in 'enroot import': 0777 & ~umask."""
        for umask, expected in ((0o022, 0o755), (0o002, 0o775)):
            eb, _ = self._cs_easyblock("formats = ['sqsh']\n")
            eb.fetch_step()
            old = os.umask(umask)
            try:
                eb.build_step()
                self.assertEqual(os.umask(umask), umask)    # the umask is left as it was
            finally:
                os.umask(old)
            self.assertEqual(self.cs_squashed['modes'][''], expected)

    def test_containerstack_rootfs_writes_stay_inside(self):
        """enroot's /etc files replace image symlinks instead of writing through them; no write escapes the rootfs."""
        outside = os.path.join(self.tmpdir, 'host')
        write_file(os.path.join(outside, 'rc'), 'host file\n')
        plant, seen = {}, {}

        def run(cmd, **kwargs):
            if cmd.startswith('mksquashfs '):
                rc = os.path.join(cmd.split()[1], 'etc', 'rc')
                seen['rc'] = (os.path.islink(rc), os.path.isfile(rc))
            res = self._cs_fake_run(cmd, **kwargs)
            sandbox = re.search(r'--sandbox (\S+) ', cmd)
            if sandbox:
                for rel, dest in plant.items():
                    path = os.path.join(sandbox.group(1), rel)
                    if os.path.isdir(path):
                        remove_dir(path)
                    os.symlink(dest, path)
            return res

        eb, _ = self._cs_easyblock("formats = ['sqsh']\n")
        eb.fetch_step()
        containerstack.run_shell_cmd = run
        plant = {'etc/rc': os.path.join(outside, 'rc')}
        eb.build_step()
        self.assertEqual(read_file(os.path.join(outside, 'rc')), 'host file\n')
        self.assertEqual(seen['rc'], (False, True))
        self.assertTrue(self.cs_squashed['rc'].startswith('# enroot-provenance: '))
        # an image whose /etc is an absolute symlink would resolve on the host: refused
        plant = {'etc': outside}
        self.assertRaisesRegex(EasyBuildError, "outside the image rootfs", eb.build_step)
        self.assertEqual(sorted(os.listdir(outside)), ['rc'])
        self.assertEqual(read_file(os.path.join(outside, 'rc')), 'host file\n')
        remove_dir(os.path.join(eb.builddir, 'rootfs'))    # apptainer build --force would replace it
        # a WorkingDir with '..' (rootfs is <tmpdir>/build/rootfs) would make the mkdir escape: refused
        eb, _ = self._cs_easyblock("formats = ['sqsh']\n", workdir='/../../x')
        eb.fetch_step()
        self.assertRaisesRegex(EasyBuildError, "outside the image rootfs", eb.build_step)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, 'x')))

    def test_containerstack_free_space(self):
        """build_step refuses to start without min_free_space_factor x the image size free."""
        eb, _ = self._cs_easyblock()
        eb.fetch_step()
        containerstack.disk_free = lambda path: 10
        self.cs_calls = []
        self.assertRaisesRegex(EasyBuildError, r"GiB free in .*, need .* GiB \(3x the image size\)", eb.build_step)
        self.assertEqual([c for c in self.cs_calls if 'apptainer' in c or 'mksquashfs' in c], [])

    def test_containerstack_install_step(self):
        """install_step copies images and writes wrappers and site.env, but no manifest yet."""
        eb, _ = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        for rel in ('image.sif', 'image.sqsh', 'bin/greet', 'bin/.cs-run', 'etc/site.env'):
            self.assertTrue(os.path.isfile(os.path.join(eb.installdir, rel)), rel)
        self.assertTrue(os.access(os.path.join(eb.installdir, 'bin', 'greet'), os.X_OK))
        self.assertEqual(read_file(os.path.join(eb.installdir, 'etc', 'site.env')), 'OMP_NUM_THREADS=1\n')
        self.assertIn('cmd=(cat /etc/greeting)', read_file(os.path.join(eb.installdir, 'bin', 'greet')))
        self.assertIn('cmd=()', read_file(os.path.join(eb.installdir, 'bin', '.cs-run')))
        self.assertFalse(os.path.exists(os.path.join(eb.installdir, 'manifest.json')))

    def test_containerstack_sanity_check_paths(self):
        """The sanity check requires every installed file, including the enroot hook the wrappers use with sqsh."""
        paths = []
        EasyBlock.sanity_check_step = lambda self, *args, **kwargs: paths.append(kwargs['custom_paths'])
        eb, _ = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        self._cs_gpu(True)
        eb.sanity_check_step()
        self.assertEqual(sorted(paths[-1]['files']), ['bin/.cs-run', 'bin/greet', 'etc/enroot-no-entrypoint.conf',
                                                      'etc/site.env', 'image.sif', 'image.sqsh'])
        for rel in paths[-1]['files']:
            self.assertTrue(os.path.isfile(os.path.join(eb.installdir, rel)), rel)
        # without sqsh there is no enroot, so no hook to require
        eb.cfg['formats'] = ['sif']
        eb.sanity_check_step()
        self.assertNotIn('etc/enroot-no-entrypoint.conf', paths[-1]['files'])

    def test_containerstack_smoke_tests(self):
        """Every smoke test runs in every format via bin/.cs-run; a missing GPU fails unless explicitly allowed."""
        eb, _ = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        self._cs_gpu(False)
        self.cs_calls = []
        self.assertRaisesRegex(EasyBuildError, "GPU smoke test 'nvidia-smi -L' cannot run: no GPU", eb.run_smoke_tests)
        self.assertEqual([c for c in self.cs_calls if '.cs-run' in c], [])    # fails before running anything
        self._cs_gpu(True)
        results = eb.run_smoke_tests()
        self.assertEqual([(r['format'], r['kind'], r['result']) for r in results],
                         [('sqsh', 'cpu', 'pass'), ('sqsh', 'gpu', 'pass'), ('sif', 'cpu', 'pass'),
                          ('sif', 'gpu', 'pass')])
        runner = os.path.join(eb.installdir, 'bin', '.cs-run')
        self.assertTrue(any(c.startswith('CS_RUNTIME=apptainer %s sh -c ' % runner) for c in self.cs_calls))
        self.assertTrue(any(c.startswith('CS_RUNTIME=enroot ENROOT_DATA_PATH=') for c in self.cs_calls))
        self.cs_fail = {'test -f /etc/greeting': 1}
        self.assertRaisesRegex(EasyBuildError, r"cpu smoke test failed in sqsh \(exit 1\): test -f /etc/greeting",
                               eb.run_smoke_tests)
        self.cs_fail = {}
        eb, _ = self._cs_easyblock("require_gpu_tests = False\n")
        eb.fetch_step()
        self._cs_gpu(False)
        results = eb.run_smoke_tests()
        self.assertEqual(sorted({r['result'] for r in results if r['kind'] == 'gpu'}), ['not-run: no GPU'])

    def test_containerstack_smoke_failure_removes_enroot_container(self):
        """A failing enroot smoke test still removes the unpacked enroot container."""
        eb, _ = self._cs_easyblock()
        eb.fetch_step()
        self._cs_gpu(True)
        self.cs_fail = {'test -f /etc/greeting': 1}
        self.assertRaisesRegex(EasyBuildError, "cpu smoke test failed in sqsh", eb.run_smoke_tests)
        self.assertTrue(re.search(r'ENROOT_DATA_PATH=\S+ enroot remove -f', self.cs_calls[-1]), self.cs_calls[-1])

    def test_containerstack_require_gpu_without_gpu_tests(self):
        """require_gpu_tests = True with no 'gpu' smoke test is a recipe error, with or without a GPU."""
        eb, _ = self._cs_easyblock("require_gpu_tests = True\nsmoke_tests = [('cpu', 'true')]\n")
        eb.fetch_step()
        for present in (False, True):
            self._cs_gpu(present)
            self.assertRaisesRegex(EasyBuildError, "require_gpu_tests is set but no smoke test is tagged gpu",
                                   eb.run_smoke_tests)

    def test_containerstack_no_manifest_on_failure(self):
        """manifest.json appears only after every smoke test passed, and records what was installed."""
        EasyBlock.sanity_check_step = lambda self, *args, **kwargs: None
        eb, img = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        self._cs_gpu(True)
        manifest_path = os.path.join(eb.installdir, 'manifest.json')
        self.cs_fail = {'nvidia-smi -L': 3}
        self.assertRaisesRegex(EasyBuildError, "smoke test failed", eb.sanity_check_step)
        self.assertFalse(os.path.exists(manifest_path))
        self.cs_fail = {}
        eb.sanity_check_step()
        manifest = json.loads(read_file(manifest_path))
        self.assertEqual((manifest['digest'], manifest['arch'], manifest['layers']),
                         (img['manifest'], 'x86_64', [img['layer']]))
        self.assertEqual(sorted(manifest['files']), ['image.sif', 'image.sqsh'])
        self.assertEqual(len(manifest['files']['image.sif']['sha256']), 71)
        self.assertEqual(len(manifest['smoke_tests']), 4)
        self.assertEqual(manifest['runtimes']['apptainer'], 'fake 1.0')

    def test_containerstack_smoke_logs(self):
        """A passing sanity check leaves each smoke test's output per format in smoke/, read-only like the rest."""
        EasyBlock.sanity_check_step = lambda self, *args, **kwargs: None
        eb, _ = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        self._cs_gpu(True)
        eb.sanity_check_step()
        smoke = os.path.join(eb.installdir, 'smoke')
        logs = ['1-cpu.sif.log', '1-cpu.sqsh.log', '2-gpu.sif.log', '2-gpu.sqsh.log']
        self.assertEqual(sorted(os.listdir(smoke)), logs)
        for log in logs:
            self.assertEqual(read_file(os.path.join(smoke, log)), 'fake 1.0\n', log)
        eb.permissions_step()
        if os.geteuid() != 0:
            self.assertFalse(os.access(os.path.join(smoke, logs[0]), os.W_OK))

    def test_containerstack_module_only(self):
        """--module-only (promotion) checks the existing manifest and never re-runs or rewrites anything."""
        EasyBlock.sanity_check_step = lambda self, *args, **kwargs: None
        eb, img = self._cs_easyblock()
        manifest_path = os.path.join(eb.installdir, 'manifest.json')
        update_build_option('module_only', True)
        self.assertRaisesRegex(EasyBuildError, "has no manifest.json", eb.sanity_check_step)
        write_file(manifest_path, json.dumps({'digest': 'sha256:' + 'b' * 64}))
        self.assertRaisesRegex(EasyBuildError, "built from sha256:b+ but the recipe pins", eb.sanity_check_step)
        write_file(manifest_path, json.dumps({'digest': img['manifest']}))
        self.cs_calls = []
        eb.sanity_check_step()
        self.assertEqual(self.cs_calls, [])
        self.assertEqual(json.loads(read_file(manifest_path)), {'digest': img['manifest']})
        # --sanity-check-only re-runs the smoke tests but still never rewrites the manifest
        update_build_option('module_only', False)
        update_build_option('sanity_check_only', True)
        self._cs_gpu(True)
        eb.sanity_check_step()
        self.assertTrue(any('.cs-run' in c for c in self.cs_calls))
        self.assertEqual(json.loads(read_file(manifest_path)), {'digest': img['manifest']})

    def test_containerstack_skip_existing(self):
        """eb --skip on an existing install only checks its manifest: no smoke tests, no rewrite."""
        EasyBlock.sanity_check_step = lambda self, *args, **kwargs: None
        eb, img = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        manifest_path = os.path.join(eb.installdir, 'manifest.json')
        write_file(manifest_path, json.dumps({'digest': img['manifest']}))
        smoke_runs = []
        eb.run_smoke_tests = lambda: smoke_runs.append(True) or []
        eb.skip = True
        eb.sanity_check_step()
        self.assertEqual(smoke_runs, [])
        self.assertEqual(json.loads(read_file(manifest_path)), {'digest': img['manifest']})
        write_file(manifest_path, json.dumps({'digest': 'sha256:' + 'b' * 64}))
        self.assertRaisesRegex(EasyBuildError, "built from sha256:b+ but the recipe pins", eb.sanity_check_step)

    def test_containerstack_module_only_force_warns(self):
        """--module-only --force skips the sanity check, so the manifest check too: the module step says so."""
        EasyBlock.make_module_step = lambda self, fake=False: 'modpath'
        eb, _ = self._cs_easyblock()
        update_build_option('module_only', True)
        for force, fake, warned in ((True, False, True), (False, False, False), (True, True, False)):
            update_build_option('force', force)
            with self.mocked_stdout_stderr() as (stdout, stderr):
                self.assertEqual(eb.make_module_step(fake=fake), 'modpath')
            output = stdout.getvalue() + stderr.getvalue()
            self.assertEqual("--force skips the manifest/digest check; do not promote with --force" in output,
                             warned, (force, fake, output))

    def test_containerstack_rebuild_readonly(self):
        """Installed files become read-only, and --rebuild can still replace the install."""
        eb, _ = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        eb.permissions_step()
        if os.geteuid() != 0:
            self.assertFalse(os.access(os.path.join(eb.installdir, 'image.sif'), os.W_OK))
        eb.make_installdir()
        self.assertEqual(os.listdir(eb.installdir), [])

    def test_containerstack_log_dir_writable(self):
        """permissions_step leaves installdir/easybuild writable, so EasyBuild can archive the recipe on promotion."""
        eb, _ = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        write_file(os.path.join(eb.installdir, 'manifest.json'), '{}')
        archived = os.path.join(eb.installdir, 'easybuild', 'tiny-container-1.0.eb')
        write_file(archived, 'hidden = True\n')
        eb.permissions_step()
        self.assertTrue(os.access(archived, os.W_OK))
        if os.geteuid() != 0:
            for rel in ('image.sif', 'manifest.json'):
                self.assertFalse(os.access(os.path.join(eb.installdir, rel), os.W_OK), rel)

    def test_containerstack_module_extra(self):
        """The module puts bin/ on PATH and sets <PREFIX>_SQSH, <PREFIX>_SIF and <PREFIX>_DIGEST."""
        eb, img = self._cs_easyblock()
        eb.fetch_step()
        eb.build_step()
        eb.install_step()
        eb.make_module_step(fake=True)
        txt = read_file(eb.module_generator.get_module_filepath(fake=True))
        with eb.module_generator.start_module_creation():
            self.assertIn(eb.module_generator.prepend_paths('PATH', ['bin']), txt)
        for var, value in (('TINYCONTAINER_SQSH', os.path.join(eb.installdir, 'image.sqsh')),
                           ('TINYCONTAINER_SIF', os.path.join(eb.installdir, 'image.sif')),
                           ('TINYCONTAINER_DIGEST', img['manifest'])):
            self.assertTrue(re.search(r'%s.*%s' % (var, re.escape(value)), txt), (var, txt))


def suite(loader):
    """Return all easyblock-specific tests."""
    return loader.loadTestsFromTestCase(EasyBlockSpecificTest)


if __name__ == '__main__':
    res = TextTestRunner(verbosity=1).run(suite(TestLoader()))
    sys.exit(len(res.failures))
