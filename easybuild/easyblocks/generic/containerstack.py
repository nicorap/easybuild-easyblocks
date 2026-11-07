##
# Copyright 2009-2026 Ghent University
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
EasyBuild support for deploying digest-pinned OCI container images as modules,
in enroot/pyxis (.sqsh) and Apptainer (.sif) formats.

@author: Nicolas Rapin (FAIC)
"""
import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import stat
import time
import urllib.error
import urllib.parse
import urllib.request

from easybuild.easyblocks import VERSION as EASYBLOCKS_VERSION
from easybuild.framework.easyblock import EasyBlock
from easybuild.framework.easyconfig import CUSTOM
from easybuild.tools.build_log import EasyBuildError, print_warning
from easybuild.tools.config import build_option, source_paths
from easybuild.tools.filetools import adjust_permissions, copy_file, mkdir, read_file, remove_dir, remove_file
from easybuild.tools.filetools import symlink, write_file
from easybuild.tools.run import run_shell_cmd
from easybuild.tools.systemtools import AARCH64, X86_64, get_cpu_architecture
from easybuild.tools.version import VERSION as EASYBUILD_VERSION

ARCH_KEYS = {X86_64: 'x86_64', AARCH64: 'aarch64'}
GOARCH = {'x86_64': 'amd64', 'aarch64': 'arm64'}
DIGEST_RE = re.compile(r'^sha256:[0-9a-f]{64}$')
DOCKER_HUB = 'registry-1.docker.io'
OCI_INDEX = 'application/vnd.oci.image.index.v1+json'
OCI_MANIFEST = 'application/vnd.oci.image.manifest.v1+json'
INDEX_TYPES = (OCI_INDEX, 'application/vnd.docker.distribution.manifest.list.v2+json')
MANIFEST_TYPES = (OCI_MANIFEST, 'application/vnd.docker.distribution.manifest.v2+json')
ACCEPT = ', '.join(MANIFEST_TYPES + INDEX_TYPES)
CHUNK = 1 << 20
FORMATS = ('sqsh', 'sif')
POLICIES = ('auto', 'apptainer', 'enroot', 'pyxis')
MANIFEST = 'manifest.json'
# paths an Apptainer sandbox adds to the image rootfs; removed before making the enroot .sqsh
APPTAINER_PATHS = ('.singularity.d', '.exec', '.run', '.shell', '.test', 'environment', 'singularity')
ENROOT_RC_BODY = '''mkdir -p "%(workdir)s" 2> /dev/null
cd "%(workdir)s" && unset OLDPWD || exit 1

if [ -s /etc/rc.local ]; then
    . /etc/rc.local
fi

if [ $# -gt 0 ]; then
    exec %(entrypoint)s "$@"
else
    exec %(entrypoint)s %(cmd)s
fi
'''
ENROOT_RC_LOCAL = ("# This file is sourced by /etc/rc when the container starts.\n"
                   "# It can be used to manipulate the entrypoint or the command of the container.\n")
ENROOT_DEFAULT_SQUASH_OPTIONS = '-comp lzo -noD -exit-on-error'


def host_arch_key():
    """Digest key ('x86_64' or 'aarch64') for the host architecture."""
    arch = get_cpu_architecture()
    if arch not in ARCH_KEYS:
        raise EasyBuildError("ContainerStack: unsupported host architecture %s", arch)
    return ARCH_KEYS[arch]


def select_digest(digests, arch_key):
    """Return the manifest digest for arch_key from the easyconfig's 'digests', or explain what is wrong."""
    digests = digests or {}
    if arch_key not in digests:
        raise EasyBuildError("ContainerStack: no digest for host architecture %s; recipe supports: %s",
                             arch_key, ', '.join(sorted(digests)) or 'none')
    digest = digests[arch_key]
    if not DIGEST_RE.match(digest or ''):
        raise EasyBuildError("ContainerStack: digest for %s is not of the form sha256:<64 hex>: %s", arch_key, digest)
    return digest


def split_image(image):
    """Split an image reference without tag/digest into (registry, repository)."""
    if '@' in image or re.search(r':[^/]+$', image):
        raise EasyBuildError("ContainerStack: 'image' must not contain a tag or digest (pin with 'digests'): %s",
                             image)
    parts = image.split('/')
    if len(parts) > 1 and ('.' in parts[0] or ':' in parts[0] or parts[0] == 'localhost'):
        registry, repo = parts[0], '/'.join(parts[1:])
    else:
        registry, repo = 'docker.io', image
    if registry == 'docker.io':
        registry = DOCKER_HUB
        if '/' not in repo:
            repo = 'library/' + repo
    return registry, repo


def env_var_prefix(name):
    """Prefix of the module's environment variables: name upper-cased, non-alphanumerics removed."""
    return re.sub(r'[^A-Z0-9]', '', name.upper())


def sha256_digest(data):
    """OCI digest of a bytes object."""
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def file_sha256(path):
    """OCI digest of a file, read in chunks."""
    sha = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b''):
            sha.update(chunk)
    return 'sha256:' + sha.hexdigest()


def disk_free(path):
    """Free bytes on the filesystem holding path."""
    return shutil.disk_usage(path).free


def registry_credentials(registry):
    """(user, password) for registry from $CONTAINERSTACK_REGISTRY_USER/_PASSWORD or Docker's config.json."""
    user, password = os.environ.get('CONTAINERSTACK_REGISTRY_USER'), os.environ.get('CONTAINERSTACK_REGISTRY_PASSWORD')
    if user and password:
        return user, password
    cfg = os.path.join(os.environ.get('DOCKER_CONFIG', os.path.expanduser('~/.docker')), 'config.json')
    if not os.path.isfile(cfg):
        return None
    try:
        auths = json.loads(read_file(cfg)).get('auths', {})
    except ValueError as err:
        raise EasyBuildError("ContainerStack: cannot parse %s: %s", cfg, err)
    keys = [registry, 'https://' + registry]
    if registry == DOCKER_HUB:
        keys += ['https://index.docker.io/v1/', 'docker.io']
    for key in keys:
        auth = auths.get(key, {}).get('auth')
        if auth:
            user, _, password = base64.b64decode(auth).decode().partition(':')
            return user, password
    return None


class RegistryClient:
    """Read-only client for the OCI distribution API: manifests and blobs by digest, bearer-token auth."""

    def __init__(self, registry, repo, scheme='https', timeout=300):
        self.registry = registry
        self.repo = repo
        self.base = '%s://%s/v2/%s' % (scheme, registry, repo)
        self.timeout = timeout
        self.token = None
        self.creds = registry_credentials(registry)

    def _open(self, url, accept):
        """GET url, answering one bearer-token challenge (new or expired token); returns the HTTP response.

        The token is sent as an unredirected header, so it never follows a redirect (e.g. a blob to a CDN).
        """
        for attempt in range(2):
            req = urllib.request.Request(url, headers={'Accept': accept})
            if self.token:
                req.add_unredirected_header('Authorization', 'Bearer ' + self.token)
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as err:
                challenge = err.headers.get('WWW-Authenticate', '')
                err.close()
                if err.code == 401 and attempt == 0:
                    self.token = None
                    self._authenticate(challenge)
                    continue
                raise EasyBuildError("ContainerStack: %s returned HTTP %s", url, err.code)
            except (urllib.error.URLError, OSError) as err:
                raise EasyBuildError("ContainerStack: cannot reach %s: %s", url, getattr(err, 'reason', err))
        raise EasyBuildError("ContainerStack: %s still unauthorized after authenticating", url)

    def _authenticate(self, challenge):
        """Get a pull token from the realm named in a 'WWW-Authenticate: Bearer ...' challenge."""
        scheme, _, params = challenge.partition(' ')
        fields = dict(re.findall(r'(\w+)="([^"]*)"', params))
        if scheme.lower() != 'bearer' or 'realm' not in fields:
            raise EasyBuildError("ContainerStack: unsupported auth challenge from %s: %r", self.registry, challenge)
        query = urllib.parse.urlencode({'service': fields.get('service', ''),
                                        'scope': fields.get('scope', 'repository:%s:pull' % self.repo)})
        req = urllib.request.Request(fields['realm'] + '?' + query)
        if self.creds:
            basic = base64.b64encode(('%s:%s' % self.creds).encode()).decode()
            req.add_unredirected_header('Authorization', 'Basic ' + basic)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                reply = json.loads(resp.read().decode())
        except urllib.error.HTTPError as err:
            err.close()
            raise EasyBuildError("ContainerStack: token request to %s failed: %s", fields['realm'], err)
        except (urllib.error.URLError, OSError, ValueError) as err:
            raise EasyBuildError("ContainerStack: token request to %s failed: %s", fields['realm'], err)
        self.token = reply.get('token') or reply.get('access_token')
        if not self.token:
            raise EasyBuildError("ContainerStack: %s returned no token", fields['realm'])

    def fetch_manifest(self, digest):
        """Fetch a per-architecture manifest by digest; returns (bytes, media_type) after checking its sha256."""
        with self._open('%s/manifests/%s' % (self.base, digest), ACCEPT) as resp:
            data = resp.read()
            content_type = resp.headers.get('Content-Type', '')
        if sha256_digest(data) != digest:
            raise EasyBuildError("ContainerStack: registry returned a manifest with sha256 %s, not the requested %s",
                                 sha256_digest(data), digest)
        doc = json.loads(data.decode())
        media_type = doc.get('mediaType') or content_type.split(';')[0].strip()
        if media_type in INDEX_TYPES or 'manifests' in doc:
            raise EasyBuildError("ContainerStack: %s is a multi-arch index, not a per-architecture manifest; "
                                 "pin the per-architecture digests (RegistryClient.platform_digests)", digest)
        return data, media_type

    def fetch_blob(self, digest, dest):
        """Stream blob digest to dest via dest.part, checking its sha256; nothing is left behind on mismatch."""
        part = dest + '.part'
        sha = hashlib.sha256()
        with self._open('%s/blobs/%s' % (self.base, digest), '*/*') as resp, open(part, 'wb') as fh:
            for chunk in iter(lambda: resp.read(CHUNK), b''):
                sha.update(chunk)
                fh.write(chunk)
        got = 'sha256:' + sha.hexdigest()
        if got != digest:
            os.remove(part)
            raise EasyBuildError("ContainerStack: blob %s downloaded with sha256 %s; refusing it", digest, got)
        os.rename(part, dest)

    def platform_digests(self, ref):
        """Per-architecture manifest digests ({'x86_64': ..., 'aarch64': ...}) of the multi-arch index ref."""
        with self._open('%s/manifests/%s' % (self.base, ref), ACCEPT) as resp:
            data = resp.read()
        doc = json.loads(data.decode())
        if 'manifests' not in doc:
            raise EasyBuildError("ContainerStack: %s is a single-architecture image (digest %s); "
                                 "pin that digest under its architecture", ref, sha256_digest(data))
        digests = {}
        for entry in doc['manifests']:
            platform = entry.get('platform', {})
            for key, goarch in GOARCH.items():
                if platform.get('os') == 'linux' and platform.get('architecture') == goarch:
                    digests.setdefault(key, entry['digest'])
        return digests


def blob_path(blobs_dir, digest):
    """Path of blob digest in a content-addressed store (an OCI layout's blobs/sha256 directory)."""
    if not DIGEST_RE.match(digest or ''):
        raise EasyBuildError("ContainerStack: invalid blob digest %r (expected sha256:<64 hex>); refusing to use it "
                             "as a path in %s", digest, blobs_dir)
    return os.path.join(blobs_dir, digest.split(':', 1)[1])


def verify_blob(path, digest):
    """True if the file at path has the given digest."""
    return file_sha256(path) == digest


def fetch_oci_blobs(client, digest, blobs_dir):
    """
    Fetch manifest digest and the config and layers it references into blobs_dir, checking every sha256.
    Cached blobs are re-verified; corrupt or half-downloaded ones are fetched again. Returns the manifest.
    """
    mkdir(blobs_dir, parents=True)
    path = blob_path(blobs_dir, digest)
    if os.path.isfile(path) and verify_blob(path, digest):
        with open(path, 'rb') as fh:
            data = fh.read()
    else:
        data, _ = client.fetch_manifest(digest)
        with open(path, 'wb') as fh:
            fh.write(data)
    manifest = json.loads(data.decode())
    for blob in [manifest['config']] + manifest['layers']:
        target = blob_path(blobs_dir, blob['digest'])
        if os.path.isfile(target):
            if verify_blob(target, blob['digest']):
                continue
            print_warning("ContainerStack: cached blob %s is corrupt; fetching it again" % blob['digest'])
            remove_file(target)
        client.fetch_blob(blob['digest'], target)
    return manifest


def verify_oci_blobs(blobs_dir, digest):
    """Re-check the manifest and every blob it references; returns the manifest, or names the first bad blob."""
    def check(blob_digest):
        path = blob_path(blobs_dir, blob_digest)
        if not os.path.isfile(path):
            raise EasyBuildError("ContainerStack: blob %s is missing from %s", blob_digest, blobs_dir)
        if not verify_blob(path, blob_digest):
            raise EasyBuildError("ContainerStack: blob %s is corrupt (sha256 mismatch) in %s; run eb again to "
                                 "re-fetch it", blob_digest, blobs_dir)
        return path

    with open(check(digest), 'rb') as fh:
        manifest = json.loads(fh.read().decode())
    for blob in [manifest['config']] + manifest['layers']:
        check(blob['digest'])
    return manifest


def write_oci_layout(layout_dir, blobs_dir, digest, media_type):
    """OCI image layout in layout_dir for manifest digest, whose blobs/sha256 is a symlink to blobs_dir."""
    mkdir(os.path.join(layout_dir, 'blobs'), parents=True)
    link = os.path.join(layout_dir, 'blobs', 'sha256')
    if os.path.lexists(link):
        remove_file(link)
    symlink(blobs_dir, link)
    write_file(os.path.join(layout_dir, 'oci-layout'), json.dumps({'imageLayoutVersion': '1.0.0'}))
    size = os.path.getsize(blob_path(blobs_dir, digest))
    index = {'schemaVersion': 2, 'manifests': [{'mediaType': media_type, 'digest': digest, 'size': size}]}
    write_file(os.path.join(layout_dir, 'index.json'), json.dumps(index))


def bash_quote(word):
    """Quote word the way bash's ${var@Q} does for printable strings."""
    return "'" + word.replace("'", "'\\''") + "'"


def enroot_files(image_config, provenance):
    """
    The /etc files 'enroot import' writes for an image config (enroot 4.2.0, docker::configure),
    so a .sqsh built from our own verified layers behaves exactly like one enroot imported itself.
    """
    cfg = image_config.get('config') or {}
    fstab = ''.join('${ENROOT_ROOTFS}%s %s none x-create=dir,bind,rw,nosuid,nodev\n' % (vol, vol)
                    for vol in sorted(cfg.get('Volumes') or {}))
    environment = ''.join('%s\n' % var for var in cfg.get('Env') or [])
    rc = '# enroot-provenance: %s\n' % provenance
    for key, value in (cfg.get('Labels') or {}).items():
        rc += '# %s %s\n' % (key, str(value).replace('\n', ' '))
    rc += '\n'
    entrypoint, cmd = cfg.get('Entrypoint') or [], cfg.get('Cmd') or []
    if not entrypoint and not cmd:
        cmd = ['/bin/sh']
    rc += ENROOT_RC_BODY % {'workdir': cfg.get('WorkingDir') or '/',
                            'entrypoint': ' '.join(bash_quote(w) for w in entrypoint),
                            'cmd': ' '.join(bash_quote(w) for w in cmd)}
    return {'etc/fstab': fstab, 'etc/environment': environment, 'etc/rc': rc, 'etc/rc.local': ENROOT_RC_LOCAL}


def det_squash_options(configured=None):
    """mksquashfs options: easyconfig value, $ENROOT_SQUASH_OPTIONS, the system enroot.conf, enroot's default."""
    if configured:
        return configured
    if os.environ.get('ENROOT_SQUASH_OPTIONS'):
        return os.environ['ENROOT_SQUASH_OPTIONS']
    conf = os.path.join(os.environ.get('ENROOT_SYSCONF_PATH', '/etc/enroot'), 'enroot.conf')
    if os.path.isfile(conf):
        for line in read_file(conf).splitlines():
            match = re.match(r'^\s*ENROOT_SQUASH_OPTIONS\s+(.+?)\s*$', line)
            if match:
                return match.group(1)
    return ENROOT_DEFAULT_SQUASH_OPTIONS


WRAPPER_TEMPLATE = r'''#!/bin/bash
# Generated by EasyBuild ContainerStack for %(name)s/%(version)s (%(digest)s).
# CS_RUNTIME=auto|apptainer|enroot|pyxis|sif|sqsh overrides the default (%(policy)s); CS_DEBUG=1 prints the choice;
# CS_SRUN_ARGS adds srun options for pyxis (word-split); APPTAINER_BIND adds Apptainer mounts.
# Under pyxis the wrapper starts ONE task, like the plain command (srun --nodes=1 --ntasks=1, even in a big batch job);
# CS_SRUN_ARGS comes after these and srun's later options win, e.g. CS_SRUN_ARGS="--nodes=2 --ntasks=2" (one per node).
root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)" || exit 1
sqsh="$root/image.sqsh"
sif="$root/image.sif"
cname=%(cname)s
mounts=(%(mounts)s)
cmd=(%(cmd)s)
names=()
if [ -r "$root/etc/site.env" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # only NAME=VALUE lines; comments, blank lines and anything else are skipped
        name="${line%%%%=*}"
        case "$line" in *=*) ;; *) continue ;; esac
        case "$name" in '' | [0-9]* | *[!A-Za-z0-9_]*) continue ;; esac
        # Apptainer lets image ENV beat host env; APPTAINERENV_<NAME> beats image ENV
        export "$line" "APPTAINERENV_$line"
        names+=("$name")
    done < "$root/etc/site.env"
fi
runtime="${CS_RUNTIME:-%(policy)s}"
case "$runtime" in sif) runtime=apptainer ;; sqsh) runtime=enroot ;; esac
if [ "$runtime" = auto ]; then
    # pyxis attaches a container only when a step launches: use it in an allocation, but not inside a step
    if [ -n "$SLURM_JOB_ID" ] && [ -z "$SLURM_STEP_ID" ] && [ -e "$sqsh" ] && command -v srun >/dev/null 2>&1; then
        runtime=pyxis
    elif [ -e "$sif" ]; then
        runtime=apptainer
    else
        runtime=enroot
    fi
fi
[ -n "$CS_DEBUG" ] && echo "ContainerStack: runtime=$runtime" >&2
gpu=
[ -e "${CS_GPU_DEVICE:-/dev/nvidia0}" ] && gpu=1
join() { local IFS=,; echo "$*"; }
case "$runtime" in
    pyxis)
        args=(--container-image="$sqsh")
        [ ${#mounts[@]} -gt 0 ] && args+=(--container-mounts="$(join "${mounts[@]}")")
        [ ${#names[@]} -gt 0 ] && args+=(--container-env="$(join "${names[@]}")")
        exec srun --nodes=1 --ntasks=1 $CS_SRUN_ARGS "${args[@]}" "${cmd[@]}" "$@" ;;
    apptainer)
        command -v apptainer >/dev/null 2>&1 || { echo "ContainerStack: apptainer not found on PATH" >&2; exit 127; }
        args=()
        [ -n "$gpu" ] && args+=(--nv)
        [ ${#mounts[@]} -gt 0 ] && args+=(--bind "$(join "${mounts[@]}")")
        exec apptainer exec "${args[@]}" "$sif" "${cmd[@]}" "$@" ;;
    enroot)
        command -v enroot >/dev/null 2>&1 || { echo "ContainerStack: enroot not found on PATH" >&2; exit 127; }
        enroot list | grep -qxF "$cname" || enroot create -n "$cname" "$sqsh" >/dev/null || exit 1
        args=()
        [ -n "$gpu" ] && args+=(--env NVIDIA_VISIBLE_DEVICES=all)
        for m in "${mounts[@]}"; do args+=(--mount "$m"); done
        for n in "${names[@]}"; do args+=(--env "$n"); done
        exec enroot start "${args[@]}" "$cname" "${cmd[@]}" "$@" ;;
    *)
        echo "ContainerStack: unknown CS_RUNTIME '$runtime' (use auto, apptainer, enroot, pyxis, sif or sqsh)" >&2
        exit 2 ;;
esac
'''


def render_wrapper(name, version, digest, command, policy, binds):
    """Bash wrapper that runs command (shell words; '' runs the wrapper's own arguments) in the container."""
    if policy not in POLICIES:
        raise EasyBuildError("ContainerStack: launch_policy must be one of %s, not %r", ', '.join(POLICIES), policy)
    mounts = [bind if ':' in bind else '%s:%s' % (bind, bind) for bind in binds]
    cname = re.sub(r'[^A-Za-z0-9_.-]', '_', 'cs-%s-%s-%s' % (name, version, digest.split(':')[-1][:12]))
    return WRAPPER_TEMPLATE % {
        'name': name, 'version': version, 'digest': digest, 'policy': policy,
        'cname': shlex.quote(cname),
        'mounts': ' '.join(shlex.quote(m) for m in mounts),
        'cmd': ' '.join(shlex.quote(w) for w in shlex.split(command)),
    }


class ContainerStack(EasyBlock):
    """Deploy a digest-pinned OCI image as a module, in enroot/pyxis (.sqsh) and Apptainer (.sif) formats."""

    @staticmethod
    def extra_options(extra_vars=None):
        """Easyconfig parameters of ContainerStack."""
        extra_vars = EasyBlock.extra_options(extra_vars)
        extra_vars.update({
            'binds': [[], "Host paths to mount in the container ('path' or 'src:dst')", CUSTOM],
            'digests': [{}, "Per-architecture manifest digests: {'x86_64': 'sha256:...', 'aarch64': 'sha256:...'}",
                        CUSTOM],
            'formats': [['sqsh', 'sif'], "Formats to produce: 'sqsh' (enroot/pyxis) and/or 'sif' (Apptainer)",
                        CUSTOM],
            'image': [None, "Image reference without tag or digest, e.g. 'nvcr.io/nvidia/pytorch'", CUSTOM],
            'launch_policy': ['auto', "Default wrapper runtime: auto, apptainer, enroot or pyxis", CUSTOM],
            'launchers': [{}, "Wrappers in bin/: {'wrapper name': 'command inside the container'}", CUSTOM],
            'min_free_space_factor': [3, "Free space needed in the build directory, as a multiple of the image size",
                                      CUSTOM],
            'registry_scheme': ['https', "Registry URL scheme ('http' only for local test registries)", CUSTOM],
            'require_gpu_tests': [None, "Fail when 'gpu' smoke tests cannot run (default: True if any exist)",
                                  CUSTOM],
            'site_env': [{}, "Environment variables set in the container: {'NAME': 'value'}", CUSTOM],
            'smoke_tests': [[], "Run in every format during the sanity check: [('cpu' or 'gpu', 'command'), ...]",
                            CUSTOM],
            'squash_options': [None, "mksquashfs options (default: $ENROOT_SQUASH_OPTIONS or enroot.conf)", CUSTOM],
        })
        return extra_vars

    def __init__(self, *args, **kwargs):
        """Initialise; easyconfig values are validated when first needed, not here."""
        super().__init__(*args, **kwargs)
        self.digest = None
        self.manifest = None
        self.blobs_dir = None
        self.registry = self.repo = self.arch_key = None

    def _validate(self):
        """Check the easyconfig and resolve image, architecture and digest."""
        if not self.cfg['image']:
            raise EasyBuildError("ContainerStack: easyconfig parameter 'image' is required")
        self.registry, self.repo = split_image(self.cfg['image'])
        self.arch_key = host_arch_key()
        self.digest = select_digest(self.cfg['digests'], self.arch_key)
        formats = self.cfg['formats'] or []
        if not formats or any(fmt not in FORMATS for fmt in formats):
            raise EasyBuildError("ContainerStack: unknown format in %s (use 'sqsh' and/or 'sif')", formats)
        if self.cfg['launch_policy'] not in POLICIES:
            raise EasyBuildError("ContainerStack: launch_policy must be one of %s, not %r",
                                 ', '.join(POLICIES), self.cfg['launch_policy'])
        for name in self.cfg['launchers']:
            if not re.match(r'^[A-Za-z0-9][A-Za-z0-9._+-]*$', name):
                raise EasyBuildError("ContainerStack: invalid launcher name %r", name)
        for key, value in self.cfg['site_env'].items():
            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', key) or not isinstance(value, str) or '\n' in value:
                raise EasyBuildError("ContainerStack: site_env needs NAME: 'single-line string', got %r: %r",
                                     key, value)
        for test in self.cfg['smoke_tests']:
            if not (isinstance(test, (list, tuple)) and len(test) == 2 and test[0] in ('cpu', 'gpu')):
                raise EasyBuildError("ContainerStack: smoke test must be ('cpu' or 'gpu', 'command'): %r", test)

    def _blobs_dir(self):
        """Shared blob store <sourcepath>/<letter>/<name>/oci/blobs/sha256 (first source path that has one)."""
        name = self.cfg['name']
        rel = os.path.join(name[0].lower(), name, 'oci', 'blobs', 'sha256')
        paths = source_paths()
        for path in paths:
            if os.path.isdir(os.path.join(path, rel)):
                return os.path.join(path, rel)
        return os.path.join(paths[0], rel)

    def fetch_step(self, skip_checksums=False):
        """Fetch the image's manifest, config and layers by digest, verifying every blob."""
        self._validate()
        super().fetch_step(skip_checksums=skip_checksums)
        self.blobs_dir = self._blobs_dir()
        if self.dry_run:
            self.dry_run_msg("Fetching %s/%s@%s into %s\n", self.registry, self.repo, self.digest, self.blobs_dir)
            return
        client = RegistryClient(self.registry, self.repo, scheme=self.cfg['registry_scheme'])
        self.manifest = fetch_oci_blobs(client, self.digest, self.blobs_dir)

    def configure_step(self):
        """Nothing to configure."""
        pass

    def build_step(self):
        """Build the requested formats from the verified OCI layout, inside the build directory."""
        self.manifest = verify_oci_blobs(self.blobs_dir, self.digest)
        need = sum(layer['size'] for layer in self.manifest['layers']) * self.cfg['min_free_space_factor']
        free = disk_free(self.builddir)
        if free < need:
            raise EasyBuildError("ContainerStack: %.1f GiB free in %s, need %.1f GiB (%dx the image size)",
                                 free / 2.0 ** 30, self.builddir, need / 2.0 ** 30, self.cfg['min_free_space_factor'])
        layout = os.path.join(self.builddir, 'oci-layout')
        write_oci_layout(layout, self.blobs_dir, self.digest, self.manifest.get('mediaType', OCI_MANIFEST))
        tmp = os.path.join(self.builddir, 'tmp')
        mkdir(tmp, parents=True)
        env = 'APPTAINER_TMPDIR=%s APPTAINER_CACHEDIR=%s ' % (shlex.quote(tmp), shlex.quote(os.path.join(tmp, 'c')))
        if 'sif' in self.cfg['formats']:
            sif = os.path.join(self.builddir, 'image.sif')
            run_shell_cmd(env + 'apptainer build --force %s oci:%s' % (shlex.quote(sif), shlex.quote(layout)))
        if 'sqsh' in self.cfg['formats']:
            self._build_sqsh(env, layout)

    def _build_sqsh(self, env, layout):
        """Rootfs via an Apptainer sandbox, minus Apptainer's files, plus enroot's /etc files, squashed."""
        rootfs = os.path.join(self.builddir, 'rootfs')
        run_shell_cmd(env + 'apptainer build --force --sandbox %s oci:%s' % (shlex.quote(rootfs), shlex.quote(layout)))
        for rel in APPTAINER_PATHS:
            path = os.path.join(rootfs, rel)
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                remove_dir(path)
        with open(blob_path(self.blobs_dir, self.manifest['config']['digest']), 'rb') as fh:
            config = json.loads(fh.read().decode())
        provenance = 'containerstack://%s@%s' % (self.cfg['image'], self.digest)
        for rel, text in enroot_files(config, provenance).items():
            write_file(os.path.join(rootfs, rel), text)
        workdir = (config.get('config') or {}).get('WorkingDir') or '/'
        mkdir(os.path.join(rootfs, workdir.lstrip('/')), parents=True)
        sqsh = os.path.join(self.builddir, 'image.sqsh')
        run_shell_cmd('mksquashfs %s %s -noappend -all-root -no-progress %s'
                      % (shlex.quote(rootfs), shlex.quote(sqsh), det_squash_options(self.cfg['squash_options'])))
        adjust_permissions(rootfs, stat.S_IWUSR, add=True, recursive=True)
        remove_dir(rootfs)

    def install_step(self):
        """Copy the images, write the wrappers and etc/site.env (manifest.json comes after the smoke tests)."""
        bindir = os.path.join(self.installdir, 'bin')
        mkdir(bindir, parents=True)
        for fmt in self.cfg['formats']:
            copy_file(os.path.join(self.builddir, 'image.' + fmt), os.path.join(self.installdir, 'image.' + fmt))
        site_env = ''.join('%s=%s\n' % item for item in sorted(self.cfg['site_env'].items()))
        write_file(os.path.join(self.installdir, 'etc', 'site.env'), site_env)
        launchers = dict(self.cfg['launchers'], **{'.cs-run': ''})
        for name, command in sorted(launchers.items()):
            path = os.path.join(bindir, name)
            write_file(path, render_wrapper(self.name, self.version, self.digest, command,
                                            self.cfg['launch_policy'], self.cfg['binds']))
            adjust_permissions(path, stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH, add=True)

    def _smoke_cmd(self, fmt, command):
        """Shell command running one smoke test in one format through bin/.cs-run."""
        runner = shlex.quote(os.path.join(self.installdir, 'bin', '.cs-run'))
        if fmt == 'sif':
            prefix = 'CS_RUNTIME=apptainer'
        else:
            prefix = 'CS_RUNTIME=enroot ENROOT_DATA_PATH=%s' % shlex.quote(os.path.join(self.builddir, 'enroot-data'))
        return '%s %s sh -c %s' % (prefix, runner, shlex.quote(command))

    def run_smoke_tests(self):
        """Run every smoke test in every format; returns the results, raises on the first failure."""
        tests = [tuple(test) for test in self.cfg['smoke_tests']]
        require_gpu = self.cfg['require_gpu_tests']
        if require_gpu is None:
            require_gpu = any(kind == 'gpu' for kind, _ in tests)
        gpu = os.path.exists(os.environ.get('CS_GPU_DEVICE', '/dev/nvidia0'))
        if require_gpu and not gpu:
            command = [cmd for kind, cmd in tests if kind == 'gpu'][0]
            raise EasyBuildError("ContainerStack: GPU smoke test %r cannot run: no GPU on %s; build on a GPU node "
                                 "or set require_gpu_tests = False", command, socket.gethostname())
        results = []
        for fmt in self.cfg['formats']:
            for kind, command in tests:
                result = {'format': fmt, 'kind': kind, 'command': command,
                          'date': time.strftime('%Y-%m-%dT%H:%M:%S')}
                if kind == 'gpu' and not gpu:
                    result['result'] = 'not-run: no GPU'
                else:
                    start = time.time()
                    res = run_shell_cmd(self._smoke_cmd(fmt, command), fail_on_error=False)
                    if res.exit_code != 0:
                        raise EasyBuildError("ContainerStack: %s smoke test failed in %s (exit %s): %s\n%s",
                                             kind, fmt, res.exit_code, command, (res.output or '')[-2000:])
                    result.update(result='pass', seconds=round(time.time() - start, 1))
                results.append(result)
            if fmt == 'sqsh':
                data = shlex.quote(os.path.join(self.builddir, 'enroot-data'))
                run_shell_cmd('ENROOT_DATA_PATH=%s enroot remove -f $(ENROOT_DATA_PATH=%s enroot list) || true'
                              % (data, data), fail_on_error=False)
        return results

    def runtime_versions(self):
        """Versions of the tools that produced the images."""
        versions = {}
        for key, cmd in (('apptainer', 'apptainer --version'), ('enroot', 'enroot version'),
                         ('mksquashfs', 'mksquashfs -version')):
            res = run_shell_cmd(cmd, fail_on_error=False, hidden=True)
            out = (res.output or '').strip()
            versions[key] = out.splitlines()[0] if res.exit_code == 0 and out else 'unavailable'
        return versions

    def write_manifest(self, path, results):
        """Write manifest.json: what was installed, from which blobs, with which tools, and the smoke results."""
        files = {}
        for fmt in self.cfg['formats']:
            image = os.path.join(self.installdir, 'image.' + fmt)
            files['image.' + fmt] = {'size': os.path.getsize(image), 'sha256': file_sha256(image)}
        data = {
            'name': self.name, 'version': self.version, 'image': self.cfg['image'],
            'registry': self.registry, 'repository': self.repo, 'arch': self.arch_key, 'digest': self.digest,
            'config_digest': self.manifest['config']['digest'],
            'layers': [layer['digest'] for layer in self.manifest['layers']],
            'files': files, 'runtimes': self.runtime_versions(), 'smoke_tests': results,
            'easybuild': str(EASYBUILD_VERSION), 'easyblocks': str(EASYBLOCKS_VERSION),
        }
        write_file(path, json.dumps(data, indent=2, sort_keys=True) + '\n')

    def check_existing_manifest(self, path):
        """For --module-only / --sanity-check-only: the install must be complete and match the pinned digest."""
        if not os.path.isfile(path):
            raise EasyBuildError("ContainerStack: %s has no manifest.json: not a complete install (rebuild it)",
                                 self.installdir)
        data = json.loads(read_file(path))
        if data.get('digest') != self.digest:
            raise EasyBuildError("ContainerStack: %s was built from %s but the recipe pins %s",
                                 self.installdir, data.get('digest'), self.digest)
        return data

    def sanity_check_step(self, *args, **kwargs):
        """Check files, run the smoke tests, then (and only then) write manifest.json."""
        if self.digest is None:
            self._validate()
        files = ['image.%s' % fmt for fmt in self.cfg['formats']] + ['bin/.cs-run', 'etc/site.env']
        files += [os.path.join('bin', name) for name in sorted(self.cfg['launchers'])]
        super().sanity_check_step(custom_paths={'files': files, 'dirs': []})
        manifest_path = os.path.join(self.installdir, MANIFEST)
        if build_option('module_only') or self.cfg['module_only'] or build_option('sanity_check_only'):
            self.check_existing_manifest(manifest_path)
            if build_option('sanity_check_only'):
                self.run_smoke_tests()
            return
        results = self.run_smoke_tests()
        self.write_manifest(manifest_path, results)

    def permissions_step(self):
        """Standard permissions, then make the installed files read-only (directories stay writable)."""
        super().permissions_step()
        adjust_permissions(self.installdir, stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH, add=False, onlyfiles=True,
                           recursive=True)

    def make_module_extra(self, *args, **kwargs):
        """Module variables <PREFIX>_SQSH, <PREFIX>_SIF and <PREFIX>_DIGEST (bin/ is on PATH by default)."""
        if self.digest is None:
            self._validate()
        txt = super().make_module_extra(*args, **kwargs)
        prefix = env_var_prefix(self.name)
        for fmt in self.cfg['formats']:
            txt += self.module_generator.set_environment('%s_%s' % (prefix, fmt.upper()),
                                                         os.path.join(self.installdir, 'image.' + fmt))
        txt += self.module_generator.set_environment(prefix + '_DIGEST', self.digest)
        return txt
