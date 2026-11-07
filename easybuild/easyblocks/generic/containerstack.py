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
