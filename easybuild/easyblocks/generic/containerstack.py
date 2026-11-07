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
