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
import contextlib
import errno
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from easybuild.easyblocks import VERSION as EASYBLOCKS_VERSION
from easybuild.framework.easyblock import EasyBlock
from easybuild.framework.easyconfig import CUSTOM
from easybuild.tools.build_log import EasyBuildError, print_msg, print_warning
from easybuild.tools.config import IGNORE, build_option, log_path, source_paths
from easybuild.tools.filetools import adjust_permissions, copy_file, mkdir, read_file, remove_dir, remove_file
from easybuild.tools.filetools import symlink, which, write_file
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
SMOKE_DIR = 'smoke'
# paths an Apptainer sandbox adds to the image rootfs; removed before making the enroot .sqsh
APPTAINER_PATHS = ('.singularity.d', '.exec', '.run', '.shell', '.test', 'environment', 'singularity')
# bind targets an Apptainer sandbox build empties or creates; put back exactly as the image layers have them
APPTAINER_BIND_TARGETS = ('etc/hosts', 'etc/resolv.conf')
OCI_LAYER_ZSTD = 'application/vnd.oci.image.layer.v1.tar+zstd'
# enroot's fixperms runs 'find . -maxdepth 15' in the directory holding one directory per layer, so it reaches image
# directories at most 14 levels below the rootfs
ENROOT_FIXPERMS_DIR_DEPTH = 14
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
# 'enroot start --conf' file for the wrappers. hooks(): enroot runs the command and not the image's entrypoint, like
# Apptainer's exec and pyxis' default (the same hook): /etc/rc sources /etc/rc.local before it
# execs the entrypoint; with no command, 'exec' is a no-op and /etc/rc goes on as usual.
# environ(): the caller's environment (the names the wrapper lists in CS_CALLER_ENV) goes in, as pyxis and Apptainer
# pass it on. As under pyxis: what the image's /etc/environment or enroot's environ.d already set (in $ENROOT_ENVIRON
# by now) keeps its value, the --env names (etc/site.env) come after and win, and PATH, LD_LIBRARY_PATH, LD_PRELOAD
# (pyxis never passes those to enroot), LANG, LANGUAGE, LC_ALL (pyxis takes the container's) and HOME, SHELL, USER,
# LOGNAME, TERM (enroot sets them for the container's user) are never the caller's. A value with a newline is skipped:
# enroot's environment file has one variable per line.
ENROOT_NO_ENTRYPOINT = os.path.join('etc', 'enroot-no-entrypoint.conf')
ENROOT_NO_ENTRYPOINT_CONF = r'''hooks() {
    echo 'exec "$@"' > "${ENROOT_ROOTFS}/etc/rc.local"
}

environ() {
    local name IFS=$' \t\n'
    for name in ${CS_CALLER_ENV-}; do
        case "${name}" in
            _ | CS_CALLER_ENV | PATH | LD_LIBRARY_PATH | LD_PRELOAD | LANG | LANGUAGE | LC_ALL | \
            HOME | SHELL | USER | LOGNAME | TERM) continue ;;
        esac
        [ -n "${!name+set}" ] || continue
        [[ "${!name}" != *$'\n'* ]] || continue
        grep -q "^[[:space:]]*${name}=" "${ENROOT_ENVIRON}" && continue
        printf '%s=%s\n' "${name}" "${!name}"
    done
    return 0
}
'''


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

    def fetch_blob(self, digest, dest, part=None):
        """
        Stream blob digest to dest via the file part (default dest.part), checking its sha256; nothing is left behind
        on mismatch. dest.part is truncated: in a shared store, use it only while holding the blob's blob_lock(), and
        pass a part of this run's own otherwise.
        """
        part = part or dest + '.part'
        sha = hashlib.sha256()
        try:
            with self._open('%s/blobs/%s' % (self.base, digest), '*/*') as resp, open(part, 'wb') as fh:
                for chunk in iter(lambda: resp.read(CHUNK), b''):
                    sha.update(chunk)
                    fh.write(chunk)
        except BaseException:
            if part != dest + '.part' and os.path.exists(part):    # a part of this run's own: nobody resumes it
                os.remove(part)
            raise
        got = 'sha256:' + sha.hexdigest()
        if got != digest:
            os.remove(part)
            raise EasyBuildError("ContainerStack: blob %s downloaded with sha256 %s; refusing it", digest, got)
        os.replace(part, dest)

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


# blob_lock(): seconds between attempts at a lock another process holds, and between 'still waiting' messages
LOCK_POLL_INTERVAL = 5
LOCK_PROGRESS_INTERVAL = 60
# errors that mean this user cannot open a lock file (another user's, a read-only store), and that the filesystem
# cannot lock at all (e.g. NFSv3 without lockd)
LOCK_DENIED = (errno.EACCES, errno.EPERM, errno.EROFS)
LOCK_UNSUPPORTED = tuple(set((errno.ENOLCK, errno.EOPNOTSUPP, getattr(errno, 'ENOTSUP', errno.EOPNOTSUPP),
                              errno.ENOSYS)))
# lock directories already warned about as unusable, so that a store without locks gets one warning, not one per blob
_UNLOCKED_STORES = set()


def warn_unlocked(lock_dir, path, err):
    """Warn (once per lock directory) that blobs are fetched without their lock."""
    if lock_dir not in _UNLOCKED_STORES:
        _UNLOCKED_STORES.add(lock_dir)
        print_warning("ContainerStack: cannot lock %s (%s); fetching without locks: concurrent runs sharing this "
                      "store are not serialised (every blob is still sha256-verified)" % (path, err))


@contextlib.contextmanager
def blob_lock(blobs_dir, digest):
    """
    Hold the exclusive lock on blob digest of the store blobs_dir, so that concurrent EasyBuild runs sharing the
    store (possibly on NFS) never write the same blob at once: a POSIX lock (fcntl.lockf, per process) on
    <blobs_dir>.locks/<hex>.lock, beside the store so that blobs_dir holds only blobs. Lock files are never removed
    (removing one another run may be waiting on is racy). Yields whether the lock is held.

    A lock another process holds is tried again every LOCK_POLL_INTERVAL seconds, with a message naming the lock
    file every LOCK_PROGRESS_INTERVAL seconds (so that a hung holder can be found); there is no timeout. Where the
    lock cannot be had at all (a lock file or directory this user may not write; a filesystem without POSIX locks),
    a warning says so once per store and this yields False: the caller then writes nothing another run may be
    writing too (see fetch_blob), and every blob is still sha256-verified.
    """
    lock_dir = os.path.normpath(blobs_dir) + '.locks'
    path = blob_path(lock_dir, digest) + '.lock'
    try:
        os.makedirs(lock_dir, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError as err:
        if err.errno not in LOCK_DENIED:
            raise EasyBuildError("ContainerStack: cannot open lock file %s: %s", path, err)
        warn_unlocked(lock_dir, path, err)
        yield False
        return
    try:
        locked, start, last = True, None, None
        while True:
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as err:
                if err.errno in LOCK_UNSUPPORTED:
                    warn_unlocked(lock_dir, path, err)
                    locked = False
                    break
                if err.errno not in (errno.EACCES, errno.EAGAIN):
                    raise EasyBuildError("ContainerStack: cannot lock %s: %s", path, err)
            now = time.monotonic()
            if start is None:
                start = last = now
                print_msg("ContainerStack: waiting for another EasyBuild run fetching blob %s (lock %s)"
                          % (digest, path))
            elif now - last >= LOCK_PROGRESS_INTERVAL:
                last = now
                print_msg("ContainerStack: still waiting after %d s for blob %s (lock %s)"
                          % (now - start, digest, path))
            time.sleep(LOCK_POLL_INTERVAL)
        yield locked
    finally:
        os.close(fd)    # releases the lock


def cached_blob(path, digest):
    """True if the store has blob digest at path, verified."""
    return os.path.isfile(path) and verify_blob(path, digest)


def download_part(path, locked):
    """
    File to download blob path to: path.part under the blob's lock, else a new file of this run's own (mkstemp,
    path.part-<random>, which no other run's file can be), made as readable as a plain new file (mkstemp's are 0600).
    """
    if locked:
        return path + '.part'
    fd, part = tempfile.mkstemp(dir=os.path.dirname(path), prefix=os.path.basename(path) + '.part-')
    try:
        os.fchmod(fd, 0o666 & ~current_umask())
    finally:
        os.close(fd)
    return part


@contextlib.contextmanager
def store_writes(blobs_dir, digest):
    """
    Turn a write into the store this user may not make (errno in LOCK_DENIED: a read-only store, another user's
    .part file) into an error saying blob digest is missing or corrupt there.
    """
    try:
        yield
    except OSError as err:
        if err.errno not in LOCK_DENIED:
            raise
        raise EasyBuildError("ContainerStack: cannot write to blob store %s (%s); blob %s is missing or corrupt there; "
                             "populate it from a run that can write there", blobs_dir, err.strerror, digest)


def fetch_oci_blobs(client, digest, blobs_dir):
    """
    Fetch manifest digest and the config and layers it references into blobs_dir, checking every sha256.
    Cached blobs are re-verified; corrupt or half-downloaded ones are fetched again. Returns the manifest.
    A verified cached blob is used without taking any lock, so a read-only store works: blobs are only ever written
    to a .part file then renamed into place, so a reader sees a whole blob or none. A blob that must be written is
    checked again and fetched under its blob_lock(): one another run is fetching is waited for, then reused. Only
    the lock holder writes a blob's .part file, so a .part found under the lock is stale; where locking is not
    possible, a run downloads to a file of its own. A blob missing or corrupt in a store this user cannot write to (a
    read-only site store), or whose .part file is another user's, is an error saying so (store_writes).
    """
    mkdir(blobs_dir, parents=True)
    path = blob_path(blobs_dir, digest)
    data = None
    if not cached_blob(path, digest):
        with blob_lock(blobs_dir, digest) as locked:
            if not cached_blob(path, digest):    # another run may have fetched it meanwhile
                data, _ = client.fetch_manifest(digest)
                with store_writes(blobs_dir, digest):
                    part = download_part(path, locked)
                    try:
                        with open(part, 'wb') as fh:
                            fh.write(data)
                        os.replace(part, path)    # atomic: a reader sees the old file or the new one
                    except BaseException:
                        if os.path.exists(part):
                            os.remove(part)
                        raise
    if data is None:
        with open(path, 'rb') as fh:
            data = fh.read()
    manifest = json.loads(data.decode())
    for blob in [manifest['config']] + manifest['layers']:
        target = blob_path(blobs_dir, blob['digest'])
        if cached_blob(target, blob['digest']):
            continue
        with blob_lock(blobs_dir, blob['digest']) as locked:
            if os.path.isfile(target):
                if verify_blob(target, blob['digest']):
                    continue
                print_warning("ContainerStack: cached blob %s is corrupt; fetching it again" % blob['digest'])
                with store_writes(blobs_dir, blob['digest']):
                    os.remove(target)
            with store_writes(blobs_dir, blob['digest']):
                client.fetch_blob(blob['digest'], target, part=download_part(target, locked))
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


def read_image_config(blobs_dir, manifest):
    """The image config (a dict) of a verified manifest."""
    with open(blob_path(blobs_dir, manifest['config']['digest']), 'rb') as fh:
        return json.loads(fh.read().decode())


def image_user(image_config):
    """
    The User the image config names (a string; None if unset or empty), and whether that is a user other than root.
    As in the OCI image spec, User is 'user', 'uid', 'user:group' or 'uid:gid' (or a mix), and empty means the
    default user, root; so does an empty user part (':1000': root, with group 1000), as runc reads it. Root is uid 0
    or the name 'root'.
    """
    user = (image_config.get('config') or {}).get('User')
    user = '' if user is None else str(user).strip()
    if not user:
        return None, False
    name = user.split(':', 1)[0]
    return user, not (name in ('', 'root') or (name.isdigit() and int(name) == 0))


def layer_path(name):
    """Image path of a layer member name or hard-link target ('./etc/x', '/etc/x' -> 'etc/x')."""
    return os.path.normpath('/' + name).lstrip('/')


def hard_link_error(layer, path, link):
    """The error for a hard link that does not resolve to a file before it in its layer."""
    return EasyBuildError("ContainerStack: hard link %s in layer %s links to %s, which is not a file before it in "
                          "that layer", path, layer['digest'], layer_path(link.linkname))


# hard_link_member(): links (hard or symbolic) followed at most, as the kernel's limit on nested symlinks
LINK_FOLLOW_MAX = 40


def hard_link_member(layer, path, link, index):
    """
    The file member whose data hard link 'link' (the member for image path 'path') has, in a layer whose members are
    index ({image path: [TarInfo, ...]}, in layer order). The one resolver for every kind of layer, with names and
    targets normalised by layer_path ('./etc/a' and '/etc/a' are etc/a), as tarfile reads a link from a seekable
    tar: a hard link is the last member of its target's name before it, a symlink the last member of the name it
    points to in the whole layer; links are followed in turn, at most LINK_FOLLOW_MAX of them (a symlink loop never
    ends: more is an error saying so, like ELOOP). So a hard link to a symlink has the data of the file the symlink
    points to, relative to the symlink's directory or, for an absolute target, to the image root (tarfile would join
    that to the directory too).
    """
    member = link
    for _ in range(LINK_FOLLOW_MAX):
        if member.islnk():
            name, before = layer_path(member.linkname), member.offset
        else:
            target = member.linkname
            if not target.startswith('/'):
                target = os.path.join(os.path.dirname(layer_path(member.name)), target)
            name, before = layer_path(target), None
        candidates = [entry for entry in index.get(name, []) if before is None or entry.offset < before]
        if not candidates:
            break
        member = candidates[-1]
        if member.isfile():
            return member
        if not (member.islnk() or member.issym()):
            break
    else:
        raise EasyBuildError("ContainerStack: hard link %s: too many levels of symbolic links (loop or > %d) in "
                             "layer %s", path, LINK_FOLLOW_MAX, layer['digest'])
    raise hard_link_error(layer, path, link)


def hard_link_data(blobs_dir, layer, targets):
    """
    {path: data} for the hard links of a layer streamed through the 'zstd' command, which cannot seek back to the
    member a link resolves to: targets is {path: that file member (hard_link_member)}. The data is read in a second
    pass over the layer, keeping only those members.
    """
    wanted = {}
    for path, member in targets.items():
        wanted.setdefault(member.offset, []).append(path)
    data = {}
    with open_layer(blobs_dir, layer) as tar:
        for member in tar:
            if member.offset in wanted:
                content = tar.extractfile(member).read()
                data.update((path, content) for path in wanted[member.offset])
    return data


def layer_is_streamed(layer):
    """Whether open_layer() streams this layer through the 'zstd' command (so its TarFile cannot seek)."""
    return layer.get('mediaType') == OCI_LAYER_ZSTD and not tarfile_reads_zstd()


def image_entries(blobs_dir, layers, paths):
    """
    For each image path in 'paths': (TarInfo, bytes or None) of its final entry once all layers are applied
    in order (OCI whiteouts and opaque directories included), or None if the image does not have it.
    A hard link's bytes are those of the file it resolves to (hard_link_member, the same for every kind of layer):
    a seekable layer reads that member where it is (tarfile seeks back), a layer streamed through the 'zstd' command
    in a second pass.
    """
    found = dict((path, None) for path in paths)
    for layer in layers:
        streamed = layer_is_streamed(layer)
        # gone: (path, whether the path itself goes too) for what this layer hides of the lower layers
        # links: the wanted hard links; index: every member of the layer, by image path, to resolve them
        entries, gone, links, index, targets = {}, [], {}, {}, {}
        with open_layer(blobs_dir, layer) as tar:
            for member in tar:
                name = layer_path(member.name)
                index.setdefault(name, []).append(member)
                parent, base = os.path.split(name)
                if base == '.wh..wh..opq':
                    gone.append((parent, False))
                elif base.startswith('.wh.'):
                    gone.append((os.path.join(parent, base[4:]), True))
                else:
                    if not member.isdir():    # a non-directory replaces whatever the lower layers had beneath it
                        gone.append((name, False))
                    if name in found:
                        links.pop(name, None)
                        data = None
                        if member.isfile():
                            data = tar.extractfile(member).read()
                        elif member.islnk():
                            links[name] = member
                        entries[name] = (member, data)
            targets = dict((path, hard_link_member(layer, path, link, index)) for path, link in links.items())
            if not streamed:
                for path, target in targets.items():
                    entries[path] = (entries[path][0], tar.extractfile(target).read())
        if streamed and targets:
            for path, data in hard_link_data(blobs_dir, layer, targets).items():
                entries[path] = (entries[path][0], data)
        for path in found:
            if any(not g or path.startswith(g + '/') or (itself and path == g) for g, itself in gone):
                found[path] = None
        found.update(entries)
    return found


def tarfile_reads_zstd():
    """Whether this Python's tarfile can read zstd (Python >= 3.14 with its zstd module)."""
    if 'zst' not in tarfile.TarFile.OPEN_METH:
        return False
    try:
        import compression.zstd  # noqa: F401
    except ImportError:
        return False
    return True


@contextlib.contextmanager
def open_layer(blobs_dir, layer):
    """
    Open a verified layer blob as a TarFile. zstd layers are read by tarfile if it can, else streamed through
    the 'zstd' command.
    """
    path = blob_path(blobs_dir, layer['digest'])
    if layer.get('mediaType') != OCI_LAYER_ZSTD or tarfile_reads_zstd():
        with tarfile.open(path, 'r:zst' if layer.get('mediaType') == OCI_LAYER_ZSTD else 'r:*') as tar:
            yield tar
        return
    zstd = which('zstd', on_error=IGNORE)
    if not zstd:
        raise EasyBuildError("ContainerStack: layer %s is zstd-compressed, but this Python's tarfile cannot read "
                             "zstd and there is no 'zstd' command in $PATH", layer['digest'])
    proc = subprocess.Popen([zstd, '-q', '-d', '-c', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
            yield tar
        while proc.stdout.read(1 << 16):    # the rest (tar padding), so zstd never dies of SIGPIPE
            pass
    except BaseException:
        proc.kill()
        raise
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors='replace').strip()
        proc.stderr.close()
        proc.wait()
    if proc.returncode:
        raise EasyBuildError("ContainerStack: 'zstd -d' of layer %s failed (exit code %s): %s",
                             layer['digest'], proc.returncode, err)


def rootfs_path(rootfs, path):
    """
    Host path of image path 'path' in rootfs. Refuses a path whose parent directory resolves outside rootfs:
    through '..' or through an image symlink (an absolute one would resolve on the host).
    """
    root = os.path.realpath(rootfs)
    target = os.path.normpath(os.path.join(root, path.lstrip('/')))
    if target != root and os.path.commonpath([root, os.path.realpath(os.path.dirname(target))]) != root:
        raise EasyBuildError("ContainerStack: image path %r leads outside the image rootfs %s; refusing it",
                             path, rootfs)
    return target


def enroot_fixperms(rootfs):
    """
    The permission fix 'enroot import' applies to everything beneath the rootfs (enroot's common::fixperms):
    'chmod -R u+rX' (u+r on all, u+x on directories and on files anyone may execute), then u+w on the directories
    at most ENROOT_FIXPERMS_DIR_DEPTH levels below the rootfs. Symlinks are neither followed nor changed.
    The rootfs directory itself is left alone (enroot makes it anew).
    """
    def fix(path, depth):
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            return
        new = stat.S_IMODE(mode) | stat.S_IRUSR
        if stat.S_ISDIR(mode):
            new |= stat.S_IXUSR | (stat.S_IWUSR if depth <= ENROOT_FIXPERMS_DIR_DEPTH else 0)
        elif mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            new |= stat.S_IXUSR
        if new != stat.S_IMODE(mode):
            os.chmod(path, new)

    # top-down: a directory is fixed (made readable and searchable) before os.walk lists it
    for top, dirs, files in os.walk(rootfs):
        rel = os.path.relpath(top, rootfs)
        depth = 1 if rel == os.curdir else rel.count(os.sep) + 2
        for name in dirs + files:
            fix(os.path.join(top, name), depth)


def current_umask():
    """The process umask (os.umask can only read it by setting it: set it straight back)."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def remove_nofollow(path):
    """Remove a file, symlink or directory if present, never following a symlink."""
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        remove_dir(path)


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
# The command runs, not the image's entrypoint: apptainer exec and enroot (etc/enroot-no-entrypoint.conf) never
# run it; pyxis does not by default, but does where the site's plugstack sets execute_entrypoint=1.
# Under pyxis the wrapper starts ONE task, like the plain command (srun --nodes=1 --ntasks=1, even in a big batch job);
# CS_SRUN_ARGS comes after these and srun's later options win, e.g. CS_SRUN_ARGS="--nodes=2 --ntasks=2" (one per node).
# Inside a job step (SLURM_STEP_ID, or SLURM_STEPID, set; not the batch, extern or interactive step) auto never picks
# pyxis, and CS_RUNTIME=pyxis fails at once (exit 1) unless its srun overlaps the step: --overlap in CS_SRUN_ARGS, or
# SLURM_OVERLAP set (a GPU request may still wait on some Slurm versions).
# With a GPU, Apptainer gets --nv and, unless (APPTAINERENV_)TRITON_LIBCUDA_PATH is set, Triton's libcuda path.
# The caller's environment goes in under every runtime (enroot: environ() in etc/enroot-no-entrypoint.conf); image ENV
# beats it, etc/site.env beats both; enroot never takes the caller's PATH, LD_*, locale, HOME, USER, SHELL, TERM.
# No runtime gets the caller's PYTHONPATH or PYTHONHOME: they name a host Python's paths (e.g. a loaded Python module's
# site-packages), which the container's Python cannot use and which are not mounted; image ENV and site.env still set
# them. PYTHONUSERBASE and PYTHONNOUSERSITE go in (the user site is per Python version; the default is in $HOME).
# CS_PYTHONPATH (paths as the container sees them, e.g. CS_PYTHONPATH=$HOME/Megatron-LM) is the container's PYTHONPATH
# under every runtime: it replaces the image's ENV PYTHONPATH (append that yourself if needed); site.env beats it.
root="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)" || exit 1
sqsh="$root/image.sqsh"
sif="$root/image.sif"
cname=%(cname)s
mounts=(%(mounts)s)
cmd=(%(cmd)s)
names=()
unset PYTHONPATH PYTHONHOME
if [ -n "${CS_PYTHONPATH:-}" ]; then
    # forwarded by name, like site.env, so that it beats image ENV; before site.env, which beats it
    export PYTHONPATH="$CS_PYTHONPATH" APPTAINERENV_PYTHONPATH="$CS_PYTHONPATH"
    names+=(PYTHONPATH)
fi
if [ -r "$root/etc/site.env" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        # only NAME=VALUE lines; comments, blank lines and anything else are skipped
        name="${line%%%%=*}"
        case "$line" in *=*) ;; *) continue ;; esac
        case "$name" in '' | [0-9]* | *[!A-Za-z0-9_]*) continue ;; esac
        # Apptainer lets image ENV beat host env; APPTAINERENV_<NAME> beats image ENV
        export "$line" "APPTAINERENV_$line"
        case " ${names[*]} " in *" $name "*) ;; *) names+=("$name") ;; esac
    done < "$root/etc/site.env"
fi
runtime="${CS_RUNTIME:-%(policy)s}"
case "$runtime" in sif) runtime=apptainer ;; sqsh) runtime=enroot ;; esac
# pyxis attaches a container only when a step launches, so not inside a step: there a nested srun waits for the
# resources this step holds (in a whole-node or exclusive step, until the job ends). The batch, extern and
# interactive steps are not such steps: by name, or by Slurm's special ids (current and pre-20.11)
in_step=
step="${SLURM_STEP_ID-${SLURM_STEPID-}}"
case "$step" in
    '' | batch | extern | interactive | 4294967290 | 4294967291 | 4294967292 | 4294967294 | 4294967295) ;;
    *) [ -n "$SLURM_JOB_ID" ] && in_step=1 ;;
esac
if [ "$runtime" = auto ]; then
    # use pyxis in an allocation (batch script, salloc), but not inside a step
    if [ -n "$SLURM_JOB_ID" ] && [ -z "$in_step" ] && [ -e "$sqsh" ] && command -v srun >/dev/null 2>&1; then
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
        # a nested srun that overlaps this step (--overlap, or SLURM_OVERLAP, which srun reads) may share its resources
        overlap=
        [ -n "${SLURM_OVERLAP+set}" ] && overlap=1
        read -ra words <<< "$CS_SRUN_ARGS"
        for word in ${words[@]+"${words[@]}"}; do
            case "$word" in --overlap | --overlap=*) overlap=1 ;; esac
        done
        if [ -n "$in_step" ] && [ -z "$overlap" ]; then
            echo "ContainerStack: CS_RUNTIME=pyxis, but already inside job step $step of job $SLURM_JOB_ID;" \
                "a nested srun would wait for resources held by this step: use CS_RUNTIME=sif or sqsh here, call" \
                "the launcher from the batch script, or add --overlap to CS_SRUN_ARGS (or set SLURM_OVERLAP) to" \
                "share this step's resources (GPU requests may still wait on some Slurm versions)" >&2
            exit 1
        fi
        args=(--container-image="$sqsh")
        [ ${#mounts[@]} -gt 0 ] && args+=(--container-mounts="$(join "${mounts[@]}")")
        [ ${#names[@]} -gt 0 ] && args+=(--container-env="$(join "${names[@]}")")
        exec srun --nodes=1 --ntasks=1 $CS_SRUN_ARGS "${args[@]}" "${cmd[@]}" "$@" ;;
    apptainer)
        command -v apptainer >/dev/null 2>&1 || { echo "ContainerStack: apptainer not found on PATH" >&2; exit 127; }
        args=()
        if [ -n "$gpu" ]; then
            args+=(--nv)
            # --nv puts the driver libraries in /.singularity.d/libs, which the image's ldconfig cache does not
            # list, so Triton (torch.compile) cannot find libcuda.so; the caller's own setting wins
            if [ -z "${TRITON_LIBCUDA_PATH+set}${APPTAINERENV_TRITON_LIBCUDA_PATH+set}" ]; then
                export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
            fi
        fi
        [ ${#mounts[@]} -gt 0 ] && args+=(--bind "$(join "${mounts[@]}")")
        exec apptainer exec "${args[@]}" "$sif" "${cmd[@]}" "$@" ;;
    enroot)
        command -v enroot >/dev/null 2>&1 || { echo "ContainerStack: enroot not found on PATH" >&2; exit 127; }
        enroot list | grep -qxF "$cname" || enroot create -n "$cname" "$sqsh" >/dev/null || exit 1
        args=(--conf "$root/etc/enroot-no-entrypoint.conf")
        [ -n "$gpu" ] && args+=(--env NVIDIA_VISIBLE_DEVICES=all)
        for m in "${mounts[@]}"; do args+=(--mount "$m"); done
        for n in "${names[@]}"; do args+=(--env "$n"); done
        # the caller's environment, through the --conf file's environ() (image ENV and site.env win, as under pyxis)
        export CS_CALLER_ENV="$(compgen -e)"
        exec enroot start "${args[@]}" "$cname" "${cmd[@]}" "$@" ;;
    *)
        echo "ContainerStack: unknown CS_RUNTIME '$runtime' (use auto, apptainer, enroot, pyxis, sif or sqsh)" >&2
        exit 2 ;;
esac
'''


def escape_chars(text, special, control):
    """
    text with each character in special backslash-escaped, newlines as \\n and other control characters as
    control % their code.
    """
    def escape(char):
        if char in special:
            return '\\' + char
        if char == '\n':
            return '\\n'
        if ord(char) < 32 or ord(char) == 127:
            return control % ord(char)
        return char
    return ''.join(escape(char) for char in text)


def lua_string(text):
    """text as a double-quoted Lua string literal, as safe as string.format('%q'): \\ and " escaped, controls \\ddd."""
    return '"%s"' % escape_chars(text, '\\"', '\\%03d')


def tcl_string(text):
    """
    text as a Tcl word that is text verbatim, also inside a braced body: double-quoted, with each character Tcl
    substitutes or counts there (\\ " [ ] $ { }; not ';', literal in a quoted word) backslash-escaped, newlines as
    \\n, other control characters as \\ooo.
    """
    return '"%s"' % escape_chars(text, '\\"[]${}', '\\%03o')


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
    """
    Deploy a digest-pinned OCI image as a module, in enroot/pyxis (.sqsh) and Apptainer (.sif) formats.

    Like pyxis' default and `apptainer exec`, the wrappers run the given command, not the image's ENTRYPOINT. Under
    enroot this is done by etc/enroot-no-entrypoint.conf (installed with the .sqsh), an `enroot start --conf` hook
    that writes 'exec "$@"' to the container's /etc/rc.local.

    With a GPU, the Apptainer wrappers add --nv and export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
    (where --nv binds the driver libraries, which the image's ldconfig cache does not list), so that Triton, e.g. for
    torch.compile, finds libcuda.so; not when the caller has set TRITON_LIBCUDA_PATH or
    APPTAINERENV_TRITON_LIBCUDA_PATH. pyxis and enroot need nothing: their NVIDIA hook updates the cache.

    Every runtime passes the caller's environment into the container, with the same precedence: the image's ENV wins
    over the caller's value, etc/site.env (site_env) over both (pyxis: --container-env; Apptainer: APPTAINERENV_*;
    enroot: --env). Under enroot, the environ() function of etc/enroot-no-entrypoint.conf adds the caller's variables
    that the image's /etc/environment and enroot's environ.d do not set, as pyxis does. It never forwards PATH,
    LD_LIBRARY_PATH, LD_PRELOAD (neither does pyxis), LANG, LANGUAGE, LC_ALL (pyxis uses the container's) or HOME,
    SHELL, USER, LOGNAME, TERM (enroot sets them for the container's user), nor a value containing a newline (enroot's
    environment file has one variable per line).

    No runtime gets the caller's PYTHONPATH or PYTHONHOME (the wrappers unset them before anything else): with a host
    Python module loaded they name that interpreter's site-packages, built for another Python version and not mounted
    in the container, so the container's Python would silently miss them or mix them with its own. The image's ENV
    and site_env still set them. To put paths of one's own on the container's PYTHONPATH (e.g. a Megatron-LM
    checkout in $HOME), set CS_PYTHONPATH: the wrappers pass it in as PYTHONPATH by name, as site_env is passed
    (pyxis --container-env, APPTAINERENV_PYTHONPATH, enroot --env), so under every runtime it replaces the image's
    ENV PYTHONPATH (the host cannot read the image's value to add to it: include it in CS_PYTHONPATH if needed) and
    site_env's PYTHONPATH beats it. PYTHONUSERBASE and PYTHONNOUSERSITE are passed on as before: the user site is
    per Python version (lib/pythonX.Y), it is the user's own choice, and dropping PYTHONUSERBASE would only send the
    container's Python to ~/.local (mounted with $HOME by default); PYTHONNOUSERSITE=1 is the switch that hides it.
    A module with launchers named python* prints a one-line warning when it is loaded while a host Python module is
    (EBROOTPYTHON is set): its python now shadows the host's, whose packages the container cannot see.

    An image's User (config.User, e.g. '1000:1000' or 'appuser') is ignored, as 'enroot import' ignores it (enroot
    4.2.0's docker::configure, src/docker.sh, never reads it) and as `apptainer exec` does: under every runtime the
    container runs as the calling user (or as root with enroot --root, pyxis --container-remap-root or apptainer
    --fakeroot). The build warns about a non-root User (see image_user), and manifest.json records it as image_user.

    The auto launch policy uses pyxis only in an allocation outside a job step (a batch script, salloc): inside a step
    (SLURM_STEP_ID set, or the older SLURM_STEPID; the batch, extern and interactive steps, named so or by Slurm's
    special step ids 4294967290-4294967292 and the pre-20.11 4294967294/4294967295, do not count) the srun pyxis
    needs would wait for the resources that step holds, in a whole-node step until the job ends. There auto uses
    Apptainer or enroot, and CS_RUNTIME=pyxis exits 1 at once, saying so, unless the nested srun shares the step's
    resources: --overlap (or --overlap=...) in CS_SRUN_ARGS, or SLURM_OVERLAP set, which srun reads as --overlap. Even
    then an srun that requests GPUs may wait on some Slurm versions; auto never uses pyxis inside a step.

    Promotion (`eb --module-only` with hidden = False) must not use --force: EasyBuild then skips the sanity check,
    and with it the check that the install's manifest.json matches the pinned digest.
    """

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
        self.image_config = None    # the image config (a dict), read once build_step has verified the blobs
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
        self.image_config = read_image_config(self.blobs_dir, self.manifest)
        user, non_root = image_user(self.image_config)
        if non_root:
            print_warning("ContainerStack: image sets User %r, which enroot and Apptainer ignore: the container runs "
                          "as the calling user (or as root with enroot --root, pyxis --container-remap-root or "
                          "apptainer --fakeroot), not as %s" % (user, user), log=self.log)
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
            remove_nofollow(os.path.join(rootfs, rel))
        # as 'enroot import' has them: the image's own bind-target files, its permission fix, a plain mkdir's root
        for rel, entry in image_entries(self.blobs_dir, self.manifest['layers'], APPTAINER_BIND_TARGETS).items():
            path = rootfs_path(rootfs, rel)
            remove_nofollow(path)
            if entry and entry[0].issym():
                os.symlink(entry[0].linkname, path)
            elif entry and entry[1] is not None:
                with open(path, 'wb') as fh:
                    fh.write(entry[1])
                os.chmod(path, entry[0].mode & 0o7777)
        enroot_fixperms(rootfs)
        os.chmod(rootfs, 0o777 & ~current_umask())
        config = self.image_config
        provenance = 'containerstack://%s@%s' % (self.cfg['image'], self.digest)
        for rel, text in enroot_files(config, provenance).items():
            path = rootfs_path(rootfs, rel)
            remove_nofollow(path)    # an image symlink here must be replaced, not written through
            write_file(path, text)
        workdir = (config.get('config') or {}).get('WorkingDir') or '/'
        mkdir(rootfs_path(rootfs, workdir), parents=True)
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
        if 'sqsh' in self.cfg['formats']:
            write_file(os.path.join(self.installdir, ENROOT_NO_ENTRYPOINT), ENROOT_NO_ENTRYPOINT_CONF)
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

    def run_smoke_tests(self, log_dir=None):
        """
        Run every smoke test in every format; returns the results, raises on the first failure.
        With log_dir, each test's output goes to <log_dir>/<n>-<kind>.<format>.log (n: position in smoke_tests).
        """
        tests = [tuple(test) for test in self.cfg['smoke_tests']]
        gpu_tests = [cmd for kind, cmd in tests if kind == 'gpu']
        require_gpu = self.cfg['require_gpu_tests']
        if require_gpu and not gpu_tests:
            raise EasyBuildError("ContainerStack: require_gpu_tests is set but no smoke test is tagged gpu")
        if require_gpu is None:
            require_gpu = bool(gpu_tests)
        gpu = os.path.exists(os.environ.get('CS_GPU_DEVICE', '/dev/nvidia0'))
        if require_gpu and not gpu:
            raise EasyBuildError("ContainerStack: GPU smoke test %r cannot run: no GPU on %s; build on a GPU node "
                                 "or set require_gpu_tests = False", gpu_tests[0], socket.gethostname())
        results = []
        for fmt in self.cfg['formats']:
            try:
                for idx, (kind, command) in enumerate(tests, start=1):
                    result = {'format': fmt, 'kind': kind, 'command': command,
                              'date': time.strftime('%Y-%m-%dT%H:%M:%S')}
                    if kind == 'gpu' and not gpu:
                        result['result'] = 'not-run: no GPU'
                    else:
                        start = time.time()
                        res = run_shell_cmd(self._smoke_cmd(fmt, command), fail_on_error=False)
                        if log_dir:
                            write_file(os.path.join(log_dir, '%d-%s.%s.log' % (idx, kind, fmt)), res.output or '')
                        if res.exit_code != 0:
                            raise EasyBuildError("ContainerStack: %s smoke test failed in %s (exit %s): %s\n%s",
                                                 kind, fmt, res.exit_code, command, (res.output or '')[-2000:])
                        result.update(result='pass', seconds=round(time.time() - start, 1))
                    results.append(result)
            finally:
                if fmt == 'sqsh':    # remove the unpacked enroot container, even after a failed test
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
            'image_user': image_user(self.image_config)[0],
            'layers': [layer['digest'] for layer in self.manifest['layers']],
            'files': files, 'runtimes': self.runtime_versions(), 'smoke_tests': results,
            'easybuild': str(EASYBUILD_VERSION), 'easyblocks': str(EASYBLOCKS_VERSION),
        }
        write_file(path, json.dumps(data, indent=2, sort_keys=True) + '\n')

    def check_existing_manifest(self, path):
        """For --module-only / --sanity-check-only / --skip: the install must be complete and match the digest."""
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
        files += [ENROOT_NO_ENTRYPOINT] if 'sqsh' in self.cfg['formats'] else []
        super().sanity_check_step(custom_paths={'files': files, 'dirs': []})
        manifest_path = os.path.join(self.installdir, MANIFEST)
        # --module-only, --sanity-check-only and --skip (eb.skip: already installed) never rewrite the manifest
        if build_option('module_only') or self.cfg['module_only'] or build_option('sanity_check_only') or self.skip:
            self.check_existing_manifest(manifest_path)
            if build_option('sanity_check_only'):
                self.run_smoke_tests()    # no smoke/ logs: the install is read-only by now
            return
        results = self.run_smoke_tests(log_dir=os.path.join(self.installdir, SMOKE_DIR))
        self.write_manifest(manifest_path, results)

    def permissions_step(self):
        """
        Standard permissions, then make the installed files read-only (directories stay writable).
        EasyBuild's own log dir (installdir/easybuild) stays writable: it archives the recipe there on promotion.
        """
        super().permissions_step()
        logdir = os.path.normpath(os.path.join(self.installdir, log_path(ec=self.cfg)))
        for root, dirs, files in os.walk(self.installdir):
            dirs[:] = [name for name in dirs if os.path.normpath(os.path.join(root, name)) != logdir]
            for name in files:
                adjust_permissions(os.path.join(root, name), stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH, add=False,
                                   recursive=False)

    def make_module_step(self, fake=False):
        """Generate the module file; warn when --module-only --force skipped the manifest/digest check."""
        module_only = build_option('module_only') or self.cfg['module_only']
        if not fake and module_only and build_option('force'):
            print_warning("ContainerStack: --force skips the manifest/digest check; do not promote with --force",
                          log=self.log)
        return super().make_module_step(fake=fake)

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
        txt += self._host_python_warning()
        return txt

    def _host_python_warning(self):
        """
        Module code that, on load only, warns when a host Python module is loaded (EBROOTPYTHON is set) and python*
        launchers of this module now come first on $PATH: their Python is the container's, which cannot see the host
        Python's packages (and the wrappers do not pass the host's PYTHONPATH in). '' without python* launchers.
        """
        names = sorted(name for name in self.cfg['launchers'] if fnmatch.fnmatchcase(name, 'python*'))
        if not names:
            return ''
        msg = ("WARNING: %s: the container's %s now shadow%s the host Python (EBROOTPYTHON is set); host Python "
               "packages are not visible inside the container; set CS_PYTHONPATH to pass paths into the container"
               % (self.short_mod_name, ', '.join(names), '' if len(names) > 1 else 's'))
        if self.module_generator.SYNTAX == 'Lua':
            return ('\nif mode() == "load" and (os.getenv("EBROOTPYTHON") or "") ~= "" then\n'
                    '    io.stderr:write(%s)\nend\n' % lua_string(msg + '\n'))
        return ('\nif { [ module-info mode load ] && [ info exists ::env(EBROOTPYTHON) ] && '
                '$::env(EBROOTPYTHON) ne "" } {\n    puts stderr %s\n}\n' % tcl_string(msg))
