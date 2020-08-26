#importing the required libraries
import argparse #to parse command line arguments
import collections #to have additional container types like ordered dict
import difflib #provides classes and functions for comparing sequences like strings, lists, blobs
import enum
import hashlib #to use SHA-1 functionality
import operator #exports a set of efficient functions corresponding to the intrinsic operators
import os
import stat #provides a portable way of using operating system dependent functionality
import struct #to convert native data types such as strings and numbers into a string of bytes and vice versa
import sys #to access the actual command-line arguments
import time
import zlib #Git compresses everything using this

sys.tracebacklimit = 0 #to limit the detail of the exception being raised!

# Data for one entry in the git index (.git/index)
IndexEntry = collections.namedtuple('IndexEntry', [
    'ctime_s', 'ctime_n', 'mtime_s', 'mtime_n', 'dev', 'ino', 'mode', 'uid',
    'gid', 'size', 'sha1', 'flags', 'path',
])


def read_file(path):
    """Read contents of file at given path as bytes."""
    with open(path, 'rb') as f:
        return f.read()


def write_file(path, data):
    """Write data bytes to file at given path."""
    with open(path, 'wb') as f:
        f.write(data)


def init(repo):
    """Create directory for repo and initialize .git directory.
       and perform the required checks."""

    if os.path.exists(repo):
        if (os.path.isdir(os.path.join(repo, '.git'))):
            raise Exception("%s is already a Git repository!" % repo)
        if os.listdir(repo):
            raise Exception("%s is not empty!" % repo)
    else:
        os.mkdir(repo)

    os.mkdir(os.path.join(repo, '.git'))
    
    for name in ['objects', 'refs', 'refs/heads']:
        os.mkdir(os.path.join(repo, '.git', name))

    write_file(os.path.join(repo, '.git', 'HEAD'), b'ref: refs/heads/master')
    open(os.path.join(repo, '.git', 'PREV'), 'w').close()

    print('initialized empty repository: {}'.format(repo))


def repo_find(path=".", required=True):
    """Looks for a repository starting at the current directory and recurses
     back until the root directory. To identify something as a repository it
     will check for the presence of a .git directory"""

    path = os.path.realpath(path)
    if os.path.isdir(os.path.join(path, ".git")):
        return path

    parent = os.path.realpath(os.path.join(path, ".."))
    if parent == path:
        if required:
            raise Exception("No git directory.")
        else:
            return None

    return repo_find(parent, required)


def hash_object(data, obj_type, write=True):
    """Compute hash of object data of given type and write to object store if
    "write" is True. Return SHA-1 object hash as hex string."""

    header = '{} {}'.format(obj_type, len(data)).encode()
    full_data = header + b'\x00' + data
    sha1 = hashlib.sha1(full_data).hexdigest()
    if write:
        path = os.path.join('.git', 'objects', sha1[:2], sha1[2:])
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_file(path, zlib.compress(full_data))
    return sha1


def find_object(sha1_prefix):
    """Find object with given SHA-1 prefix and return path to object in object
    store, or raise ValueError if there are no objects or multiple objects
    with this prefix."""

    if len(sha1_prefix) < 2:
        raise ValueError('hash prefix must be 2 or more characters')
    obj_dir = os.path.join('.git', 'objects', sha1_prefix[:2])
    rest = sha1_prefix[2:]
    objects = [name for name in os.listdir(obj_dir) if name.startswith(rest)]
    if not objects:
        raise ValueError('object {!r} not found'.format(sha1_prefix))
    if len(objects) >= 2:
        raise ValueError('multiple objects ({}) with prefix {!r}'.format(
                len(objects), sha1_prefix))
    return os.path.join(obj_dir, objects[0])


def read_object(sha1_prefix):
    """Read object with given SHA-1 prefix and return tuple of
    (object_type, data_bytes), or raise ValueError if not found."""

    path = find_object(sha1_prefix)
    full_data = zlib.decompress(read_file(path))
    nul_index = full_data.index(b'\x00')
    header = full_data[:nul_index]
    obj_type, size_str = header.decode().split()
    size = int(size_str)
    data = full_data[nul_index + 1:]
    assert size == len(data), 'expected size {}, got {} bytes'.format(
            size, len(data))
    return (obj_type, data)


def cat_file(mode, sha1_prefix):
    """Write the contents of object with given SHA-1 prefix to
    stdout. If mode is 'commit', 'tree', or 'blob', print raw data bytes of
    object. If mode is 'size', print the size of the object. If mode is
    'type', print the type of the object. If mode is 'pretty', print a
    prettified version of the object."""

    obj_type, data = read_object(sha1_prefix)
    if mode in ['commit', 'tree', 'blob']:
        if obj_type != mode:
            raise ValueError('expected object type {}, got {}'.format(
                    mode, obj_type))
        sys.stdout.buffer.write(data)
    elif mode == 'size':
        print(len(data))
    elif mode == 'type':
        print(obj_type)
    elif mode == 'pretty':
        if obj_type in ['commit', 'blob']:
            sys.stdout.buffer.write(data)
        elif obj_type == 'tree':
            for mode, path, sha1 in read_tree(data=data):
                type_str = 'tree' if stat.S_ISDIR(mode) else 'blob'
                print('{:06o} {} {}\t{}'.format(mode, type_str, sha1, path))
        else:
            assert False, 'unhandled object type {!r}'.format(obj_type)
    else:
        raise ValueError('unexpected mode {!r}'.format(mode))


def read_index():
    """Read git index file and return list of IndexEntry objects."""
    try:
        data = read_file(os.path.join('.git', 'index'))
    except FileNotFoundError:
        return []
    digest = hashlib.sha1(data[:-20]).digest()
    assert digest == data[-20:], 'invalid index checksum'
    signature, version, num_entries = struct.unpack('!4sLL', data[:12])
    assert signature == b'DIRC', \
            'invalid index signature {}'.format(signature)
    assert version == 2, 'unknown index version {}'.format(version)
    entry_data = data[12:-20]
    entries = []
    i = 0
    while i + 62 < len(entry_data):
        fields_end = i + 62
        fields = struct.unpack('!LLLLLLLLLL20sH', entry_data[i:fields_end])
        path_end = entry_data.index(b'\x00', fields_end)
        path = entry_data[fields_end:path_end]
        entry = IndexEntry(*(fields + (path.decode(),)))
        entries.append(entry)
        entry_len = ((62 + len(path) + 8) // 8) * 8
        i += entry_len
    assert len(entries) == num_entries
    return entries


def ls_files(details=False):
    """Print list of files in index (including mode, SHA-1, and stage number
    if "details" is True)."""

    for entry in read_index():
        if details:
            stage = (entry.flags >> 12) & 3
            print('{:6o} {} {:}\t{}'.format(
                    entry.mode, entry.sha1.hex(), stage, entry.path))
        else:
            print(entry.path)


def get_status():
    """Get status of working copy, return tuple of (changed_paths, new_paths,
    deleted_paths)."""

    paths = set()
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d != '.git']
        for file in files:
            path = os.path.join(root, file)
            path = path.replace('\\', '/')
            if path.startswith('./'):
                path = path[2:]
            paths.add(path)
    entries_by_path = {e.path: e for e in read_index()}
    entry_paths = set(entries_by_path)
    changed = {p for p in (paths & entry_paths)
               if hash_object(read_file(p), 'blob', write=False) !=
                  entries_by_path[p].sha1.hex()}
    new = paths - entry_paths
    deleted = entry_paths - paths
    return (sorted(changed), sorted(new), sorted(deleted))


def status():
    """Show status of working copy."""
    changed, new, deleted = get_status()
    if changed:
        print('changed files:')
        for path in changed:
            print('   ', path)
    if new:
        print('new files:')
        for path in new:
            print('   ', path)
    if deleted:
        print('deleted files:')
        for path in deleted:
            print('   ', path)


def diff():
    """Show diff of files changed (between index and working copy)."""
    changed, _, _ = get_status()
    entries_by_path = {e.path: e for e in read_index()}
    for i, path in enumerate(changed):
        sha1 = entries_by_path[path].sha1.hex()
        obj_type, data = read_object(sha1)
        assert obj_type == 'blob'
        index_lines = data.decode().splitlines()
        working_lines = read_file(path).decode().splitlines()
        diff_lines = difflib.unified_diff(
                index_lines, working_lines,
                '{} (index)'.format(path),
                '{} (working copy)'.format(path),
                lineterm='')
        for line in diff_lines:
            print(line)
        if i < len(changed) - 1:
            print('-' * 70)


def write_index(entries):
    """Write list of IndexEntry objects to git index file."""
    packed_entries = []
    for entry in entries:
        entry_head = struct.pack('!LLLLLLLLLL20sH',
                entry.ctime_s, entry.ctime_n, entry.mtime_s, entry.mtime_n,
                entry.dev, entry.ino, entry.mode, entry.uid, entry.gid,
                entry.size, entry.sha1, entry.flags)
        path = entry.path.encode()
        length = ((62 + len(path) + 8) // 8) * 8
        packed_entry = entry_head + path + b'\x00' * (length - 62 - len(path))
        packed_entries.append(packed_entry)
    header = struct.pack('!4sLL', b'DIRC', 2, len(entries))
    all_data = header + b''.join(packed_entries)
    digest = hashlib.sha1(all_data).digest()
    write_file(os.path.join('.git', 'index'), all_data + digest)


def add(paths):
    """Add all file paths to git index."""
    paths = [p.replace('\\', '/') for p in paths]
    all_entries = read_index()

    prev_content = read_previous()
    prev_entries = [e for e in all_entries if e.path in paths]
    for entry in prev_entries:
        prev_content[entry.path] = entry.sha1.hex()
    write_previous(prev_content)

    entries = [e for e in all_entries if e.path not in paths]
    for path in paths:
        sha1 = hash_object(read_file(path), 'blob')
        st = os.stat(path)
        flags = len(path.encode())
        assert flags < (1 << 12)
        entry = IndexEntry(
                int(st.st_ctime), 0, int(st.st_mtime), 0, st.st_dev,
                st.st_ino, st.st_mode, st.st_uid, st.st_gid, st.st_size,
                bytes.fromhex(sha1), flags, path)
        entries.append(entry)
    write_index(entries)


def write_tree():
    """Write a tree object from the current index entries."""
    tree_entries = []
    for entry in read_index():
        assert '/' not in entry.path, \
                'currently only supports a single, top-level directory'
        mode_path = '{:o} {}'.format(entry.mode, entry.path).encode()
        tree_entry = mode_path + b'\x00' + entry.sha1
        tree_entries.append(tree_entry)
    return hash_object(b''.join(tree_entries), 'tree')


def get_local_master_hash():
    """Get current commit hash (SHA-1 string) of local master branch."""
    master_path = os.path.join('.git', 'refs', 'heads', 'master')
    try:
        return read_file(master_path).decode().strip()
    except FileNotFoundError:
        return None


def commit(message, author=None):
    """Commit the current state of the index to master with given message.
    Return hash of commit object."""

    tree = write_tree()
    parent = get_local_master_hash()
    if author is None:
        author = '{} <{}>'.format(
               os.environ['USER'], os.environ['USERNAME'])
    obj = time.localtime(time.time())
    timestamp = str(obj[0])+"/"+str(obj[1])+"/"+str(obj[2])+" "+str(obj[3])+":"+str(obj[4])+":"+str(obj[5])
    utc_offset = -time.timezone
    author_time = '{} {}{:02}{:02}'.format(
            timestamp,
            '+' if utc_offset > 0 else '-',
            abs(utc_offset) // 3600,
            (abs(utc_offset) // 60) % 60)
    lines = ['tree ' + tree]
    if parent:
        lines.append('parent ' + parent)
    lines.append('author {} {}'.format(author, author_time))
    lines.append('committer {} {}'.format(author, author_time))
    lines.append('')
    lines.append(message)
    lines.append('')
    data = '\n'.join(lines).encode()
    sha1 = hash_object(data, 'commit')
    master_path = os.path.join('.git', 'refs', 'heads', 'master')
    write_file(master_path, (sha1 + '\n').encode())
    print('committed to master: {:7}'.format(sha1))
    return sha1


def read_tree(sha1=None, data=None):
    """Read tree object with given SHA-1 (hex string) or data, and return list
    of (mode, path, sha1) tuples.
    """
    if sha1 is not None:
        obj_type, data = read_object(sha1)
        assert obj_type == 'tree'
    elif data is None:
        raise TypeError('must specify "sha1" or "data"')
    i = 0
    entries = []
    for _ in range(1000):
        end = data.find(b'\x00', i)
        if end == -1:
            break
        mode_str, path = data[i:end].decode().split()
        mode = int(mode_str, 8)
        digest = data[end + 1:end + 21]
        entries.append((mode, path, digest.hex()))
        i = end + 1 + 20
    return entries


def kvlm_parse(raw, start=0, dct=None):
    """"Key Value List Message parser function: It is basically used to seperate
    out the various Key value pairs such as tree, parent, author and committer.
    It returns a ordered dictionary containing these pairs"""

    if not dct:
        dct = collections.OrderedDict()

    spc = raw.find(b' ', start)
    nl = raw.find(b'\n', start)

    if (spc < 0) or (nl < spc):
        assert(nl == start)
        dct[b''] = raw[start+1:]
        return dct

    key = raw[start:spc]
    end = start
    while True:
        end = raw.find(b'\n', end+1)
        if raw[end+1] != ord(' '): break

    value = raw[spc+1:end].replace(b'\n ', b'\n')

    if key in dct:
        if type(dct[key]) == list:
            dct[key].append(value)
        else:
            dct[key] = [ dct[key], value ]
    else:
        dct[key]=value

    return kvlm_parse(raw, start=end+1, dct=dct)


def resolve(repo, ref):
    """It returns the sha of the object to which the head is currently pointing."""
    with open(os.path.join(repo, ".git", ref), 'r') as fp:
        data = fp.read()
    if data.startswith("ref: "):
        return resolve(repo, data[5:])
    else:
        return data

def log(args):
    """It returns all the commits (starting from the lastest commit) that have been made in the repository."""
    repo = repo_find()
    if args.commit == "HEAD":
        sha = resolve(repo, "HEAD")[:-1]
    log_helper(repo, sha, set())


def log_helper(repo, sha, seen):
    """It is a helper function which recursively goes to the parent of the current commit and displays the information regarding that commit."""
    if sha in seen:
        return
    seen.add(sha)

    fmt, commit = read_object(sha)
    dct = kvlm_parse(commit)

    print("commit " + str(sha))
    print("author: " , dct[b'author'].decode("ascii"))
    print("committer: ", dct[b'committer'].decode("ascii"))
    print()
    print("	", dct[b''].decode("ascii"))

    if not b'parent' in dct.keys():
        return

    parents = dct[b'parent']

    if type(parents) != list:
        parents = [ parents ]

    for p in parents:
        p = p.decode("ascii")
        log_helper(repo, p, seen)


def read_previous():
    """It reads the contents of the .git/PREV file, separates key values pairs
    and returns the dictionary of these key value pairs."""

    prev_content = dict()
    repo = repo_find()
    with open(os.path.join(repo, ".git", "PREV"), 'r') as fp:
        data = fp.read()
    start = 0
    spc = data.find(' ', start)
    nl = data.find('\n', start)

    while start < spc:
        key = data[start:spc]
        value = data[spc+1:nl]
        prev_content[key] = value
        start = nl+1
        spc = data.find(' ', start)
        nl = data.find('\n', start)

    return prev_content


def write_previous(prev_content):
    """It concatenates the key value pairs of the dictionary and writes it to the .git/PREV file."""
    repo = repo_find()
    for k,v in prev_content.items():
        with open(os.path.join(repo, ".git", "PREV"), 'w') as fp:
            fp.write(k+" "+v+"\n")


def reset(commit):
    """It helps to revert back to the last commit and bring backs the contents of the previous commit."""
    repo = repo_find()

    with open(os.path.join(repo, ".git", "refs", "heads", "master"), 'w') as fp:
        fp.write(commit)

    entries = read_index()
    last_entry = entries[-1]
    entries = entries[0:-1]
    write_index(entries)

    path = last_entry.path
    prev_content = read_previous()
    if path in prev_content.keys():
        sha = prev_content[path]
        obj_type, data = read_object(sha)
        with open(path , 'wb') as fp:
            fp.write(data)
    else:
        open(path, 'w').close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='command', metavar='command')
    sub_parsers.required = True

    sub_parser = sub_parsers.add_parser('add',
            help='add file(s) to index')
    sub_parser.add_argument('paths', nargs='+', metavar='path',
            help='path(s) of files to add')

    sub_parser = sub_parsers.add_parser('cat-file',
            help='display contents of object')
    valid_modes = ['commit', 'tree', 'blob', 'size', 'type', 'pretty']
    sub_parser.add_argument('mode', choices=valid_modes,
            help='object type (commit, tree, blob) or display mode (size, '
                 'type, pretty)')
    sub_parser.add_argument('hash_prefix',
            help='SHA-1 hash (or hash prefix) of object to display')

    sub_parser = sub_parsers.add_parser('commit',
            help='commit current state of index to master branch')
    sub_parser.add_argument('-a', '--author')
    sub_parser.add_argument('-m', '--message', required=True,
            help='text of commit message')

    sub_parser = sub_parsers.add_parser('diff',
            help='show diff of files changed (between index and working '
                 'copy)')

    sub_parser = sub_parsers.add_parser('hash-object',
            help='hash contents of given path (and optionally write to '
                 'object store)')
    sub_parser.add_argument('path',
            help='path of file to hash')
    sub_parser.add_argument('-t', choices=['commit', 'tree', 'blob'],
            default='blob', dest='type',
            help='type of object (default %(default)r)')
    sub_parser.add_argument('-w', action='store_true', dest='write',
            help='write object to object store (as well as printing hash)')

    sub_parser = sub_parsers.add_parser('init',
            help='initialize a new repo')
    sub_parser.add_argument('repo',
            help='directory name for new repo')

    sub_parser = sub_parsers.add_parser('ls-files',
            help='list files in index')
    sub_parser.add_argument('-s', '--stage', action='store_true',
            help='show object details (mode, hash, and stage number) in '
                 'addition to path')

    sub_parser = sub_parsers.add_parser('log',
            help='Display history of a given commit.')
    sub_parser.add_argument('commit',
                   default='HEAD', nargs="?", help="Commit to start at.")

    sub_parser = sub_parsers.add_parser('reset',
            help='Revert back to a previous commit.')
    sub_parser.add_argument('commit',
                   nargs="?", help="Commit to revert back to")

    sub_parser = sub_parsers.add_parser('status',
            help='show status of working copy')

    args = parser.parse_args()
    if args.command == 'add':
        add(args.paths)
    elif args.command == 'cat-file':
        try:
            cat_file(args.mode, args.hash_prefix)
        except ValueError as error:
            print(error, file=sys.stderr)
            sys.exit(1)
    elif args.command == 'commit':
        commit(args.message, author=args.author)
    elif args.command == 'diff':
        diff()
    elif args.command == 'hash-object':
        sha1 = hash_object(read_file(args.path), args.type, write=args.write)
        print(sha1)
    elif args.command == 'init':
        init(args.repo)
    elif args.command == 'ls-files':
        ls_files(details=args.stage)
    elif args.command == 'status':
        status()
    elif args.command == 'log':
        log(args)
    elif args.command == 'reset':
        reset(args.commit)
    else:
        assert False, 'unexpected command {!r}'.format(args.command)
