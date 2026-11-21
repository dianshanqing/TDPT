"""Transparent, byte-preserving result storage with bounded-memory gzip IO.

Logical .json/.log names remain usable after compression. Original-content
hashes always cover decompressed bytes; physical storage hashes are separate.
Only closed, immutable files may be finalized or migrated. Historical input is
never parsed or reformatted by compression, verification or removal helpers.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

AUTO_COMPRESS_MIN_BYTES = 4 * 1024 * 1024
BLOCK_BYTES = 256 * 1024
STRING_CHARS = 64 * 1024


def logical_path(path):
    path = Path(path)
    return path.with_suffix('') if path.suffix.lower() == '.gz' else path


def _spellings(path):
    logical = logical_path(path)
    return logical, Path(str(logical) + '.gz')


def assert_no_collision(path):
    """Protect both spellings, including dangling symlinks."""
    for candidate in _spellings(path):
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(f'Refusing to overwrite existing output: {candidate}')
    return Path(path)


def resolve_result_path(path):
    """Resolve an old logical name; an explicitly present gzip never falls back."""
    path = Path(path)
    if path.is_file():
        return path
    if path.suffix.lower() != '.gz':
        compressed = Path(str(path) + '.gz')
        if compressed.is_file():
            return compressed
    raise FileNotFoundError(f'Result not found: {path} (or its .gz storage)')


def result_exists(path):
    try:
        resolve_result_path(path)
    except FileNotFoundError:
        return False
    return True


def open_result(path, mode='rt', encoding='utf-8', **kwargs):
    """Open original content for reading; writes use protected APIs below."""
    if mode not in ('r', 'rt', 'rb'):
        raise ValueError('open_result supports read-only modes')
    physical = resolve_result_path(path)
    options = kwargs if 'b' in mode else dict(encoding=encoding, **kwargs)
    if physical.suffix.lower() == '.gz':
        return gzip.open(physical, 'rb' if 'b' in mode else 'rt', **options)
    return physical.open(mode, **options)


def read_json(path, *, encoding='utf-8'):
    with open_result(path, encoding=encoding) as handle:
        return json.load(handle)


def _digest_stream(handle):
    state = hashlib.sha256()
    length = 0
    for chunk in iter(lambda: handle.read(BLOCK_BYTES), b''):
        state.update(chunk)
        length += len(chunk)
    return {'sha256': state.hexdigest(), 'bytes': length}


def content_digest(path):
    with open_result(path, 'rb') as handle:
        return _digest_stream(handle)


def storage_record(path):
    physical = resolve_result_path(path)
    original = content_digest(physical)
    with physical.open('rb') as handle:
        stored = _digest_stream(handle)
    return {'logical_path': str(logical_path(physical)), 'path': str(physical),
            'compression': 'gzip' if physical.suffix.lower() == '.gz' else 'none',
            'original_bytes': original['bytes'], 'original_sha256': original['sha256'],
            'stored_bytes': stored['bytes'], 'stored_sha256': stored['sha256']}


def result_files(directory, pattern):
    """Find a logical glob once even while a verified migration retains both."""
    directory = Path(directory)
    logical = {logical_path(path) for suffix in ('', '.gz')
               for path in directory.glob(pattern + suffix) if path.is_file()}
    return [resolve_result_path(path) for path in sorted(logical)]


def _checked_path(path):
    from runtime_identity import validate_output_path
    return validate_output_path(path)


@contextmanager
def _storage_lock(path):
    logical, compressed = _spellings(_checked_path(path))
    _checked_path(logical)
    _checked_path(compressed)
    logical.parent.mkdir(parents=True, exist_ok=True)
    lock = _checked_path(logical.with_name('.' + logical.name + '.storage.lock'))
    # The same lock protects both names. A stale lock fails safely for review.
    handle = lock.open('xb')
    try:
        with handle:
            yield logical, compressed
    finally:
        lock.unlink()


def _temporary(path):
    fd, name = tempfile.mkstemp(prefix='.' + Path(path).name + '.', suffix='.tmp', dir=Path(path).parent)
    os.close(fd)
    return Path(name)


def _publish(temporary, destination):
    """Publish complete bytes atomically without replacing an existing file."""
    destination = _checked_path(destination)
    # Windows rename refuses an existing destination. POSIX rename replaces,
    # so use an exclusive hard link there. Never use os.replace for evidence.
    if os.name == 'nt':
        os.rename(temporary, destination)
    else:
        os.link(temporary, destination)
        temporary.unlink()


def _stat_identity(path):
    value = Path(path).stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _gzip_copy(source, destination):
    with Path(source).open('rb') as reader, Path(destination).open('wb') as writer:
        with gzip.GzipFile(filename='', mode='wb', fileobj=writer, compresslevel=6, mtime=0) as archive:
            for chunk in iter(lambda: reader.read(BLOCK_BYTES), b''):
                archive.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())


def verify_lossless_copy(original, compressed=None):
    """Read-only exact byte comparison, including full gzip CRC/trailer checks."""
    original = Path(original)
    compressed = Path(compressed) if compressed is not None else Path(str(original) + '.gz')
    if original == compressed:
        raise ValueError('Original and compressed paths must differ')
    before = _stat_identity(original), _stat_identity(compressed)
    state = hashlib.sha256()
    length = 0
    with original.open('rb') as source, gzip.open(compressed, 'rb') as restored:
        while True:
            expected, actual = source.read(BLOCK_BYTES), restored.read(BLOCK_BYTES)
            if expected != actual:
                raise ValueError(f'Lossless verification failed: {original}')
            if not expected:
                break
            state.update(expected)
            length += len(expected)
    with compressed.open('rb') as handle:
        stored = _digest_stream(handle)
    if before != (_stat_identity(original), _stat_identity(compressed)):
        raise ValueError('File changed during lossless verification; original retained')
    return {'logical_path': str(original), 'path': str(compressed), 'compression': 'gzip',
            'original_bytes': length, 'original_sha256': state.hexdigest(),
            'stored_bytes': stored['bytes'], 'stored_sha256': stored['sha256'],
            'verified': True, 'removed_original': False}


def _remove_verified(original, compressed):
    before = _stat_identity(original), _stat_identity(compressed)
    record = verify_lossless_copy(original, compressed)
    if before != (_stat_identity(original), _stat_identity(compressed)):
        raise ValueError('File changed before removal; original retained')
    original.unlink()
    record['removed_original'] = True
    return record


def remove_verified_original(original):
    """Finish an explicitly reviewed migration after re-verifying both files."""
    if Path(original).suffix.lower() == '.gz':
        raise ValueError('Expected the uncompressed original path')
    with _storage_lock(original) as (logical, compressed):
        return _remove_verified(logical, compressed)


def compress_file_lossless(path, *, remove_original=False, min_bytes=0):
    """Compress a closed file; retain original by default for semantic checks.

    min_bytes can leave small files readable. Existing gzip destinations are
    always refused; verify_lossless_copy/remove_verified_original explicitly
    continue a staged migration without rewriting either representation.
    """
    if Path(path).suffix.lower() == '.gz':
        raise ValueError('Expected the uncompressed original path')
    if not isinstance(min_bytes, int) or isinstance(min_bytes, bool) or min_bytes < 0:
        raise ValueError('min_bytes must be a nonnegative integer')
    with _storage_lock(path) as (logical, compressed):
        if compressed.exists() or compressed.is_symlink():
            raise FileExistsError(f'Refusing to overwrite existing output: {compressed}')
        before = _stat_identity(logical)
        if logical.stat().st_size < min_bytes:
            return {**storage_record(logical), 'verified': False, 'removed_original': False,
                    'skipped': 'below_min_bytes'}
        temporary = _temporary(compressed)
        try:
            _gzip_copy(logical, temporary)
            record = verify_lossless_copy(logical, temporary)
            if before != _stat_identity(logical):
                raise ValueError('Source changed during compression; original retained')
            _publish(temporary, compressed)
            record['path'] = str(compressed)
            if remove_original:
                return _remove_verified(logical, compressed)
            return record
        finally:
            temporary.unlink(missing_ok=True)


def finalize_log(path, *, min_bytes=None):
    """Finalize only after all live log writers have closed their handles."""
    threshold = AUTO_COMPRESS_MIN_BYTES if min_bytes is None else min_bytes
    physical = resolve_result_path(path)
    if physical.suffix.lower() == '.gz':
        return physical
    return Path(compress_file_lossless(physical, remove_original=True, min_bytes=threshold)['path'])


def _string_chunks(value):
    yield '"'
    for start in range(0, len(value), STRING_CHARS):
        # Escaping is character-local. Slicing before escaping avoids a second
        # allocation proportional to a multi-gigabyte embedded Aer log string.
        yield json.encoder.encode_basestring(value[start:start + STRING_CHARS])[1:-1]
    yield '"'


def _json_chunks(value, level=0, *, sort_mapping=True):
    """Match json.dump(normalize(value), ensure_ascii=False, indent=2).

    Normalize each node as it is visited. Do not deep-copy the full payload or
    ask JSONEncoder to escape an entire raw log string in one allocation.
    """
    if is_dataclass(value):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, dict):
        # A shallow mapping also preserves normalize's string-key collisions.
        items = sorted(value.items(), key=lambda pair: str(pair[0])) if sort_mapping else value.items()
        mapping = {str(key): item for key, item in items}
        if not mapping:
            yield '{}'
            return
        yield '{\n'
        for index, (key, item) in enumerate(mapping.items()):
            if index:
                yield ',\n'
            yield '  ' * (level + 1)
            yield from _string_chunks(key)
            yield ': '
            yield from _json_chunks(item, level + 1)
        yield '\n' + '  ' * level + '}'
    elif isinstance(value, (tuple, list)):
        if not value:
            yield '[]'
            return
        yield '[\n'
        for index, item in enumerate(value):
            if index:
                yield ',\n'
            yield '  ' * (level + 1)
            yield from _json_chunks(item, level + 1)
        yield '\n' + '  ' * level + ']'
    elif isinstance(value, Path):
        yield from _string_chunks(str(value))
    elif isinstance(value, complex):
        # normalize returns this mapping directly, preserving real/imag order.
        yield from _json_chunks({'real': value.real, 'imag': value.imag}, level, sort_mapping=False)
    elif hasattr(value, 'tolist'):
        yield from _json_chunks(value.tolist(), level)
    elif isinstance(value, str):
        yield from _string_chunks(value)
    elif value is None or isinstance(value, (bool, int)):
        yield json.dumps(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Non-finite identity value')
        yield json.dumps(value, allow_nan=False)
    elif hasattr(value, 'item'):
        yield from _json_chunks(value.item(), level)
    else:
        raise TypeError(f'Unsupported identity value: {type(value).__name__}')


def write_json_new(path, payload):
    """Atomically write normalized JSON; compress large results automatically."""
    _checked_path(_checked_path(path).parent)
    force_gzip = Path(path).suffix.lower() == '.gz'
    with _storage_lock(path) as (logical, compressed):
        assert_no_collision(logical)
        plain_temporary = _temporary(logical)
        gzip_temporary = None
        try:
            # newline=None retains the historical text-mode platform bytes.
            with plain_temporary.open('w', encoding='utf-8', newline=None) as handle:
                for chunk in _json_chunks(payload):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if force_gzip or plain_temporary.stat().st_size >= AUTO_COMPRESS_MIN_BYTES:
                gzip_temporary = _temporary(compressed)
                _gzip_copy(plain_temporary, gzip_temporary)
                verify_lossless_copy(plain_temporary, gzip_temporary)
                assert_no_collision(logical)
                _publish(gzip_temporary, compressed)
                return compressed
            assert_no_collision(logical)
            _publish(plain_temporary, logical)
            return logical
        finally:
            plain_temporary.unlink(missing_ok=True)
            if gzip_temporary is not None:
                gzip_temporary.unlink(missing_ok=True)
