"""Verify a TDPT distribution and open all files referenced by its evidence index."""
import csv
import hashlib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def repository_file(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Expected a repository-relative path in the archive: {name!r}')
    path = (ROOT / relative).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'Invalid file: {name}')
    return path


def main():
    listed = set()
    for line in (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        checksum, name = line.split('  ', 1)
        if name in listed:
            raise ValueError(f'Duplicate manifest entry: {name}')
        with repository_file(name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != checksum:
                raise ValueError(f'SHA-256 mismatch: {name}')
        listed.add(name)

    with (ROOT / 'outputs/evidence_index.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = set()
    opened = 0
    for row in rows:
        if not row['id'] or row['id'] in ids:
            raise ValueError(f'Missing or duplicate evidence ID: {row}')
        ids.add(row['id'])
        for field in ('input', 'raw_result', 'reference'):
            for name in filter(None, row[field].split('|')):
                if name not in listed:
                    raise ValueError(f"{row['id']} points outside the manifest: {name}")
                with repository_file(name).open('rb') as referenced:
                    if not referenced.read(1):
                        raise ValueError(f'Empty evidence file: {name}')
                opened += 1
    with (ROOT / 'provenance/file_roles.csv').open(encoding='utf-8-sig', newline='') as stream:
        described = set()
        for row in csv.DictReader(stream):
            name = row['repository_path']
            if name in described or not row['role'].strip() or not row['purpose'].strip():
                raise ValueError(f'Duplicate or incomplete file guide entry: {name}')
            described.add(name)
    expected = listed | {'SHA256SUMS.txt'}
    if described != expected:
        raise ValueError(f'File guide and manifest disagree: {sorted(described ^ expected)}')
    print(f'{len(listed)} SHA-256 digests matched. Opened {opened} file references across {len(ids)} evidence records.')
    print(f'The file guide describes all {len(expected)} distributed files.')


if __name__ == '__main__':
    main()
