"""Verify a TDPT distribution and open all files referenced by its evidence index."""
import argparse
import csv
import hashlib
import os
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


def distributed_files():
    files = set()
    for directory, children, names in os.walk(ROOT):
        folder = Path(directory)
        if folder == ROOT and '.git' in children:
            children.remove('.git')
        for name in children + names:
            path = folder / name
            if path.is_symlink() or path.is_junction():
                raise ValueError(f'Redirected path in distribution: {path.relative_to(ROOT)}')
        for name in names:
            if folder == ROOT and name == '.git':
                continue
            files.add((folder / name).relative_to(ROOT).as_posix())
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-tree', action='store_true',
                        help='also reject extra files in a clean extraction or checkout')
    args = parser.parse_args()
    listed = {}
    for line in (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        checksum, name = line.split('  ', 1)
        if name in listed:
            raise ValueError(f'Duplicate manifest entry: {name}')
        with repository_file(name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != checksum:
                raise ValueError(f'SHA-256 mismatch: {name}')
        listed[name] = checksum

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
    expected = set(listed) | {'SHA256SUMS.txt'}
    if described != expected:
        raise ValueError(f'File guide and manifest disagree: {sorted(described ^ expected)}')
    with (ROOT / 'provenance/source_files.csv').open(encoding='utf-8-sig', newline='') as stream:
        sourced = set()
        for row in csv.DictReader(stream):
            name = row['file']
            if name in sourced or listed.get(name) != row['public_sha256']:
                raise ValueError(f'Source map and public file disagree: {name}')
            sourced.add(name)
            if row['treatment'] == 'original bytes' and row['source_sha256'] != listed[name]:
                raise ValueError(f'Original-source digest differs: {name}')
    if args.strict_tree:
        actual = distributed_files()
        if expected != actual:
            raise ValueError(f'Distribution tree and manifest disagree: {sorted(expected ^ actual)}')
    print(f'{len(listed)} SHA-256 digests matched. Opened {opened} file references across {len(ids)} evidence records.')
    print(f'The file guide describes all {len(expected)} distributed files.')
    print(f'Checked the public digests of {len(sourced)} sourced files.')


if __name__ == '__main__':
    main()
