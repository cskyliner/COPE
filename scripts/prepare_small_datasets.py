"""Prepare official ReID image archives in the directory layout COPE expects."""

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile


def copy_file(archive, info, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with archive.open(info) as source, target.open('xb') as dest:
        shutil.copyfileobj(source, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['market', 'pduke', 'occluded'], required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    with ZipFile(args.archive) as z:
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or '__MACOSX' in path.parts:
                continue
            if '..' in path.parts or path.is_absolute():
                raise ValueError(f'Unsafe archive entry: {path}')
            if args.dataset == 'market':
                if len(path.parts) != 3 or path.parts[1] not in {'bounding_box_train', 'bounding_box_test', 'query'} or path.suffix.lower() != '.jpg':
                    continue
                folder = path.parts[1]
                dest = args.root / 'market1501' / folder / path.name
            elif args.dataset == 'pduke':
                if len(path.parts) != 4 or path.parts[0] not in {'train', 'test'} or path.parts[1] not in {'occluded_body_images', 'whole_body_images'} or path.suffix.lower() != '.jpg':
                    continue
                folder = '/'.join(path.parts[:2])
                dest = args.root / 'P-DukeMTMC-reid' / Path(*path.parts)
            else:
                if len(path.parts) != 3 or path.parts[0] not in {'occluded_body_images', 'whole_body_images'} or path.suffix.lower() != '.tif':
                    continue
                pid, index = path.stem.split('_')
                if pid != path.parts[1] or not (pid.isdigit() and index.isdigit()):
                    raise ValueError(f'Unexpected Occluded-ReID filename: {path}')
                folder, camera = ('query', 1) if path.parts[0] == 'occluded_body_images' else ('bounding_box_test', 2)
                dest = args.root / 'Occluded_REID' / folder / f'{pid}_c{camera}_{index}.tif'
            copy_file(z, info, dest)
            counts[folder] += 1
    expected = {'market': {'bounding_box_train': 12936, 'bounding_box_test': 19732, 'query': 3368},
                'pduke': {'train/whole_body_images': 10280, 'train/occluded_body_images': 2647, 'test/whole_body_images': 9053, 'test/occluded_body_images': 2163},
                'occluded': {'query': 1000, 'bounding_box_test': 1000}}[args.dataset]
    if dict(counts) != expected:
        raise ValueError(f'Unexpected image counts: {dict(counts)}; expected {expected}')
    print(json.dumps({'dataset': args.dataset, 'counts': dict(counts), 'status': 'prepared'}, indent=2))


if __name__ == '__main__':
    main()
