"""Stage 1: board-level split -> results/splits.json + results/manifest.csv.

Test = 2 held-out boards, NEVER touched by classifier/generator/filter.
All 4 augmentation variants of a base crop share a board, so a board split has
zero augmentation leakage.
"""
import os
import csv
import glob
import json
from collections import Counter

from pcb_utils import CLASSES, CLASS_TO_IDX, CROP_DIR, parse_filename

TEST_BOARDS = ['06', '09']
SEED = 0


def main():
    os.makedirs('results', exist_ok=True)
    files = sorted(glob.glob(os.path.join(CROP_DIR, '*.jpg')))
    rows, unparsed = [], []
    for f in files:
        p = parse_filename(os.path.basename(f))
        if p is None:
            unparsed.append(f)
            continue
        rows.append({
            'crop_path': f,
            'board_id': p['board'],
            'class_name': p['cls'],
            'class_idx': CLASS_TO_IDX[p['cls']],
            'variant': p['variant'],
            'split': 'test' if p['board'] in TEST_BOARDS else 'train',
        })
    assert not unparsed, f'{len(unparsed)} unparsed, e.g. {unparsed[:5]}'

    train_boards = sorted({r['board_id'] for r in rows if r['split'] == 'train'})
    test_boards = sorted({r['board_id'] for r in rows if r['split'] == 'test'})
    assert set(test_boards) == set(TEST_BOARDS), (test_boards, TEST_BOARDS)
    assert not (set(train_boards) & set(TEST_BOARDS)), 'LEAK: test board in train'

    with open('results/manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['crop_path', 'board_id', 'class_name',
                                           'class_idx', 'variant', 'split'])
        w.writeheader()
        w.writerows(rows)

    n_train = sum(r['split'] == 'train' for r in rows)
    n_test = sum(r['split'] == 'test' for r in rows)
    splits = {
        'seed': SEED,
        'test_boards': TEST_BOARDS,
        'train_boards': train_boards,
        'classes': CLASSES,
        'class_to_idx': CLASS_TO_IDX,
        'n_crops': len(rows),
        'n_train_crops': n_train,
        'n_test_crops': n_test,
    }
    with open('results/splits.json', 'w') as fh:
        json.dump(splits, fh, indent=2)

    print(f'manifest rows: {len(rows)}  train: {n_train}  test: {n_test}')
    print(f'train boards ({len(train_boards)}): {train_boards}')
    print(f'test  boards ({len(test_boards)}): {test_boards}')
    for split in ['train', 'test']:
        c = Counter(r['class_name'] for r in rows
                    if r['split'] == split and r['variant'] == 'plain')
        print(f'[{split}] base(plain) crops/class: '
              + ' '.join(f'{k}={c[k]}' for k in CLASSES) + f'  total={sum(c.values())}')


if __name__ == '__main__':
    main()
