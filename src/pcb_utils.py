"""Shared helpers for the ICETA 2026 synthetic-defect experiment.

Single source of truth for: class order, filename parsing (board id / class /
augmentation variant), and the board-level split contract.
"""
import os
import re
import json

# class_idx order (fixed, used everywhere)
CLASSES = [
    'missing_hole',
    'mouse_bite',
    'open_circuit',
    'short',
    'spur',
    'spurious_copper',
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

CROP_DIR = 'PCB-cropped/all'

# NB: spurious_copper must precede spur in the alternation (prefix collision).
_CLASS_ALT = 'missing_hole|mouse_bite|open_circuit|spurious_copper|short|spur'
_PAT = re.compile(
    r'^(?:rotation_(?P<rot>90|270)_)?(?P<lflag>l_)?'
    r'light_(?P<board>\d+)_(?P<cls>' + _CLASS_ALT + r')_'
    r'(?P<mm>\d+)_(?P<k>\d+)_600(?:__b(?P<bbox>\d+))?\.(?:jpg|png)$'
)


def parse_filename(name):
    """Parse a crop basename -> dict(board, cls, mm, k, variant) or None.

    variant in {plain, l, rot90, rot270}. board is a zero-padded string id.
    """
    m = _PAT.match(name)
    if m is None:
        return None
    if m['rot'] == '90':
        variant = 'rot90'
    elif m['rot'] == '270':
        variant = 'rot270'
    elif m['lflag']:
        variant = 'l'
    else:
        variant = 'plain'
    return {
        'board': m['board'],
        'cls': m['cls'],
        'mm': m['mm'],
        'k': m['k'],
        'variant': variant,
    }


def load_splits(path='results/splits.json'):
    with open(path) as f:
        return json.load(f)


def assert_no_leak(train_items, test_boards):
    """train_items: iterable of board-id strings used in training.
    Raises if any test board leaked into training."""
    test_boards = set(test_boards)
    leaked = test_boards & set(train_items)
    assert not leaked, f'LEAK: test boards {leaked} present in training data'
