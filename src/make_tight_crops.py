"""Cut tight, defect-centred crops from the plain 600px crops using VOC bboxes.

For each PLAIN base 600px crop, cut a TIGHT window (128x128 native, centred on the
bbox centre, clamped to frame), resized to 256. One tight crop per annotated defect.

Writes two manifests, both board-split (test = splits.json test_boards), both sharing
an IDENTICAL 518-crop test set (primary defect per test crop):
  - manifest_tight.csv         : one tight crop per base crop (primary=nearest-centre
                                 defect). Size-MATCHED to the 600px base set -> isolates
                                 crop scale.
  - manifest_tight_perbbox.csv : train = one tight crop per annotated defect (all bboxes),
                                 test = primary only. Adds data -> isolates dataset SIZE.
"""
import os
import csv
import glob
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
from PIL import Image

from pcb_utils import parse_filename, CLASS_TO_IDX, load_splits

TIGHT = 128
OUT_SIZE = 256
OUT_DIR = 'PCB-cropped-tight/all'
IMG_DIR = 'VOC_PCB/JPEGImages'
ANN_DIR = 'VOC_PCB/Annotations'
W = H = 600


def boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for o in root.findall('object'):
        b = o.find('bndbox')
        out.append((o.find('name').text,
                    int(b.find('xmin').text), int(b.find('ymin').text),
                    int(b.find('xmax').text), int(b.find('ymax').text)))
    return out


def crop_tight(img, cx, cy, size=TIGHT):
    half = size // 2
    left = max(0, min(int(round(cx - half)), W - size))
    top = max(0, min(int(round(cy - half)), H - size))
    return img.crop((left, top, left + size, top + size))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    test_boards = set(load_splits()['test_boards'])
    rows = []
    for xml_path in sorted(glob.glob(os.path.join(ANN_DIR, '*.xml'))):
        base = os.path.basename(xml_path)[:-4]
        p = parse_filename(base + '.jpg')
        if p is None or p['variant'] != 'plain':
            continue
        bs = boxes(xml_path)
        img = Image.open(os.path.join(IMG_DIR, base + '.jpg')).convert('RGB')
        cens = [((xmn + xmx) / 2, (ymn + ymx) / 2) for (_, xmn, ymn, xmx, ymx) in bs]
        prim = int(np.argmin([np.hypot(cx - 300, cy - 300) for cx, cy in cens]))
        split = 'test' if p['board'] in test_boards else 'train'
        for i, (cx, cy) in enumerate(cens):
            fn = f'{base}__b{i}.png'
            crop_tight(img, cx, cy).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS).save(
                os.path.join(OUT_DIR, fn))
            rows.append({'crop_path': os.path.join(OUT_DIR, fn), 'board_id': p['board'],
                         'class_name': p['cls'], 'class_idx': CLASS_TO_IDX[p['cls']],
                         'variant': 'plain', 'split': split,
                         'is_primary': int(i == prim), 'base_crop': base})

    fields = ['crop_path', 'board_id', 'class_name', 'class_idx', 'variant', 'split',
              'is_primary', 'base_crop']

    def write(path, keep):
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(r for r in rows if keep(r))

    # matched: primary defect only (train + test)
    write('results/manifest_tight.csv', lambda r: r['is_primary'] == 1)
    # perbbox: all train defects; test = primary only (shared 518 test set)
    write('results/manifest_tight_perbbox.csv',
          lambda r: r['split'] == 'train' or r['is_primary'] == 1)

    print(f'total tight crops written (one per defect): {len(rows)}')
    for label, keep in [('MATCHED train', lambda r: r['is_primary'] == 1 and r['split'] == 'train'),
                        ('MATCHED test ', lambda r: r['is_primary'] == 1 and r['split'] == 'test'),
                        ('PERBBOX train', lambda r: r['split'] == 'train'),
                        ('PERBBOX test ', lambda r: r['is_primary'] == 1 and r['split'] == 'test')]:
        c = Counter(r['class_name'] for r in rows if keep(r))
        print(f'  {label}: total {sum(c.values()):4d}  ' +
              ' '.join(f'{k[:4]}={c[k]}' for k in sorted(c)))


if __name__ == '__main__':
    main()
