"""A defect / no-defect detector, so "is a defect visibly present?" becomes a measured number
instead of my opinion.

Why this is needed: the 6-way classifier CANNOT answer the question. It has no "no defect"
class, so it assigns every image -- including a blank green board -- to one of the six defect
types. That is precisely how the epoch-80 check produced a 22.4% "consistency" out of samples
containing no defect at all.

Positives: the tight crops (a defect fills ~22% of the frame, by construction).
Negatives: windows cut from THE SAME 600px crops, same boards, same 128px-native -> 256 recipe,
           but rejected unless they miss every annotated bbox by a margin. So the negative is a
           real photograph of real PCB substrate -- traces, pads, solder mask -- with no defect.
           The detector therefore cannot cheat on colour/texture/board identity; the ONLY thing
           separating the classes is the presence of a defect.

Trained on the train boards, validated on the HELD-OUT boards (06/09). The validation accuracy
is printed so the reader can calibrate exactly how much the "fraction with a defect" number is
worth.
"""
import os
import csv
import glob
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from pcb_utils import parse_filename, load_splits

TIGHT, OUT, W, H = 128, 256, 600, 600
MARGIN = 12                      # a negative must miss every bbox by this many px
NEG_DIR = 'PCB-nodefect/all'
ANN, IMG = 'VOC_PCB/Annotations', 'VOC_PCB/JPEGImages'
CKPT = 'checkpoint/defect_detector.pt'
SEED = 0

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_TF = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                               transforms.ToTensor(), NORM])
TEST_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


class DS(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert('RGB')), y


def boxes(xml):
    root = ET.parse(xml).getroot()
    out = []
    for o in root.findall('object'):
        b = o.find('bndbox')
        out.append(tuple(int(b.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')))
    return out


def clear_of_all(x, y, bs):
    """Is the 128px window at (x,y) clear of EVERY bbox, with margin?"""
    x0, y0, x1, y1 = x, y, x + TIGHT, y + TIGHT
    for bx0, by0, bx1, by1 in bs:
        if not (x1 + MARGIN <= bx0 or x0 - MARGIN >= bx1 or
                y1 + MARGIN <= by0 or y0 - MARGIN >= by1):
            return False
    return True


def build_negatives():
    """One negative per annotated defect, from the same base crop -> class-balanced by design."""
    os.makedirs(NEG_DIR, exist_ok=True)
    rng = random.Random(SEED)
    made = {'train': 0, 'test': 0}
    test_boards = set(load_splits()['test_boards'])
    rows = []
    for xml in sorted(glob.glob(f'{ANN}/*.xml')):
        base = os.path.basename(xml)[:-4]
        p = parse_filename(base + '.jpg')
        if p is None or p['variant'] != 'plain':
            continue
        bs = boxes(xml)
        split = 'test' if p['board'] in test_boards else 'train'
        img = None
        for i in range(len(bs)):                       # one negative per defect
            for _ in range(200):                       # rejection sampling
                x, y = rng.randint(0, W - TIGHT), rng.randint(0, H - TIGHT)
                if clear_of_all(x, y, bs):
                    break
            else:
                continue
            fn = f'{base}__neg{i}.png'
            out = os.path.join(NEG_DIR, fn)
            if not os.path.exists(out):
                if img is None:
                    img = Image.open(f'{IMG}/{base}.jpg').convert('RGB')
                img.crop((x, y, x + TIGHT, y + TIGHT)).resize((OUT, OUT), Image.LANCZOS).save(out)
            rows.append({'crop_path': out, 'board_id': p['board'], 'split': split})
            made[split] += 1
    with open('results/manifest_nodefect.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['crop_path', 'board_id', 'split'])
        w.writeheader(); w.writerows(rows)
    print(f'negatives: train {made["train"]}  test {made["test"]}  -> {NEG_DIR}/')
    return rows


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    negs = build_negatives()

    pos = list(csv.DictReader(open('results/manifest_tight_perbbox.csv')))
    tr = [(r['crop_path'], 1) for r in pos if r['split'] == 'train'] + \
         [(r['crop_path'], 0) for r in negs if r['split'] == 'train']
    te = [(r['crop_path'], 1) for r in pos if r['split'] == 'test'] + \
         [(r['crop_path'], 0) for r in negs if r['split'] == 'test']
    print(f'train {len(tr)} (pos {sum(y for _,y in tr)})   '
          f'held-out {len(te)} (pos {sum(y for _,y in te)})')

    tl = DataLoader(DS(tr, TRAIN_TF), batch_size=64, shuffle=True, num_workers=8)
    vl = DataLoader(DS(te, TEST_TF), batch_size=64, num_workers=8)

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m = m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    for ep in range(10):
        m.train()
        for x, y in tl:
            opt.zero_grad()
            crit(m(x.to(device)), y.to(device)).backward()
            opt.step()
        m.eval()
        yt, yp, ps = [], [], []
        with torch.no_grad():
            for x, y in vl:
                lo = m(x.to(device))
                ps += lo.softmax(1)[:, 1].cpu().tolist()
                yp += lo.argmax(1).cpu().tolist()
                yt += y.tolist()
        yt, yp, ps = np.array(yt), np.array(yp), np.array(ps)
        acc = (yt == yp).mean()
        tpr = (yp[yt == 1] == 1).mean()
        tnr = (yp[yt == 0] == 0).mean()
        print(f'  epoch {ep+1}/10  held-out acc {acc:.4f}  '
              f'defect-recall {tpr:.4f}  no-defect-recall {tnr:.4f}', flush=True)

    torch.save(m.state_dict(), CKPT)
    print(f'\nHELD-OUT (boards 06/09): accuracy {acc:.4f}, '
          f'detects a real defect {100*tpr:.1f}% of the time, '
          f'correctly says "no defect" on clean substrate {100*tnr:.1f}% of the time.')
    print(f'=> when this detector says "no defect" in a generated crop, it is right ~{100*tnr:.0f}% '
          f'of the time on real clean board.')
    print(f'saved {CKPT}')


if __name__ == '__main__':
    main()
