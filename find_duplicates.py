"""Are the 600px crops DUPLICATED across class labels? If so, that is a hard ceiling on the
600px classifier that has nothing to do with the model.

The pixel-space nearest-neighbour null came out at 1.000, which is impossible between genuinely
different boards -- so the "different" boards are not different. Spot checks found pairs like
    rotation_270_light_06_MOUSE_BITE_06_1_600.jpg
    rotation_270_light_06_SHORT_02_1_600.jpg
that are pixel-identical to 0.04/255 but carry DIFFERENT class labels.

Hypothesis: HRIPCB images contain SEVERAL defects of DIFFERENT types. The crop set appears to
emit one copy of the same 600px window per defect class present, each copy named (and labelled)
for one class, with an XML listing only that class's boxes. A classifier then sees the SAME
IMAGE with DIFFERENT labels, which is irreducible label noise -- no model can beat it.

This measures exactly that: cluster the plain 600px crops by pixel identity, and report how many
clusters carry more than one class label, and what accuracy ceiling that imposes.
"""
import glob
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image

from pcb_utils import parse_filename, CLASSES

R = 64
THRESH = 0.9995          # pixel cosine above this = the same photograph


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths = [p for p in sorted(glob.glob('VOC_PCB/JPEGImages/*.jpg'))
             if (q := parse_filename(os.path.basename(p))) and q['variant'] == 'plain']
    meta = [parse_filename(os.path.basename(p)) for p in paths]
    print(f'plain 600px crops: {len(paths)}')

    X = []
    for i in range(0, len(paths), 256):
        arr = [np.asarray(Image.open(p).convert('RGB').resize((R, R), Image.LANCZOS),
                          dtype=np.float32).ravel() / 255. for p in paths[i:i + 256]]
        x = torch.from_numpy(np.stack(arr)).to(device)
        x = x - x.mean(1, keepdim=True)
        X.append(torch.nn.functional.normalize(x, dim=1))
    X = torch.cat(X)

    # union-find over the near-duplicate graph
    parent = list(range(len(paths)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(0, len(paths), 512):
        sim = (X[i:i + 512] @ X.T).cpu().numpy()
        for r in range(sim.shape[0]):
            for j in np.where(sim[r] > THRESH)[0]:
                if int(j) != i + r:
                    union(i + r, int(j))

    clusters = defaultdict(list)
    for i in range(len(paths)):
        clusters[find(i)].append(i)

    multi = {k: v for k, v in clusters.items() if len(v) > 1}
    multi_cls = {k: v for k, v in multi.items()
                 if len({meta[i]['cls'] for i in v}) > 1}
    n_in_multi = sum(len(v) for v in multi.values())
    n_in_multicls = sum(len(v) for v in multi_cls.values())

    print(f'\nclusters of pixel-identical crops: {len(clusters)} clusters for {len(paths)} images')
    print(f'  clusters with >1 image            : {len(multi)}  ({n_in_multi} images, '
          f'{100*n_in_multi/len(paths):.1f}% of the set)')
    print(f'  clusters with >1 CLASS LABEL      : {len(multi_cls)}  ({n_in_multicls} images, '
          f'{100*n_in_multicls/len(paths):.1f}% of the set)   <-- SAME IMAGE, DIFFERENT LABEL')

    # the accuracy ceiling: on a duplicated image the best any model can do is pick the
    # most frequent label in its cluster
    correct = 0
    for v in clusters.values():
        c = Counter(meta[i]['cls'] for i in v)
        correct += c.most_common(1)[0][1]
    print(f'\n  BEST POSSIBLE accuracy for ANY model on these crops: '
          f'{100*correct/len(paths):.1f}%')
    print(f'  (a model cannot distinguish images it cannot tell apart; on a cluster carrying')
    print(f'   several labels the best it can do is always answer the majority one)')

    ex = []
    for k, v in list(multi_cls.items())[:6]:
        ex.append([os.path.basename(paths[i]) for i in v])
        print(f'\n  example cluster ({len(v)} images, classes '
              f'{sorted({meta[i]["cls"] for i in v})}):')
        for i in v[:4]:
            print(f'     {os.path.basename(paths[i])}')

    # do the XMLs of a duplicated cluster list DIFFERENT boxes? (that is the mechanism)
    print('\n  do duplicate images carry different annotations?')
    for k, v in list(multi_cls.items())[:3]:
        for i in v[:2]:
            b = os.path.basename(paths[i])[:-4]
            xml = f'VOC_PCB/Annotations/{b}.xml'
            names = [o.find('name').text for o in ET.parse(xml).getroot().findall('object')]
            print(f'     {b}: {Counter(names)}')
        print()

    json.dump({'n_images': len(paths), 'n_clusters': len(clusters),
               'n_multi_image_clusters': len(multi), 'n_multi_class_clusters': len(multi_cls),
               'images_in_multiclass_clusters': n_in_multicls,
               'accuracy_ceiling': correct / len(paths), 'examples': ex},
              open('results/duplicates.json', 'w'), indent=2)
    print('wrote results/duplicates.json')


if __name__ == '__main__':
    main()
