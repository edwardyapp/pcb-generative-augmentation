"""Stage 2: encode TRAIN-POOL crops only into top/bottom code maps (LMDB).

Uses the frozen VQ-VAE-2 (vqvae_560.pt). Filenames are preserved so class_idx is
recoverable at load time (dataset.LMDBDataset). Test-board crops are excluded here
to prevent leaking them back in as "synthetic" data via the prior.
"""
import os
import csv
import pickle
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm
from PIL import Image

from dataset import CodeRow
from vqvae import VQVAE


class ListDS(Dataset):
    def __init__(self, paths, tf):
        self.paths = paths
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        return self.tf(Image.open(p).convert('RGB')), os.path.basename(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoint/vqvae_560.pt')
    ap.add_argument('--manifest', default='results/manifest.csv')
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', default='lmdb/train_pool')
    ap.add_argument('--size', type=int, default=256)
    args = ap.parse_args()
    device = 'cuda'

    with open(args.manifest) as fh:
        paths = sorted(r['crop_path'] for r in csv.DictReader(fh) if r['split'] == args.split)
    print(f'encoding {len(paths)} {args.split} crops -> {args.out}', flush=True)

    tf = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    loader = DataLoader(ListDS(paths, tf), batch_size=128, shuffle=False, num_workers=8)

    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    model = model.to(device).eval()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    env = lmdb.open(args.out, map_size=100 * 1024 ** 3)
    index = 0
    with env.begin(write=True) as txn, torch.no_grad():
        for img, names in tqdm(loader):
            _, _, _, id_t, id_b = model.encode(img.to(device))
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()
            for nm, t, b in zip(names, id_t, id_b):
                txn.put(str(index).encode(), pickle.dumps(CodeRow(top=t, bottom=b, filename=nm)))
                index += 1
        txn.put('length'.encode(), str(index).encode())
    print(f'done, wrote {index} code rows', flush=True)


if __name__ == '__main__':
    main()
