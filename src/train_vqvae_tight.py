"""Re-base the VQ-VAE-2 on TIGHT train-pool crops (test boards 06+09 excluded).

Reads the tight one-per-defect manifest (split=='train' -> 4274 crops), trains VQVAE()
with the original recipe (batch 256, lr 3e-4, latent weight 0.25), checkpoints EVERY
epoch to a `vqvae_tight_NNN.pt` prefix (never overwrites the original vqvae_560.pt).
Single GPU (set CUDA_VISIBLE_DEVICES).
"""
import os
import csv
import argparse

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image

from vqvae import VQVAE
from pcb_utils import load_splits


class ListDS(Dataset):
    def __init__(self, paths, tf):
        self.paths = paths
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='results/manifest_tight_perbbox.csv')
    ap.add_argument('--split', default='train')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--epoch', type=int, default=560)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--out_prefix', default='checkpoint/vqvae_tight')
    args = ap.parse_args()
    print(args, flush=True)
    device = 'cuda'

    test_boards = set(load_splits()['test_boards'])
    paths = []
    for r in csv.DictReader(open(args.manifest)):
        if r['split'] == args.split:
            assert r['board_id'] not in test_boards, f"LEAK: test board {r['board_id']}"
            paths.append(r['crop_path'])
    print(f"VQ-VAE tight: training on {len(paths)} crops (test boards {sorted(test_boards)} excluded)",
          flush=True)

    tf = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    # drop_last=True with batch 256 and a 215-crop budget subsample yields ZERO batches: the
    # loop never runs, the model never takes a gradient step, and a randomly-initialised VQ-VAE
    # gets saved and used downstream. That happened. Never again:
    batch = min(args.batch, len(paths))
    loader = DataLoader(ListDS(paths, tf), batch_size=batch, shuffle=True,
                        num_workers=4, drop_last=False)
    assert len(loader) > 0, f'empty loader: {len(paths)} crops, batch {batch}'
    print(f'{len(paths)} crops, batch {batch} -> {len(loader)} batches/epoch', flush=True)

    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    os.makedirs('sample_tight', exist_ok=True)

    for ep in range(args.epoch):
        model.train()
        mse_sum = mse_n = 0
        pbar = tqdm(loader)
        img = None
        for img, _ in pbar:
            img = img.to(device)
            model.zero_grad()
            out, latent = model(img)
            recon = criterion(out, img)
            latent = latent.mean()
            (recon + 0.25 * latent).backward()
            optimizer.step()
            mse_sum += recon.item() * img.shape[0]
            mse_n += img.shape[0]
            pbar.set_description(
                f'ep {ep+1} mse {recon.item():.5f} latent {latent.item():.3f} avgmse {mse_sum/mse_n:.5f}')

        # every-epoch checkpointing wrote 537GB and filled the disk. Keep every 40th + the last.
        if (ep + 1) % 40 == 0 or (ep + 1) == args.epoch:
            torch.save(model.state_dict(), f'{args.out_prefix}_{str(ep + 1).zfill(3)}.pt')

        if img is not None and (ep == 0 or (ep + 1) % 20 == 0):
            model.eval()
            with torch.no_grad():
                s = img[:16]
                o, _ = model(s)
            utils.save_image(torch.cat([s, o], 0), f'sample_tight/{str(ep + 1).zfill(3)}.png',
                             nrow=16, normalize=True, value_range=(-1, 1))

    print('VQ-VAE TIGHT DONE', flush=True)


if __name__ == '__main__':
    main()
