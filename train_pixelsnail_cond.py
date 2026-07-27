"""Stage 3: class-conditional PixelSNAIL training (top or bottom prior).

Adds a per-class global bias (pixelsnail.py: n_img_class / class_emb). Warm-starts
from the existing unconditional checkpoint (only class_emb is new -> strict=False).
Checkpoints every epoch. Run one hier per process; drive top then bottom.
"""
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL


def train(args, epoch, loader, model, optimizer, device):
    loader = tqdm(loader)
    crit = nn.CrossEntropyLoss()
    for top, bottom, label in loader:
        model.zero_grad()
        top = top.to(device)
        label = label.to(device)
        if args.hier == 'top':
            target = top
            out, _ = model(top, class_label=label)
        else:
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top, class_label=label)
        loss = crit(out, target)
        loss.backward()
        optimizer.step()
        _, pred = out.max(1)
        acc = (pred == target).float().mean()
        loader.set_description(
            f'ep {epoch + 1} {args.hier} loss {loss.item():.4f} acc {acc.item():.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epoch', type=int, default=60)
    ap.add_argument('--hier', default='top', choices=['top', 'bottom'])
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--channel', type=int, default=256)
    ap.add_argument('--n_res_block', type=int, default=4)
    ap.add_argument('--n_res_channel', type=int, default=256)
    ap.add_argument('--n_out_res_block', type=int, default=0)
    ap.add_argument('--n_cond_res_block', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--n_img_class', type=int, default=6)
    ap.add_argument('--save_every', type=int, default=40)   # not every epoch: 350MB a pop
    ap.add_argument('--warm', default=None)
    ap.add_argument('--out_prefix', default='checkpoint/pixelsnail_cond')
    ap.add_argument('--path', default='lmdb/train_pool')
    args = ap.parse_args()
    print(args, flush=True)
    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)

    if args.hier == 'top':
        model = PixelSNAIL([32, 32], 512, args.channel, 5, 4, args.n_res_block,
                           args.n_res_channel, dropout=args.dropout,
                           n_out_res_block=args.n_out_res_block, n_img_class=args.n_img_class)
    else:
        model = PixelSNAIL([64, 64], 512, args.channel, 5, 4, args.n_res_block,
                           args.n_res_channel, attention=False, dropout=args.dropout,
                           n_cond_res_block=args.n_cond_res_block,
                           cond_res_channel=args.n_res_channel, n_img_class=args.n_img_class)

    if args.warm:
        sd = torch.load(args.warm, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'warm-start {args.warm}: missing={list(missing)} unexpected={list(unexpected)}',
              flush=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = nn.DataParallel(model)

    # A PixelSNAIL checkpoint is ~350MB. Saving one EVERY epoch across 320-epoch runs wrote
    # 537GB and filled the disk mid-run, which corrupted checkpoints and silently poisoned
    # everything downstream. Keep every 40th (the conditioning-check epochs) and the last.
    for i in range(args.epoch):
        train(args, i, loader, model, optimizer, device)
        ep = i + 1
        if ep % args.save_every == 0 or ep == args.epoch:
            torch.save({'model': model.module.state_dict(), 'args': args},
                       f'{args.out_prefix}_{args.hier}_{str(ep).zfill(3)}.pt')


if __name__ == '__main__':
    main()
