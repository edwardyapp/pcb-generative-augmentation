import argparse
import sys
import os
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist


def train(epoch, train_loader, val_loader, model, optimizer, scheduler, device, log_losses):
    # Initialize loss trackers
    train_loss = 0
    val_loss = 0

    # Training Phase
    model.train()
    criterion = nn.MSELoss()
    latent_loss_weight = 0.25

    if dist.is_primary():
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")

    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)
        optimizer.zero_grad()

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        train_loss += loss.item()

        # Save reconstruction progress
        if dist.is_primary() and i % 100 == 0:
            model.eval()
            with torch.no_grad():
                sample = img[:8]  # Select 8 images for display
                out, _ = model(sample)
                combined = torch.cat([sample, out], 0)  # Stack originals and reconstructions

                # Save as a 2x8 grid
                utils.save_image(
                    combined,
                    f"sample/epoch_{epoch + 1}_step_{i}.png",
                    nrow=8,  # Number of images in each row
                    normalize=True,
                    value_range=(-1, 1),
                )
            model.train()

    # Validation Phase
    model.eval()
    with torch.no_grad():
        if dist.is_primary():
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation")

        for img, _ in val_loader:
            img = img.to(device)

            out, latent_loss = model(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss
            val_loss += loss.item()

    # Calculate average losses for the epoch
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    # Log losses
    log_losses["train"].append(train_loss)
    log_losses["val"].append(val_loss)

    if dist.is_primary():
        print(
            f"Epoch {epoch + 1} Summary: "
            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

    return train_loss, val_loss


def plot_losses(train_losses, val_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-args.randomRotation, args.randomRotation)),
            transforms.Resize(args.resize),
            transforms.RandomCrop(args.randomCrop),
            transforms.CenterCrop(args.centerCrop),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # Dataset and splitting
    dataset = datasets.ImageFolder(args.path, transform=transform)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = dist.data_sampler(
        train_dataset, shuffle=True, distributed=args.distributed
    )
    val_sampler = dist.data_sampler(
        val_dataset, shuffle=False, distributed=args.distributed
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128 // args.n_gpu, sampler=train_sampler, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128 // args.n_gpu, sampler=val_sampler, num_workers=2
    )

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(train_loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    # Log losses
    log_losses = {"train": [], "val": []}

    os.makedirs("sample", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)

    for epoch in range(args.epoch):
        train(epoch, train_loader, val_loader, model, optimizer, scheduler, device, log_losses)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae_epoch_{epoch + 1}.pt")

    # Plot losses after training
    if dist.is_primary():
        plot_losses(log_losses["train"], log_losses["val"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--resize", type=int, default=292)
    parser.add_argument("--centerCrop", type=int, default=256)
    parser.add_argument("--randomCrop", type=int, default=282)
    parser.add_argument("--randomRotation", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
