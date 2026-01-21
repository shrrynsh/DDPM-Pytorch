
import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.append(PROJECT_ROOT)

import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from collections import defaultdict
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from scheduler.cosine_noise_scheduler import CosineNoiseScheduler
import json

import torch.distributed as dist
import torch.multiprocessing as mp



def train(rank, world_size, args):
    # -------------------- DDP INIT --------------------
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # -------------------- CONFIG --------------------
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # -------------------- SCHEDULER --------------------
    scheduler = CosineNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps']
    )

    # -------------------- DATASET --------------------
    mnist = MnistDataset('train', im_path=dataset_config['im_path'])

    sampler = DistributedSampler(
        mnist,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    mnist_loader = DataLoader(
        mnist,
        batch_size=train_config['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # -------------------- MODEL --------------------
    model = Unet(model_config).to(device)
    model.train()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank]
    )

    # -------------------- OUTPUT DIR --------------------
    if rank == 0 and not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    # -------------------- CHECKPOINT --------------------
    ckpt_path = os.path.join(
        train_config['task_name'],
        train_config['ckpt_name']
    )

    if os.path.exists(ckpt_path):
        if rank == 0:
            print("Loading checkpoint")
        map_location = {'cuda:0': f'cuda:{rank}'}
        model.module.load_state_dict(
            torch.load(ckpt_path, map_location=map_location)
        )

    # -------------------- OPTIM --------------------
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()

    num_epochs = train_config['num_epochs']
    training_logs = []

    # -------------------- TRAIN LOOP --------------------
    for epoch_idx in range(num_epochs):
        sampler.set_epoch(epoch_idx)

        losses = []
        timestep_losses = defaultdict(list)
        timestep_gradients = defaultdict(list)

        for im in tqdm(mnist_loader, disable=(rank != 0)):
            optimizer.zero_grad()

            im = im.float().to(device)
            noise = torch.randn_like(im)

            t = torch.randint(
                0,
                diffusion_config['num_timesteps'],
                (im.shape[0],),
                device=device
            )

            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())

            # ----- per-timestep loss -----
            with torch.no_grad():
                for i, ts in enumerate(t.tolist()):
                    l = criterion(
                        noise_pred[i:i+1],
                        noise[i:i+1]
                    ).item()
                    timestep_losses[int(ts)].append(l)

            loss.backward()

            # ----- gradient norm -----
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            for ts in t.tolist():
                timestep_gradients[int(ts)].append(total_grad_norm)

            optimizer.step()

        # -------------------- LOGGING (rank 0 only) --------------------
        if rank == 0:
            epoch_log = {
                'epoch': epoch_idx + 1,
                'overall_loss': float(np.mean(losses)),
                'timestep_data': {}
            }

            for ts in timestep_losses:
                epoch_log['timestep_data'][ts] = {
                    'losses': timestep_losses[ts],
                    'gradients': timestep_gradients[ts],
                    'mean_loss': float(np.mean(timestep_losses[ts])),
                    'mean_gradient': float(np.mean(timestep_gradients[ts]))
                }

            training_logs.append(epoch_log)

            print(
                f"Epoch [{epoch_idx+1}/{num_epochs}] "
                f"Loss: {np.mean(losses):.4f}"
            )

            torch.save(
                model.module.state_dict(),
                ckpt_path
            )

    # -------------------- SAVE LOGS --------------------
    if rank == 0:
        log_path = os.path.join(
            train_config['task_name'],
            'training_logs_cosine.json'
        )
        with open(log_path, 'w') as f:
            json.dump(training_logs, f, indent=2)

        print("Training complete")
        print(f"Logs saved to {log_path}")

    dist.destroy_process_group()


# -------------------- MAIN --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM Training (DDP)")
    parser.add_argument(
        '--config',
        dest='config_path',
        default='config/default.yaml',
        type=str
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()  # should be 2 on Kaggle

    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
