import torch
import torchvision
from torchvision.utils import make_grid


import os
import pyiqa
from tqdm import tqdm
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SAVE_ROOT = "outputs/DDPM"
CKPT_SAVE = "checkpoints/DDPM"

os.makedirs(IMG_SAVE_ROOT,exist_ok=True)
os.makedirs(CKPT_SAVE,exist_ok=True)

def train(
        num_epochs,
        data_loader,
        optimizer,
        T,
        scheduler,
        model,
        criterion,
        vae = None
):
    min_loss = 1e9
    for epoch in range(num_epochs):
        losses = []
        for im in tqdm(data_loader):
            optimizer.zero_grad()
            L,A,B = torch.split(im,[1,1,1],dim=1)
            z = torch.zeros_like(L)
            
            im = im.float().to(DEVICE)

            # # moving from pixel space to latent space
            if vae is not None:
                with torch.no_grad():
                    latent_im, _, _, _ = vae.encode(im)

            t = torch.randint(0,T,(im.shape[0],)).to(DEVICE)

            noisy_im, noise = scheduler.forward(im, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"{epoch} Loss {np.mean(losses)}")
        torch.save(model.state_dict(), f"{CKPT_SAVE}/ddpm_{epoch}.pth")
        if min_loss > np.mean(losses):
            min_loss = np.mean(losses)
            torch.save(model.state_dict(), f"{CKPT_SAVE}/best_ddpm.pth")