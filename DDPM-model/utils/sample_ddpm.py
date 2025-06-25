import torch
import yaml
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DDPM_CONFIG = "configs/ddpm.yaml"
VAE_CONFIG = "configs/vae.yaml"

with open(DDPM_CONFIG, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
ddpm_model_config = config['model_config']
ddpm_dataset_config = config['dataset_config']
ddpm_training_config = config['training_config']
ddpm_inference_config = config['inference_config']

with open(VAE_CONFIG, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
vae_model_config = config['model_config']


def sample(model, vae, scheduler, num_samples, nrow):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = ddpm_dataset_config['IMG_SIZE'] // 2**sum(vae_model_config['DOWN_SAMPLE'])
    out_shape = (num_samples,vae_model_config['Z_CHANNELS'],im_size,im_size)
    xt = torch.randn(out_shape).to(DEVICE)

    for t in tqdm(reversed(range(ddpm_training_config['NUM_TIMESTEPS']))):
        # Get prediction of noise
        timestep = torch.ones(num_samples, dtype=torch.long, device=DEVICE) * t

        noise_pred = model(xt, timestep)
        
        # Use scheduler to get x0 and xt-1
        xt = scheduler.backward(xt, noise_pred, timestep)

        if t == 0:
            # Decode ONLY the final iamge to save time
            ims = vae.decode(xt)
        else:
            ims = xt

    if num_samples > 1:
        grid = make_grid(ims,nrow=nrow).cpu()
    else:
        grid = ims.squeeze(0).cpu()
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.show()