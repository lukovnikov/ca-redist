import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import SeedSwitch, seed_everything


def nested_to(x, device):
    if isinstance(x, list):
        return [nested_to(xe, device) for xe in x]
    elif isinstance(x, dict):
        return {k: nested_to(v, device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        raise NotImplemented(f"Type {type(x)} not supported by nested_to().")
    return ret


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=-1, clamp=True, increase_log_steps=True, seed=42,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, dl=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.dl = dl    # dataloader instead of training batch
        self.seed = seed
        self.batch_size = self.dl.batch_size if self.dl is not None else -1

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=self.batch_size)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:03}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)
            
    def do_log_img(self, pl_module, save_dir, split="train"):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()
            
        device = pl_module.device
        numgenerated = 0
        
        allimages = []
        
        for batch_idx, batch in enumerate(iter(self.dl)):
            batch = nested_to(batch, device)

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                allimages += images["all"].unbind(0)
                
            for k in images:
                N = images[k].shape[0]
                numgenerated += N
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(save_dir, split, images,
                        pl_module.global_step, pl_module.current_epoch, batch_idx)
            
            if self.max_images >= 0 and numgenerated >= self.max_images:
                break

        if is_train:
            pl_module.train()
            
        return allimages

    def log_img(self, pl_module, global_step, split="train"):
        if isinstance(self.seed, SeedSwitch):
            self.seed.__enter__()
        else:
            seed_everything(self.seed)
        # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(global_step) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images)):
            # logger = type(pl_module.logger)
            self.do_log_img(pl_module=pl_module, save_dir=pl_module.logger.save_dir, split=split)
            
        if isinstance(self.seed, SeedSwitch):
            self.seed.__exit__(None, None, None)

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0
    
    # def on_train_start(self, trainer, pl_module):
    #     if not self.disabled:
    #         # self.log_img(pl_module, batch, batch_idx, split="train")
    #         self.log_img(pl_module, None, trainer.global_step, split="train")      # Fixed logging to actually log every N steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            # self.log_img(pl_module, batch, batch_idx, split="train")
            self.log_img(pl_module, trainer.global_step, split="train")      # Fixed logging to actually log every N steps
