
from copy import deepcopy
import json
from pathlib import Path
import pickle as pkl
import fire
import os
from PIL import Image
import numpy as np

import torch
from cldm.logger import ImageLogger, nested_to

from dataset import COCODataLoader, COCOPanopticDataset
from ldm.util import SeedSwitch, seed_everything
from controlnet_pww import CASMode, create_controlnet_pww_model
import torchvision


def tensor_to_pil(x, rescale=True):
    if rescale:
        x = (x + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
    x = x.numpy()
    x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)


def write_image_grid(savedir, images, i, batsize, rescale=True):
    for k in images:
        grid = torchvision.utils.make_grid(images[k], nrow=batsize)
        grid = tensor_to_pil(grid)
        filename = f"{i}.png"
        path = savedir / filename
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        grid.save(path)
        
        
def do_log_img(imagelogger, batch, pl_module):
    is_train = pl_module.training
    if is_train:
        pl_module.eval()
        
    device = pl_module.device
    numgenerated = 0
    
    batch = nested_to(batch, device)

    with torch.no_grad():
        images = pl_module.log_images(batch, split="train", **imagelogger.log_images_kwargs)
        
    for k in images:
        N = images[k].shape[0]
        numgenerated += N
        images[k] = images[k][:N]
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().cpu()
            if imagelogger.clamp:
                images[k] = torch.clamp(images[k], -1., 1.)

    if is_train:
        pl_module.train()
        
    return images


def main(
        expdir="experiments/controlnet_seg_ft_ca_redist_ma",
        loadckpt="latest*.ckpt",
        datadir="/path/to/unzipped/coco2017/",      # (with panoptic annotations)
        devices=(0,),
        seed=123456,
        threshold=-1.,
        softness=-1.,
        strength=-1.,
        limitpadding=False,
        batsize=1,
        ):    
    localargs = locals().copy()
    expdir = Path(expdir)
    with open(expdir / "args.json") as f:
        trainargs = json.load(f)
        
    args = deepcopy(trainargs)
    for k, v in localargs.items():      # override original args with args specified here
        if v == -1.:
            pass
        elif k == "loadckpt":
            args[k] = args[k]
        else:
            args[k] = v
    
    # unpack args
    cas = args["cas"]
    simpleencode = args["simpleencode"]
    mergeregions = args["mergeregions"]
    limitpadding = args["limitpadding"]
    freezedown = args["freezedown"]
    threshold = args["threshold"]
    softness = args["softness"]
    strength = args["strength"]
    
    seed_everything(seed)
    
    # print(args)
    print(json.dumps(args, indent=4))     
    print(devices, type(devices), devices[0])
    
    cas = (CASMode(cas) + "coco") + "augmentonly"
    cas.augment_global_caption = True
    
    # which ckpt to load
    loadckpt = list(expdir.glob(loadckpt))
    assert len(loadckpt) in (0, 1)
    if len(loadckpt) == 1:
        loadckpt = loadckpt[0]
        args["loadckpt"] += "," + str(loadckpt)
    elif len(loadckpt) > 1:
        raise Exception("multiple matches for loadckpt, unclear")
    else:
        print("ckpt not found, not loading")
    
    model = create_controlnet_pww_model(casmode=cas, freezedown=freezedown, simpleencode=simpleencode, 
                                        threshold=threshold, strength=strength, softness=softness, 
                                        loadckpt=args["loadckpt"])
    model.limitpadding = args["limitpadding"]
    
    print("model loaded")
    
    valid_ds = COCOPanopticDataset(maindir=datadir, split="valid", casmode=cas, simpleencode=simpleencode,
                    mergeregions=mergeregions, limitpadding=limitpadding,
                    max_masks=100, min_masks=1, min_size=128, upscale_to=512)
    
    print(len(valid_ds))
    valid_dl = COCODataLoader(valid_ds, batch_size=batsize, 
                        num_workers=batsize+1,
                        shuffle=False)
    
    # batch = next(iter(valid_dl))
        
    imagelogger = ImageLogger(batch_frequency=999, dl=None, seed=seed)
    
    
    devexamplepath = "coco2017val"
    
    i = 1
    exppath = expdir / f"generated_{devexamplepath}_{i}"
    while exppath.exists():
        i += 1
        exppath = expdir / f"generated_{devexamplepath}_{i}"
        
    exppath.mkdir(parents=True, exist_ok=False)
    print(f"logging in {exppath}")
    with open(exppath / "args.json", "w") as f:
        json.dump(args, f, indent=4)
    
    device = torch.device("cuda", devices[0])
    print("generation device", device)
    model = model.to(device)
    
    allexamples = []
    for _, v in valid_ds.examples:
        for example in v:
            allexamples.append(example)
            
    print("total examples:", len(allexamples))
    numgen = 1
    
    for i, example in enumerate(allexamples):
        _examples = [valid_ds.materialize_example(example) for _ in range(numgen)]
        _batch = valid_ds.collate_fn(_examples)
        images = do_log_img(imagelogger, _batch, model)
        j = 1
        for image in images["generated"]:
            savepath = exppath / f"{example.image_path.stem}_{j}.png"
            while savepath.exists():
                k = 1
                savepath = exppath / f"{example.image_path.stem}_{j}_{k}.png"
                k += 1
            tensor_to_pil(image).save(savepath, format="png")
            j += 1
        
    print(f"done")
            
        
    
    
if __name__ == "__main__":
    fire.Fire(main)