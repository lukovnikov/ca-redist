import copy
import json
from pathlib import Path
import re
from PIL import Image
import requests
import torch

# from cldm.logger import nested_to
# from dataset import COCOPanopticDataset, COCOPanopticExample, COCODataLoader
import pickle

import numpy as np
import tqdm

import fire


def load_clip_model(modelname="openai/clip-vit-base-patch32"):
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained(modelname)
    processor = CLIPProcessor.from_pretrained(modelname)
    return model, processor


def display_example(x):
    img = x.load_image()
    seg_img = x.load_seg_image()
    print(x.captions)
    print(repr(x.seg_info))
    return None


def region_code_to_rgb(rcode):
    B = rcode // 256**2
    rcode = rcode % 256**2
    G = rcode // 256
    R = rcode % 256
    return (R, G, B)


def rgb_to_regioncode(r, g, b):
    ret = r + g * 256 + b * (256**2)
    return ret
    
    
def run():
    model, processor = load_clip_model()
    
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print(probs)
    
    
class MetricScore():
    def __init__(self, name, higher_is_better=True):
        self.name = name
        self.higher_is_better = higher_is_better
        
    @classmethod
    def from_string(cls, x):   # must be in format like __str__
        m = re.match(r"Metric\[([^,]+),(.)\]", x)
        if m is not None:
            name, higherbetter = m.groups()
            return cls(name, higherbetter == "+")
        else:
            return None
        
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other):
        return str(self).__eq__(str(other))
    
    def __str__(self):
        return f"Metric[{self.name},{'+' if self.higher_is_better else '-'}]"
    
    def __repr__(self):
        return str(self)
    
    
class DictExampleWrapper:
    def __init__(self, datadict):
        super().__init__()
        self._datadict = datadict
        
    def load_image(self):
        return self._datadict["image"]
    
    def load_seg_image(self):
        return self._datadict["seg_img"]
    
    @property
    def captions(self):
        return [self._datadict["caption"]]
    
    
class LocalCLIPEvaluator():
    LOGITS = MetricScore("localclip_logits")
    COSINE = MetricScore("localclip_cosine")
    PROBS = MetricScore("localclip_probs")
    ACCURACY = MetricScore("localclip_acc")
    
    def __init__(self, clipmodel, clipprocessor, tightcrop=True, fill="none"):
        super().__init__()
        self.clipmodel, self.clipprocessor, self.tightcrop, self.fill = clipmodel, clipprocessor, tightcrop, fill
        
        
    def prepare_example(self, x, tightcrop=None, fill=None):
        if isinstance(x, DictExampleWrapper):
            return self.prepare_example_dictwrapper(x, tightcrop=tightcrop, fill=fill)
        else:
            return self.prepare_example_controlnet(x, tightcrop=tightcrop, fill=fill)
        
    def prepare_example_dictwrapper(self, x, tightcrop=None, fill=None):
        tightcrop = self.tightcrop if tightcrop is None else tightcrop
        fill = self.fill if fill is None else fill
        
        regionimages = []
        regiondescriptions = []
        
        image = np.array(x.load_image())
        
        if "masks" in x._datadict:
            for mask, regioncaption in zip(x._datadict["masks"], x._datadict["prompts"]):
                height, width = mask.shape
            
                if tightcrop:
                    bbox_left = np.where(mask > 0)[1].min()
                    bbox_right = np.where(mask > 0)[1].max()
                    bbox_top = np.where(mask > 0)[0].min()
                    bbox_bottom = np.where(mask > 0)[0].max()
                    
                    bbox_size = ((bbox_right-bbox_left), (bbox_bottom - bbox_top))
                    bbox_center = (bbox_left + bbox_size[0] / 2, bbox_top + bbox_size[1] / 2)
                    
                    _bbox_size = (max(bbox_size), max(bbox_size))
                    _bbox_center = (min(max(_bbox_size[0] / 2, bbox_center[0]), width - _bbox_size[0] /2), 
                                    min(max(_bbox_size[1] / 2, bbox_center[1]), height - _bbox_size[1] /2))
                    
                    _image = image[int(_bbox_center[1]-_bbox_size[1]/2):int(_bbox_center[1] + _bbox_size[1]/2),
                                int(_bbox_center[0]-_bbox_size[0]/2):int(_bbox_center[0] + _bbox_size[0]/2)]
                    _mask = mask[int(_bbox_center[1]-_bbox_size[1]/2):int(_bbox_center[1] + _bbox_size[1]/2),
                                int(_bbox_center[0]-_bbox_size[0]/2):int(_bbox_center[0] + _bbox_size[0]/2)]
                else:
                    _image = image
                    _mask = mask
                
                if fill in ("none", "") or fill is not None:
                    regionimage = _image
                else:
                    if fill == "black":
                        fillcolor = np.array([0, 0, 0])
                    elif fill == "white":
                        fillcolor = np.array([1, 1, 1]) * 255
                    else:
                        avgcolor = np.mean((1 - _mask) * _image, (0, 1))
                        avgcolor = np.round(avgcolor).astype(np.int32)
                        fillcolor = avgcolor
                    regionimage = _image * _mask + fillcolor[None, None, :] * (1 - _mask)
            
                regionimages.append(regionimage)
                regiondescriptions.append(regioncaption)            
        
        elif "bboxes" in x._datadict:
            assert fill in ("none", "") or fill is None
            height, width, _ = image.shape
            
            for bbox, regioncaption in zip(x._datadict["bboxes"], x._datadict["bbox_captions"]):
                if tightcrop:
                    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
                    bbox_left, bbox_top, bbox_right, bbox_bottom = \
                        int(bbox_left * width), int(bbox_top * height), int(bbox_right * width), int(bbox_bottom * height)
                
                    bbox_size = ((bbox_right-bbox_left), (bbox_bottom - bbox_top))
                    bbox_center = (bbox_left + bbox_size[0] / 2, bbox_top + bbox_size[1] / 2)
                    
                    _bbox_size = (max(bbox_size), max(bbox_size))
                    _bbox_center = (min(max(_bbox_size[0] / 2, bbox_center[0]), width - _bbox_size[0] /2), 
                                    min(max(_bbox_size[1] / 2, bbox_center[1]), height - _bbox_size[1] /2))
                    
                    _image = image[int(_bbox_center[1]-_bbox_size[1]/2):int(_bbox_center[1] + _bbox_size[1]/2),
                                int(_bbox_center[0]-_bbox_size[0]/2):int(_bbox_center[0] + _bbox_size[0]/2)]
                else:
                    _image = image
                    
                regionimage = _image
                
                regionimages.append(regionimage)
                regiondescriptions.append(regioncaption) 
        
        else:
            pass
            
        return regionimages, regiondescriptions
        
    def prepare_example_controlnet(self, x, tightcrop=None, fill=None):
        tightcrop = self.tightcrop if tightcrop is None else tightcrop
        fill = self.fill if fill is None else fill
        
        regionimages = []
        regiondescriptions = []
        
        image = np.array(x.load_image())
        seg_image = np.array(x.load_seg_image())
        seg_image_codes = rgb_to_regioncode(*np.split(seg_image, 3, -1))
        for regioncode, regioninfo in x.seg_info.items():
            mask = (seg_image_codes == regioncode) * 1.
            height, width, _ = mask.shape
            
            if tightcrop:
                bbox_left = np.where(mask > 0)[1].min()
                bbox_right = np.where(mask > 0)[1].max()
                bbox_top = np.where(mask > 0)[0].min()
                bbox_bottom = np.where(mask > 0)[0].max()
                
                bbox_size = ((bbox_right-bbox_left), (bbox_bottom - bbox_top))
                bbox_center = (bbox_left + bbox_size[0] / 2, bbox_top + bbox_size[1] / 2)
                
                _bbox_size = (max(bbox_size), max(bbox_size))
                _bbox_center = (min(max(_bbox_size[0] / 2, bbox_center[0]), width - _bbox_size[0] /2), 
                                min(max(_bbox_size[1] / 2, bbox_center[1]), height - _bbox_size[1] /2))
                
                _image = image[int(_bbox_center[1]-_bbox_size[1]/2):int(_bbox_center[1] + _bbox_size[1]/2),
                            int(_bbox_center[0]-_bbox_size[0]/2):int(_bbox_center[0] + _bbox_size[0]/2)]
                _mask = mask[int(_bbox_center[1]-_bbox_size[1]/2):int(_bbox_center[1] + _bbox_size[1]/2),
                            int(_bbox_center[0]-_bbox_size[0]/2):int(_bbox_center[0] + _bbox_size[0]/2)]
            else:
                _image = image
                _mask = mask
            
            if fill in ("none", "") or fill is not None:
                regionimage = _image
            else:
                if fill == "black":
                    fillcolor = np.array([0, 0, 0])
                elif fill == "white":
                    fillcolor = np.array([1, 1, 1]) * 255
                else:
                    avgcolor = np.mean((1 - _mask) * _image, (0, 1))
                    avgcolor = np.round(avgcolor).astype(np.int32)
                    fillcolor = avgcolor
                regionimage = _image * _mask + fillcolor[None, None, :] * (1 - _mask)
                
            regioncaption = regioninfo["caption"]
            
            # extra region caption from global prompt
            assert len(x.captions) == 1
            caption = x.captions[0]
            matches = re.findall("{([^:]+):" + str(regioncode) + "}", caption)
            assert len(matches) == 1
            regioncaption = matches[0]
            
            regionimages.append(regionimage)
            regiondescriptions.append(regioncaption)
            
        return regionimages, regiondescriptions
    
    def run(self, x):
        regionimages, regiondescriptions = self.prepare_example(x)
            
        inputs = self.clipprocessor(text=regiondescriptions, images=regionimages, return_tensors="pt", padding=True)
        inputs = inputs.to(self.clipmodel.device)
        outputs = self.clipmodel(**inputs)
        logits_per_image = outputs.logits_per_image
        cosine_per_image = (outputs.image_embeds @ outputs.text_embeds.T)
        prob_per_image = logits_per_image.softmax(-1)
        acc_per_image = logits_per_image.softmax(-1).max(-1)[1] == torch.arange(len(logits_per_image), device=logits_per_image.device)
        # probs = logits_per_image.softmax(dim=1)
        return {self.LOGITS: logits_per_image.diag().mean().detach().cpu().item(),
                self.COSINE: cosine_per_image.diag().mean().detach().cpu().item(),
                self.PROBS: prob_per_image.diag().mean().detach().cpu().item(),
                self.ACCURACY: acc_per_image.float().mean().detach().cpu().item()}


class AestheticsPredictor():
    SCORE = MetricScore("laion_aest_score")
    
    weight_url_dict = {
        "openai/clip-vit-base-patch32": "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth",
        "openai/clip-vit-large-patch14": "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth",
    }
    hf_clipname_to_openclip = {
        "openai/clip-vit-base-patch32": ("openai", "ViT-B-32"),
        "openai/clip-vit-large-patch14": ("openai", "ViT-L-14"),
    }
    
    def __init__(self, clipname="openai/clip-vit-large-patch14", weightdir="extramodels/aesthetics_predictor", device=torch.device("cuda")):
    # def __init__(self, clipmodel, clipprocessor, weightdir="extramodels/aesthetics_predictor"):
        super().__init__()
        import open_clip
        cliptrainer, clipmodelname = self.hf_clipname_to_openclip[clipname]
        self.clipmodel, _, self.clipprocess = open_clip.create_model_and_transforms(clipmodelname, pretrained=cliptrainer)
        self.clipmodel.to(device)
        weighturl = self.weight_url_dict[clipname]
        self.device = device
        # self.clipmodel, self.clipprocessor = clipmodel, clipprocessor
        # weighturl = self.weight_url_dict[self.clipmodel.name_or_path]
        weightpath = Path(weightdir) / Path(weighturl).name
        if not weightpath.exists():
            # download weights
            weightpath.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(weighturl)
            with open(weightpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        
        assert weightpath.exists()
        if clipname == "openai/clip-vit-large-patch14":
        # if self.clipmodel.name_or_path == "openai/clip-vit-large-patch14":
            self.m = torch.nn.Linear(768, 1)
        else:
            self.m = torch.nn.Linear(512, 1)
        self.m.load_state_dict(torch.load(weightpath))
        self.m.eval()
        self.m.to(self.device)
        
    def run_image(self, x:Image.Image):
        image_input = self.clipprocess(x).unsqueeze(0).to(self.device)
        image_features = self.clipmodel.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = self.m(image_features)
        # image = np.array(x)
        # inputs = self.clipprocessor(text=[""], images=[image], return_tensors="pt", padding=True)
        # inputs = inputs.to(self.clipmodel.device)
        # outputs = self.clipmodel(**inputs)
        # image_embed = outputs.image_embeds
        # image_embed /= image_embed.norm(dim=-1, keepdim=True)
        # prediction = self.m(image_embed)
        assert prediction.shape == (1, 1)
        return prediction[0, 0].cpu().item()
        
    def run(self, x):
        image = x.load_image()
        return {self.SCORE: self.run_image(image), }
    

class MANIQAEvaluator():
    SCORE = MetricScore("maniqa")
    def __init__(self, device=torch.device("cuda"), **kw):
        import pyiqa
        super().__init__(**kw)
        self.metric = pyiqa.create_metric("maniqa", device=device, as_loss=False)
        
    def run(self, x):
        image = x.load_image()
        score = self.metric(image)
        assert score.shape == (1, 1)
        return {self.SCORE: score[0, 0].detach().cpu().item()}
    

class BRISQUEEvaluator():
    SCORE = MetricScore("brisque", higher_is_better=False)
    def __init__(self, device=torch.device("cuda"), **kw):
        import pyiqa
        super().__init__(**kw)
        self.metric = pyiqa.create_metric("brisque", device=device, as_loss=False)
        
    def run(self, x):
        image = x.load_image()
        score = self.metric(image)
        assert score.shape == (1,)
        return {self.SCORE: score[0].detach().cpu().item()}
        
    
def do_example(x, **evaluators):
    if isinstance(x, dict):
        x = DictExampleWrapper(x)
    ret = {}
    for _, evaluator in evaluators.items():
        resultdic = evaluator.run(x)
        for k, v in resultdic.items():
            ret[k] = v
    return ret

    
def run2(path="coco2017.4dev.examples.pkl"):
    model, processor = load_clip_model()
    with open(path, "rb") as f:
        loadedexamples = pickle.load(f)
        
    all_logits, all_cosines, all_probs, all_accs = [], [], [], []
        
    for example in loadedexamples:
        logits, cosines, probs, accs, descriptions = do_example(example, model, processor, tightcrop=True, fill="avg")
        all_logits.append(logits.mean().cpu().item())
        all_cosines.append(cosines.mean().cpu().item())
        all_probs.append(probs.mean().cpu().item())
        all_accs.append(accs.mean().cpu().item())
        
    print(all_accs)
    print("done")
    
    
def load_everything(clip_version="openai/clip-vit-large-patch14", device=0, tightcrop=True, fill="none"):
    print("loading clip model")
    model, processor = load_clip_model(modelname=clip_version)
    model = model.to(device)
        
    print("loading models for evaluation")
    clipevaluator = LocalCLIPEvaluator(model, processor, tightcrop=tightcrop, fill=fill)
    aestheticspredictor = AestheticsPredictor(clip_version, device=device)
    maniqaevaluator = MANIQAEvaluator(device=device)
    brisque = BRISQUEEvaluator(device=device)
    
    print("models loaded")
    # return {
    #     "clipeval": clipevaluator, 
    #     "aesthetics": aestheticspredictor, 
    #     "maniqa": maniqaevaluator, 
    #     "brisque": brisque,
    # }
    return clipevaluator, aestheticspredictor, maniqaevaluator, brisque
    
    
def run3(
         path="/USERSPACE/lukovdg1/controlnet11/checkpoints/v5/checkpoints_coco_global_v5_exp_1/generated_extradev.pkl_1",
         tightcrop=True,
         fill="none",
         device=0,
        #  clip_version="openai/clip-vit-base-patch32",
         clip_version="openai/clip-vit-large-patch14",
         evalmodels=None,
         ):
    device = torch.device("cuda", device)
    batchespath = Path(path) / "outbatches.pkl"
    
    with open(batchespath, "rb") as f:
        loadedexamples = pickle.load(f)
        
    if evalmodels is None:
        evalmodels = load_everything(clip_version=clip_version, device=device, tightcrop=tightcrop, fill=fill)
        
    clipevaluator, aestheticspredictor, maniqaevaluator, brisque = evalmodels
    
    colnames = None
    higherbetter = []
    allmetrics = []
        
    with torch.no_grad():
        print("iterating over batches")
        for batch in tqdm.tqdm(loadedexamples):
            batch_metrics = []
            for example in batch:       # all examples in one batch have been generated as different seeds of the same starting point
                outputs = do_example(example, clipevaluator=clipevaluator, aestheticspredictor=aestheticspredictor, maniqa=maniqaevaluator, brisque=brisque)
                outputcolnames, outputdata = zip(*sorted(outputs.items(), key=lambda x: x[0].name))
                if colnames is None:
                    colnames = copy.deepcopy(outputcolnames)
                assert colnames == outputcolnames
                batch_metrics.append(tuple(outputdata))
                # local_clip_metrics = outputs["clipevaluator"]      # metrics here are over regions in this one example
                # example_metrics = [metric.mean().cpu().item() for metric in local_clip_metrics]
                # example_metrics.append(outputs["aestheticspredictor"])
                # example_metrics.append(outputs["maniqa"])
                # example_metrics.append(outputs["brisque"])
                # batch_metrics.append(tuple(example_metrics))
            allmetrics.append(batch_metrics)        # aggregate over all batches --> (numbats, numseeds, nummetrics)
    
    tosave = {"colnames": [str(colname) for colname in colnames],
              "data": allmetrics}
    with open(Path(path) / "evaluation_results_raw.json", "w") as f:
        json.dump(tosave, 
                  f, 
                  indent=4)
    print(f"saved raw results in {Path(path)}")
            
    allmetrics = np.array(allmetrics)
        
    # compute averages per seed --> (nummetrics, numseeds,)
    means_per_seed = allmetrics.mean(0).T
    
    means_over_seeds = means_per_seed.mean(1)
    stds_over_seeds = means_per_seed.std(1)
    
    higher_is_better = np.array([True if colname.higher_is_better else False for colname in colnames])
    max_over_seeds = allmetrics.max(1).mean(0)
    min_over_seeds = allmetrics.min(1).mean(0)
    best_over_seeds = np.where(higher_is_better, max_over_seeds, min_over_seeds)
    
    means_over_seeds_dict = dict(zip(colnames, list(means_over_seeds)))
    stds_over_seeds_dict = dict(zip(colnames, list(stds_over_seeds)))
    best_over_seeds_dict = dict(zip(colnames, list(best_over_seeds)))
        
    print(means_over_seeds_dict)
    print(stds_over_seeds_dict)
    print(best_over_seeds_dict)
    
    means_over_seeds_dict = {str(k): v for k, v in means_over_seeds_dict.items()}
    stds_over_seeds_dict = {str(k): v for k, v in stds_over_seeds_dict.items()}
    best_over_seeds_dict = {str(k): v for k, v in best_over_seeds_dict.items()}
    
    tosave = {     "means": means_over_seeds_dict, 
                   "stds": stds_over_seeds_dict, 
                   "best": best_over_seeds_dict, 
                #    "alldata": allmetrics, 
                #    "colnames": colnames
             } 
    
    with open(Path(path) / "evaluation_results_summary.json", "w") as f:
        json.dump(tosave, 
                  f, 
                  indent=4)
    print(json.dumps(tosave, indent=4))
    print(f"saved in {Path(path)}")
    
    
def tst_aesthetics():
    image = Image.open("lovely-cat-as-domestic-animal-view-pictures-182393057.jpg")
    
    aestheticspredictor = AestheticsPredictor()
    score = aestheticspredictor.run_image(image)
    
    print("aesthetic score:", score)
    
    
def run4(
        paths=[
            "experiments/controlnet_seg_ft_base/*",
        ],
        tightcrop=True,
        fill="none",
        device=0,
        #  clip_version="openai/clip-vit-base-patch32",
        clip_version="openai/clip-vit-large-patch14",
        sim=False,
    ):
    print("loading everything")
    if not sim:
        evalmodels = load_everything(clip_version=clip_version, device=device, tightcrop=tightcrop, fill=fill)
    
    totalcount = 0
    for i, path in enumerate(paths):
        assert path.startswith("/")
        subpaths = list(Path("/").glob(path[1:]))
        for j, subpath in enumerate(subpaths):
            if subpath.is_dir() and (subpath / "outbatches.pkl").exists():
                totalcount += 1
                print(f"Doing {subpath} ({i+1}/{len(paths)} path, {j+1}/{len(subpaths)} subpath) (total: {totalcount})")
                if not sim:
                    run3(path=subpath, tightcrop=tightcrop, fill=fill, device=device, clip_version=clip_version, evalmodels=evalmodels)
                
    
if __name__ == "__main__":
    # tst_aesthetics()
    # run3()
    fire.Fire(run4)