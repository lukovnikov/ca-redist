from copy import deepcopy
from datetime import timedelta
import json
import math
import pickle as pkl
from pathlib import Path
import random
import re
from typing import Any, Dict
import cv2
import einops
import fire
import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange, repeat
import os
import gc

from torch.nn.utils.rnn import pad_sequence
from torchvision.utils import make_grid

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.cldm import ControlLDM
from cldm.logger import ImageLogger, nested_to
from cldm.model import create_model, load_state_dict
from dataset import COCOPanopticDataset, COCODataLoader
from ldm.modules.attention import BasicTransformerBlock, default
from ldm.modules.diffusionmodules.util import torch_cat_nested
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import SeedSwitch, log_txt_as_img, seed_everything

from gradio_pww import _tokenize_annotated_prompt, create_tools, CustomTextConditioning as CTC_gradio_pww

_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


class CustomTextConditioning():
    def __init__(self, embs, layer_ids=None, token_ids=None, global_prompt_mask=None, global_bos_eos_mask=None):
        """
        embs:       (batsize, seqlen, embdim)
        layer_ids:  (batsize, seqlen) integers, with 0 for no-layer global tokens
        token_ids:  (batsize, seqlen) integers for tokens from tokenizer
        global_prompt_mask:  (batsize, seqlen) bool that is 1 where the global prompt is and 0 where the local regional prompts are
        global_bos_eos_mask: (batsize, seqlen) bool that is 1 where the global bos and eos tokens are and 0 elsewhere
        """
        self.embs = embs
        self.device = self.embs.device
        self.layer_ids = layer_ids
        self.token_ids = token_ids
        self.global_prompt_mask = global_prompt_mask
        self.global_bos_eos_mask = global_bos_eos_mask
        self.cross_attn_masks = None
        self.progress = None
        self.strength = 10
        self.threshold = None
        self.softness = 0.2
        self.controlonly = False
        self.controlledonly = False
        
    def flatten_inputs_for_gradient_checkpoint(self):
        flat_out = [self.embs]
        def recon_f(x:list):
            self.embs = x[0]
            return self
        return flat_out, recon_f
    
    def torch_cat_nested(self, other):
        # concatenate all torch tensors along batch dimension
        ret = deepcopy(self)
        batsize = self.embs.shape[0]
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == batsize:       # probably concatenatable tensor
                setattr(ret, k, torch_cat_nested(getattr(self, k), getattr(other, k)))
        # ret.embs = torch_cat_nested(self.embs, other.embs)
        # ret.layer_ids = torch_cat_nested(self.layer_ids, other.layer_ids)
        # ret.token_ids = torch_cat_nested(self.token_ids, other.token_ids)
        # ret.global_bos_eos_mask = torch_cat_nested(self.global_bos_eos_mask, other.global_bos_eos_mask)
        # ret.global_prompt_mask = torch_cat_nested(self.global_prompt_mask, other.global_prompt_mask)
        ret.cross_attn_masks = torch_cat_nested(self.cross_attn_masks, other.cross_attn_masks)
        # ret.progress = torch_cat_nested(self.progress, other.progress)
        return ret
    
    
class CustomCrossAttentionBase(nn.Module):
    
    @classmethod
    def from_base(cls, m):
        m.__class__ = cls
        m.init_extra()
        return m
    
    def init_extra(self):
        for p in self.get_trainable_parameters():
            p.train_param = True
    
    def weight_func(self, sim, context, sim_mask=None):
        with torch.no_grad():
            if sim_mask is None:
                sim_mask = context.captiontypes >= 0
                sim_mask = repeat(sim_mask, 'b j -> (b h) () j', h=sim.shape[0] // sim_mask.shape[0])
            simstd = torch.masked_select(sim, sim_mask).std()
            mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)
            ret = mask * simstd * context.strength
        return ret
    
    def get_trainable_parameters(self):
        return []    

    
class CustomCrossAttentionBaseline(CustomCrossAttentionBase):
    # Tries to emulate the basic setting where only the global prompt is available, and attention computation is not changed.

    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        if context is not None:
            context = context["c_crossattn"][0]
            contextembs = context.embs
        else:
            assert False        # this shouldn't be used as self-attention
            contextembs = x
            
        q = self.to_q(x)
        k = self.to_k(contextembs)
        v = self.to_v(contextembs)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            sim = einsum('b i d, b j d -> b i j', q, k)
        
        del q, k

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if context is not None:     # cross-attention
            sim = self.cross_attention_control(sim, context, numheads=h)
        
        # attention
        sim = (sim * self.scale).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 0
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        wf = self.weight_func(sim, context, sim_mask=mask)
        wf.masked_fill_(~context.global_prompt_mask[:, None, :], max_neg_value)
        
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        sim.masked_fill_(~mask, max_neg_value)
        sim = sim + wf
        return sim
    
    def get_trainable_parameters(self):
        params = list(self.to_q.parameters())
        params += list(self.to_k.parameters())
        return params
    
    
class CustomCrossAttentionLegacy(CustomCrossAttentionBaseline):
    def cross_attention_control(self, sim, context, numheads=None):
        return context.cross_attention_control(sim)
    

class CustomCrossAttentionDelegated(CustomCrossAttentionBaseline):      # delegated cross attention manipulation to the context object
    def cross_attention_control(self, sim, context, numheads=None):
        ret = context.cross_attention_control(sim)
        # TODO: below is for debugging
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        ret2 = deepcopy(sim.clone())
        ret2.masked_fill_(~mask, max_neg_value)
        
        if not torch.allclose(ret, ret2):
            print("not same")
        
        return ret
    

class CustomCrossAttentionBaselineBoth(CustomCrossAttentionBaseline):
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 0
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        sim.masked_fill_(~mask, max_neg_value)
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        wf = self.weight_func(sim, context)
        
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        sim = sim + wf
        return sim
    

class CustomCrossAttentionBaselineLocal(CustomCrossAttentionBaseline):
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        """ Takes the unscaled unnormalized attention scores computed by cross-attention module, returns adapted attention scores. """
        wf = self.weight_func(sim, context, sim_mask=mask)
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        wf.masked_fill_(~mask, max_neg_value)
        sim = sim + wf
        return sim
    
    
class CustomCrossAttentionBaselineLocalGlobalFallback(CustomCrossAttentionBaseline):
    """ Uses only local descriptions, unless there is none (all local ones are masked), then falls back to global description."""
    
    def cross_attention_control(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        
        # get mask that selects global as well as applicable local prompt
        wf = self.weight_func(sim, context, sim_mask=mask)

        # expand dimensions to match heads
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        # remove the global part from the mask
        wf.masked_fill_(~mask, 0)  # max_neg_value)
        # determine where to use global mask (where no local descriptions are available)
        useglobal = wf.sum(-1) == 0
        
        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        wfmaxes = wf.max(-1, keepdim=True)[0]
        wf = (wf >= (wfmaxes.clamp_min(1e-3) - 1e-4)).float()
        
        # get mask that selects only global tokens
        gmask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        gmask = repeat(gmask, 'b j -> (b h) () j', h=numheads)
        
        # update stimulation to attend to global tokens when no local tokens are available
        wf = wf + (useglobal[:, :, None] & gmask)
        
        sim.masked_fill_(wf == 0, max_neg_value)
        # sim = sim + wf
        return sim
    

class CustomCrossAttentionSepSwitch(CustomCrossAttentionBaseline):
    threshold = 0.2
    """ Uses only local descriptions, unless there is none (all local ones are masked), then falls back to global description."""
    
    def cross_attention_control(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2                # localmask
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        
        # get mask that selects global as well as applicable local prompt
        wf = self.weight_func(sim, context, sim_mask=mask)

        # expand dimensions to match heads
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        # remove the global part from the mask
        wf.masked_fill_(~mask, 0)  # max_neg_value)
        # determine where to use global mask (where no local descriptions are available)
        useglobal = wf.sum(-1) == 0
        
        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        wfmaxes = wf.max(-1, keepdim=True)[0]
        wf = (wf >= (wfmaxes.clamp_min(1e-3) - 1e-4)).float()
        
        # get mask that selects only global tokens
        gmask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        gmask = repeat(gmask, 'b j -> (b h) () j', h=numheads)
        
        # update stimulation to attend to global tokens when no local tokens are available
        lmask = wf + (useglobal[:, :, None] & gmask)
        
        prog = context.progress
        prog = prog[:, None].repeat(1, self.heads).view(-1)
        lorg = prog <= self.threshold
        
        mask = torch.where(lorg[:, None, None], lmask, gmask)
        
        sim.masked_fill_(mask == 0, max_neg_value)
        # sim = sim + wf
        return sim
    
    
class CustomCrossAttentionCAC(CustomCrossAttentionBaseline):
    
    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        if context is not None:
            context = context["c_crossattn"][0]
            contextembs = context.embs
        else:
            assert False        # this shouldn't be used as self-attention
            contextembs = x
            
        q = self.to_q(x)
        k = self.to_k(contextembs)
        v = self.to_v(contextembs)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            sim = einsum('b i d, b j d -> b i j', q, k)
        
        del q, k

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention
        sim = (sim * self.scale).softmax(dim=-1)
        
        if context is not None:     # cross-attention
            sim = self.cross_attention_control(sim, context, numheads=h)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def cross_attention_control(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        wf = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # expand dimensions to match heads
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        wfmaxes = wf.max(-1, keepdim=True)[0]
        wf = (wf >= (wfmaxes.clamp_min(1e-3) - 1e-4)).float()
        
        sim.masked_fill_(wf == 0, 0)
        # sim = sim + wf
        return sim
    

class CustomCrossAttentionDenseDiffusion(CustomCrossAttentionBaseline):
    # strength values: 0.5 (good) , 1.0 (maybe a bit too strong)
    
    def cross_attention_control(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        # globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        # max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        boostmask = mask * mask2        # selects only those tokens belonging to regional description corresponding to the region pixels
        # R is boostmask
        mpos = sim.max(-1, keepdims=True)[0] - sim
        mneg = sim - sim.min(-1, keepdims=True)[0]
        lambda_t = (1 - context.progress) ** 5
        # lambda_t.masked_fill_(context.progress > 0.3, 0)
        
        boostmask = boostmask[:, None].repeat(1, numheads, 1, 1)
        boostmask = boostmask.view(-1, boostmask.shape[-2], boostmask.shape[-1])
        lambda_t = lambda_t[:, None].repeat(1, numheads)
        lambda_t = lambda_t.view(-1)
        
        # effect of region size (S in paper equations)
        S = boostmask.sum(1, keepdims=True) / boostmask.shape[1]

        ret = sim + context.strength * lambda_t[:, None, None] * boostmask * mpos * (1 - S) - context.strength * lambda_t[:, None, None] * (1 - boostmask) * mneg * (1 - S)
        
        # # expand dimensions to match heads
        # mask = mask[:, None].repeat(1, numheads, 1, 1)
        # mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
        
        # a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        # weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        
        
        # ret = sim + boostmask * context.strength * sim.std()
        
        # # don't attend to other region-specific tokens or non-global prompt ones
        # ret.masked_fill_(mask == 0, max_neg_value)
        
        return ret
    
    
class CustomCrossAttentionPosattn(CustomCrossAttentionBaseline):
    
    def cross_attention_control(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask[:, :, :77] == 1)
        assert torch.all(globalmask[:, :, 77:] == 0)
        globalmask = globalmask[0:1, :, :]
        simscale = sim[:, :, :77].std(-1, keepdim=True)
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(globalmask == 0, max_neg_value)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        boostmask = mask * mask2        # selects only those tokens not belonging to any regional description
        
        # # expand dimensions to match heads
        # mask = mask[:, None].repeat(1, numheads, 1, 1)
        # mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
        
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        
        boostmask = boostmask * weight
        boostmask = boostmask[:, None].repeat(1, numheads, 1, 1)
        boostmask = boostmask.view(-1, boostmask.shape[-2], boostmask.shape[-1])
        
        ret = sim + boostmask * context.strength * simscale #sim.std()
        
        # # don't attend to other region-specific tokens or non-global prompt ones
        # ret.masked_fill_(mask == 0, max_neg_value)
        
        return ret
    
    
class CustomCrossAttentionPosattn2(CustomCrossAttentionBaseline):       
    # when attending from a region:   tokens belonging to other regions are completely ignored 
    #                                 and all their probability is transfered to the correct region's tokens
    # additional boosting of attention can be done according to a schedule, this boosting
    #    is added before normalizing, so tokens get boosted equally and everywhere, resulting in more uniform distributions
    
    def compute_attention_losses(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask == 1)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        boostmask = mask * mask2        # selects only those tokens not belonging to any regional description
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        boostmask = boostmask * weight
        boostmaskext = boostmask[:, None].repeat(1, numheads, 1, 1)
        boostmaskext = boostmaskext.view(-1, boostmaskext.shape[-2], boostmaskext.shape[-1])
        
        logprobs_vanilla = (sim * self.scale).log_softmax(-1)
        # compute objective to maximize the probability assigned to boosted tokens
        target = boostmaskext / boostmaskext.sum(-1, keepdim=True)
        loss = torch.nn.functional.kl_div(logprobs_vanilla, target, reduction="none")
        loss = loss.sum(-1).mean(-1).mean(-1)
        prob_boosted = (logprobs_vanilla.exp() * boostmaskext).sum(-1).mean(-1).mean(-1)
        
        # store them on the context object
        context.attention_losses.append(loss)
        return logprobs_vanilla.exp()
    
    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        if context is not None:
            context = context["c_crossattn"][0]
            contextembs = context.embs
        else:
            assert False        # this shouldn't be used as self-attention
            contextembs = x
            
        q = self.to_q(x)
        k = self.to_k(contextembs)
        v = self.to_v(contextembs)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            sim = einsum('b i d, b j d -> b i j', q, k)
        
        del q, k

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if context is not None:     # cross-attention
            if hasattr(context, "compute_attention_losses") and context.compute_attention_losses is True:
                sim = self.compute_attention_losses(sim, context, numheads=h)
            else:
                sim = self.cross_attention_control(sim, context, numheads=h)
        
        # attention
        # sim = (sim * self.scale).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def cross_attention_control(self, sim, context, numheads=None):
        assert torch.all(context.progress == context.progress[0])
        if hasattr(context, "threshold_lot") and torch.all(context.progress[0] < context.threshold_lot):
            sim = CustomCrossAttentionSepSwitch.cross_attention_control(self, sim, context, numheads=numheads)
            return (sim * self.scale).softmax(-1)
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask[:, :, :77] == 1)
        assert torch.all(globalmask[:, :, 77:] == 0)
        globalmask = globalmask[0:1, :, :]
        simscale = sim[:, :, :77].std(-1, keepdim=True)
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(globalmask == 0, max_neg_value)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        maskext = mask[:, None].repeat(1, numheads, 1, 1)
        maskext = maskext.view(-1, maskext.shape[-2], maskext.shape[-1])
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        mask2ext = mask2[:, None].repeat(1, numheads, 1, 1)
        mask2ext = mask2ext.view(-1, mask2ext.shape[-2], mask2ext.shape[-1])
        
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        boostmask = mask * mask2        # selects only those tokens not belonging to any regional description
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        boostmask = boostmask * weight
        boostmaskext = boostmask[:, None].repeat(1, numheads, 1, 1)
        boostmaskext = boostmaskext.view(-1, boostmaskext.shape[-2], boostmaskext.shape[-1])
        
        sim = sim + boostmaskext * context.strength * simscale
        
        # do mixture of two distributions: one over region tokens, and one over non-region tokens
        sim = sim * self.scale
        probs_vanilla = sim.softmax(-1)
        prob_mass_of_regions = (probs_vanilla * mask2ext).sum(-1, keepdims=True)
        
        probs_inside_regions = sim.masked_fill(maskext * mask2ext == 0, max_neg_value).softmax(-1)
        probs_outside_regions = sim.masked_fill(mask2ext == 1, max_neg_value).softmax(-1)
        
        outprobs = prob_mass_of_regions * probs_inside_regions + (1 - prob_mass_of_regions) * probs_outside_regions
        
        return outprobs
    
    
    
class CustomCrossAttentionPosattn3(CustomCrossAttentionPosattn2):
    
    def cross_attention_control(self, sim, context, numheads=None):
        assert torch.all(context.progress == context.progress[0])
        if hasattr(context, "threshold_lot") and torch.all(context.progress[0] < context.threshold_lot):
            sim = CustomCrossAttentionSepSwitch.cross_attention_control(self, sim, context, numheads=numheads)
            return (sim * self.scale).softmax(-1)
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask[:, :, :77] == 1)
        assert torch.all(globalmask[:, :, 77:] == 0)
        globalmask = globalmask[0:1, :, :]
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(globalmask == 0, max_neg_value)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        maskext = mask[:, None].repeat(1, numheads, 1, 1)
        maskext = maskext.view(-1, maskext.shape[-2], maskext.shape[-1])
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        mask2ext = mask2[:, None].repeat(1, numheads, 1, 1)
        mask2ext = mask2ext.view(-1, mask2ext.shape[-2], mask2ext.shape[-1])
        
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        boostmask = mask * mask2        # selects only those tokens not belonging to any regional description
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        boostmask = boostmask * weight
        boostmaskext = boostmask[:, None].repeat(1, numheads, 1, 1)
        boostmaskext = boostmaskext.view(-1, boostmaskext.shape[-2], boostmaskext.shape[-1])
        
        mpos = sim.max(-1, keepdims=True)[0] - sim
        # mneg = sim - sim.min(-1, keepdims=True)[0]
        #S = boostmaskext.sum(1, keepdims=True) / boostmaskext.shape[1]
        
        sim = sim + boostmaskext * context.strength * mpos   #* (1 - S)
        
        # do mixture of two distributions: one over region tokens, and one over non-region tokens
        sim = sim * self.scale
        probs_vanilla = sim.softmax(-1)
        prob_mass_of_regions = (probs_vanilla * mask2ext).sum(-1, keepdims=True)
        
        probs_inside_regions = sim.masked_fill(maskext * mask2ext == 0, max_neg_value).softmax(-1)
        probs_outside_regions = sim.masked_fill(mask2ext == 1, max_neg_value).softmax(-1)
        
        outprobs = prob_mass_of_regions * probs_inside_regions + (1 - prob_mass_of_regions) * probs_outside_regions
        
        return outprobs
    

class CustomCrossAttentionPosattn4(CustomCrossAttentionPosattn3):
    
    def cross_attention_control(self, sim, context, numheads=None):
        assert torch.all(context.progress == context.progress[0])
        if hasattr(context, "threshold_lot") and torch.all(context.progress[0] < context.threshold_lot):
            sim = CustomCrossAttentionSepSwitch.cross_attention_control(self, sim, context, numheads=numheads)
            return (sim * self.scale).softmax(-1)
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask[:, :, :77] == 1)
        assert torch.all(globalmask[:, :, 77:] == 0)
        globalmask = globalmask[0:1, :, :]
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(globalmask == 0, max_neg_value)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        maskext = mask[:, None].repeat(1, numheads, 1, 1)
        maskext = maskext.view(-1, maskext.shape[-2], maskext.shape[-1])
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        mask2ext = mask2[:, None].repeat(1, numheads, 1, 1)
        mask2ext = mask2ext.view(-1, mask2ext.shape[-2], mask2ext.shape[-1])
        
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        boostmask = mask * mask2        # selects only those tokens not belonging to any regional description
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        boostmask = boostmask * weight
        boostmaskext = boostmask[:, None].repeat(1, numheads, 1, 1)
        boostmaskext = boostmaskext.view(-1, boostmaskext.shape[-2], boostmaskext.shape[-1])
        
        mpos = sim.max(-1, keepdims=True)[0] - sim
        # mneg = sim - sim.min(-1, keepdims=True)[0]
        S = boostmaskext.sum(1, keepdims=True) / boostmaskext.shape[1]
        
        sim = sim + boostmaskext * context.strength * mpos * (1 - S)
        
        # do mixture of two distributions: one over region tokens, and one over non-region tokens
        sim = sim * self.scale
        probs_vanilla = sim.softmax(-1)
        prob_mass_of_regions = (probs_vanilla * mask2ext).sum(-1, keepdims=True)
        
        probs_inside_regions = sim.masked_fill(maskext * mask2ext == 0, max_neg_value).softmax(-1)
        probs_outside_regions = sim.masked_fill(mask2ext == 1, max_neg_value).softmax(-1)
        
        outprobs = prob_mass_of_regions * probs_inside_regions + (1 - prob_mass_of_regions) * probs_outside_regions
        
        return outprobs
    

class CustomCrossAttentionPosattn5(CustomCrossAttentionPosattn2):
    # when attending from a region:   tokens belonging to other regions are completely ignored 
    #                                 and all their probability is transfered to the correct region's tokens
    # additional boosting of attention can be done according to a schedule, this boosting
    #    multiplies the fraction of region-specific attention with the boost factor, 
    #    so heads and layers where a lot of probability was spent on region-specific tokens see a larger increase
    #    while heads that did not attend to any regions see no increase (unlike addition-based boosting)
    
    def cross_attention_control(self, sim, context, numheads=None):
        assert torch.all(context.progress == context.progress[0])
        if hasattr(context, "threshold_lot") and torch.all(context.progress[0] < context.threshold_lot):
            sim = CustomCrossAttentionSepSwitch.cross_attention_control(self, sim, context, numheads=numheads)
            return (sim * self.scale).softmax(-1)
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask[:, :, :77] == 1)
        assert torch.all(globalmask[:, :, 77:] == 0)
        globalmask = globalmask[0:1, :, :]
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(globalmask == 0, max_neg_value)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        maskext = mask[:, None].repeat(1, numheads, 1, 1)
        maskext = maskext.view(-1, maskext.shape[-2], maskext.shape[-1])
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        mask2ext = mask2[:, None].repeat(1, numheads, 1, 1)
        mask2ext = mask2ext.view(-1, mask2ext.shape[-2], mask2ext.shape[-1])
        
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        weightext = weight[:, None].repeat(1, numheads, 1, 1)
        weightext = weightext.view(-1, weightext.shape[-2], weightext.shape[-1])
        
        # do mixture of two distributions: one over region tokens, and one over non-region tokens
        sim = sim * self.scale
        probs_vanilla = sim.softmax(-1)
        prob_mass_of_regions = (probs_vanilla * mask2ext).sum(-1, keepdims=True)
        prob_mass_of_regions = prob_mass_of_regions * (1 + context.strength * weightext) 
        prob_mass_of_regions.clamp_(0, 1)
        
        probs_inside_regions = sim.masked_fill(maskext * mask2ext == 0, max_neg_value).softmax(-1)
        probs_outside_regions = sim.masked_fill(mask2ext == 1, max_neg_value).softmax(-1)
        
        outprobs = prob_mass_of_regions * probs_inside_regions + (1 - prob_mass_of_regions) * probs_outside_regions
        
        return outprobs
    
    
class CustomCrossAttentionPosattn5a(CustomCrossAttentionPosattn2):  # "a" for additive: we boost regional part using addition instead of multiplication
    # when attending from a region:   tokens belonging to other regions are completely ignored 
    #                                 and all their probability is transfered to the correct region's tokens
    # additional boosting of attention can be done according to a schedule, this boosting
    #    multiplies the fraction of region-specific attention with the boost factor, 
    #    so heads and layers where a lot of probability was spent on region-specific tokens see a larger increase
    #    while heads that did not attend to any regions see no increase (unlike addition-based boosting)
    
    def cross_attention_control(self, sim, context, numheads=None):
        assert torch.all(context.progress == context.progress[0])
        if hasattr(context, "threshold_lot") and torch.all(context.progress[0] < context.threshold_lot):
            sim = CustomCrossAttentionSepSwitch.cross_attention_control(self, sim, context, numheads=numheads)
            return (sim * self.scale).softmax(-1)
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask[:, :, :77] == 1)
        assert torch.all(globalmask[:, :, 77:] == 0)
        globalmask = globalmask[0:1, :, :]
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(globalmask == 0, max_neg_value)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        maskext = mask[:, None].repeat(1, numheads, 1, 1)
        maskext = maskext.view(-1, maskext.shape[-2], maskext.shape[-1])
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        mask2ext = mask2[:, None].repeat(1, numheads, 1, 1)
        mask2ext = mask2ext.view(-1, mask2ext.shape[-2], mask2ext.shape[-1])
        
        if not isinstance(context.threshold, (list, tuple)):
            context.threshold = [context.threshold, context.threshold]
        if not isinstance(context.softness, (list, tuple)):
            context.softness = [context.softness, context.softness]
            
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        a, b = max(0, context.threshold[0] - context.softness[0] / 2), min(context.threshold[0] + context.softness[0] / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        weightext = weight[:, None].repeat(1, numheads, 1, 1)
        weightext = weightext.view(-1, weightext.shape[-2], weightext.shape[-1])
        
        a2, b2 = max(0, context.threshold[1] - context.softness[1] / 2), min(context.threshold[1] + context.softness[1] / 2, 1)
        weight2 = 1 - _threshold_f(context.progress, a2, b2)[:, None, None]
        weightext2 = weight2[:, None].repeat(1, numheads, 1, 1)
        weightext2 = weightext2.view(-1, weightext2.shape[-2], weightext2.shape[-1])
        
        # do mixture of two distributions: one over region tokens, and one over non-region tokens
        sim = sim * self.scale
        probs_vanilla = sim.softmax(-1)
        prob_mass_of_regions = (probs_vanilla * mask2ext).sum(-1, keepdims=True)
        prob_mass_of_regions = prob_mass_of_regions * (1 + context.strength[0] * weightext)
        prob_mass_of_regions.clamp_(0, 1)
        
        boostmaskext = maskext * mask2ext
        S = boostmaskext.sum(1, keepdims=True) / boostmaskext.shape[1]
        surfacemod = (((1 - S) * boostmaskext).sum(-1, keepdim=True) / boostmaskext.sum(-1, keepdim=True).clamp_min(1e-6))
        prob_mass_of_regions = prob_mass_of_regions + (1 - prob_mass_of_regions) * (context.strength[1] * weightext2) * surfacemod
        
        probs_inside_regions = sim.masked_fill(maskext * mask2ext == 0, max_neg_value).softmax(-1)
        probs_outside_regions = sim.masked_fill(mask2ext == 1, max_neg_value).softmax(-1)
        
        outprobs = prob_mass_of_regions * probs_inside_regions + (1 - prob_mass_of_regions) * probs_outside_regions
        
        return outprobs
    
    
class CustomCrossAttentionPosattn5b(CustomCrossAttentionPosattn5):
    # adds rebalancing of tokens within region descriptions: HOW?
    #    - makes sure that all words are attended to equally, when all pixels in the region considered
    #    - makes sure that relative amounts of attention between tokens are still respected, relative to other pairs
    
    def cross_attention_control(self, sim, context, numheads=None):
        assert torch.all(context.progress == context.progress[0])
        if hasattr(context, "threshold_lot") and torch.all(context.progress[0] < context.threshold_lot):
            sim = CustomCrossAttentionSepSwitch.cross_attention_control(self, sim, context, numheads=numheads)
            return (sim * self.scale).softmax(-1)
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask[:, :, :77] == 1)
        assert torch.all(globalmask[:, :, 77:] == 0)
        globalmask = globalmask[0:1, :, :]
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(globalmask == 0, max_neg_value)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        maskext = mask[:, None].repeat(1, numheads, 1, 1)
        maskext = maskext.view(-1, maskext.shape[-2], maskext.shape[-1])
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        mask2ext = mask2[:, None].repeat(1, numheads, 1, 1)
        mask2ext = mask2ext.view(-1, mask2ext.shape[-2], mask2ext.shape[-1])
        
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        weightext = weight[:, None].repeat(1, numheads, 1, 1)
        weightext = weightext.view(-1, weightext.shape[-2], weightext.shape[-1])
        
        # do mixture of two distributions: one over region tokens, and one over non-region tokens
        sim = sim * self.scale
        probs_vanilla = sim.softmax(-1)
        prob_mass_of_regions = (probs_vanilla * mask2ext).sum(-1, keepdims=True)
        prob_mass_of_regions = prob_mass_of_regions * (1 + context.strength * weightext) 
        prob_mass_of_regions.clamp_(0, 1)
        
        boostmask = maskext * mask2ext
        
        # ONLY FOR CHECKING HOW PROBMASSES CHANGED:
        # probs_inside_regions_A = sim.masked_fill(boostmask == 0, max_neg_value).softmax(-1)
        # probmasses_A = probs_inside_regions_A.sum(1, keepdim=True) * mask2ext
        
        sim_masked = sim.masked_fill(boostmask == 0, 0)
        sim_mean_per_token = sim_masked.sum(-2, keepdim=True) \
            / (boostmask).sum(-2, keepdim=True).clamp_min(1e-6)
        sim_diff_per_token = sim_masked - sim_mean_per_token
        sim_mean_sum = (sim_mean_per_token * boostmask).sum(-1, keepdim=True)
        numtokens_per_pixel = boostmask.sum(-1, keepdims=True)
        sim_mean_target = sim_mean_sum / numtokens_per_pixel.clamp_min(1e-6)
        newsim_masked = sim_mean_target + sim_diff_per_token
        
        # ONLY FOR CHECKING HOW PROBMASSES CHANGED:
        # probs_inside_regions_B = newsim_masked.masked_fill(boostmask == 0, max_neg_value).softmax(-1)
        # probmasses_B = probs_inside_regions_B.sum(1, keepdim=True) * mask2ext
        
        probs_inside_regions = newsim_masked.masked_fill(boostmask == 0, max_neg_value).softmax(-1)
        probs_outside_regions = sim.masked_fill(mask2ext == 1, max_neg_value).softmax(-1)
        
        outprobs = prob_mass_of_regions * probs_inside_regions + (1 - prob_mass_of_regions) * probs_outside_regions
        
        # probmasses_out = outprobs.sum(1, keepdim=True) #* mask2ext
        
        return outprobs
    
    
class CustomCrossAttentionPosattn5c(CustomCrossAttentionPosattn5):
    # adds rebalancing of tokens within region descriptions: HOW?
    #    - makes sure that all words are attended to equally, when all pixels in the region considered
    #    - makes sure that relative amounts of attention between tokens are still respected, relative to other pairs
    
    def cross_attention_control(self, sim, context, numheads=None):
        assert torch.all(context.progress == context.progress[0])
        if hasattr(context, "threshold_lot") and torch.all(context.progress[0] < context.threshold_lot):
            sim = CustomCrossAttentionSepSwitch.cross_attention_control(self, sim, context, numheads=numheads)
            return (sim * self.scale).softmax(-1)
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask[:, :, :77] == 1)
        assert torch.all(globalmask[:, :, 77:] == 0)
        globalmask = globalmask[0:1, :, :]
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(globalmask == 0, max_neg_value)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        maskext = mask[:, None].repeat(1, numheads, 1, 1)
        maskext = maskext.view(-1, maskext.shape[-2], maskext.shape[-1])
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        mask2ext = mask2[:, None].repeat(1, numheads, 1, 1)
        mask2ext = mask2ext.view(-1, mask2ext.shape[-2], mask2ext.shape[-1])
        
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        weightext = weight[:, None].repeat(1, numheads, 1, 1)
        weightext = weightext.view(-1, weightext.shape[-2], weightext.shape[-1])
        
        # do mixture of two distributions: one over region tokens, and one over non-region tokens
        sim = sim * self.scale
        probs_vanilla = sim.softmax(-1)
        _prob_mass_of_regions = (probs_vanilla * mask2ext).sum(-1, keepdims=True)        # per pixel
        prob_mass_per_head = _prob_mass_of_regions.max(-2, keepdims=True)[0]
        prob_mass_of_regions = (weightext * 0.5) * prob_mass_per_head + (1 - weightext * 0.5) * _prob_mass_of_regions
        prob_mass_of_regions = prob_mass_of_regions * (1 + context.strength * weightext) 
        prob_mass_of_regions.clamp_(0, 1)
        
        probs_inside_regions = sim.masked_fill(maskext * mask2ext == 0, max_neg_value).softmax(-1)
        probs_outside_regions = sim.masked_fill(mask2ext == 1, max_neg_value).softmax(-1)
        
        outprobs = prob_mass_of_regions * probs_inside_regions + (1 - prob_mass_of_regions) * probs_outside_regions
        return outprobs


class CustomCrossAttentionPosattn5u(CustomCrossAttentionPosattn5):   
    # adds rebalancing of tokens within region descriptions: forces uniform attention distribution within a region description
    
    def cross_attention_control(self, sim, context, numheads=None):
        assert torch.all(context.progress == context.progress[0])
        if hasattr(context, "threshold_lot") and torch.all(context.progress[0] < context.threshold_lot):
            sim = CustomCrossAttentionSepSwitch.cross_attention_control(self, sim, context, numheads=numheads)
            return (sim * self.scale).softmax(-1)
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask[:, :, :77] == 1)
        assert torch.all(globalmask[:, :, 77:] == 0)
        globalmask = globalmask[0:1, :, :]
        max_neg_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(globalmask == 0, max_neg_value)
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        maskext = mask[:, None].repeat(1, numheads, 1, 1)
        maskext = maskext.view(-1, maskext.shape[-2], maskext.shape[-1])
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        mask2ext = mask2[:, None].repeat(1, numheads, 1, 1)
        mask2ext = mask2ext.view(-1, mask2ext.shape[-2], mask2ext.shape[-1])
        
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        weightext = weight[:, None].repeat(1, numheads, 1, 1)
        weightext = weightext.view(-1, weightext.shape[-2], weightext.shape[-1])
        
        # do mixture of two distributions: one over region tokens, and one over non-region tokens
        sim = sim * self.scale
        probs_vanilla = sim.softmax(-1)
        prob_mass_of_regions = (probs_vanilla * mask2ext).sum(-1, keepdims=True)
        prob_mass_of_regions = prob_mass_of_regions * (1 + context.strength * weightext) 
        prob_mass_of_regions.clamp_(0, 1)
        
        probs_inside_regions = torch.zeros_like(sim).masked_fill(maskext * mask2ext == 0, max_neg_value).softmax(-1)
        probs_outside_regions = sim.masked_fill(mask2ext == 1, max_neg_value).softmax(-1)
        
        outprobs = prob_mass_of_regions * probs_inside_regions + (1 - prob_mass_of_regions) * probs_outside_regions
        
        return outprobs
    

    
class SimpleScheduleWeights(torch.nn.Module):
    def __init__(self, numheads=8, numsteps=100):
        super().__init__()
        self.numheads = numheads
        self.numsteps = numsteps
        self.param = torch.nn.Parameter(torch.randn(self.numsteps, self.numheads) * 1e-3)
        
    def forward(self, progress):        # progress is float
        # discretize to number of steps:
        steps = torch.round(progress * self.numsteps).long().clamp(0, self.numsteps - 1)
        headweights = torch.sigmoid(self.param[steps])      # initially, head weights are around 0.5, so halving the strength
        return headweights
    
    
class CustomCrossAttentionPosattnOptimized(CustomCrossAttentionBaseline):
    def init_extra(self):
        self.scheduleweights = SimpleScheduleWeights(numheads=self.heads)
        for p in self.get_trainable_parameters():
            p.train_param = True
        
    def get_trainable_parameters(self):
        params = list(self.scheduleweights.parameters())
        return params
    
    def cross_attention_control(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        boostmask = mask * mask2        # selects only those tokens not belonging to any regional description
        
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
    
        headweights = self.scheduleweights(context.progress)
        
        boostmask = boostmask * weight
        boostmask = boostmask[:, None].repeat(1, numheads, 1, 1)
        boostmask = boostmask * headweights[:, :, None, None]
        boostmask = boostmask.view(-1, boostmask.shape[-2], boostmask.shape[-1])
        
        ret = sim + boostmask * context.strength * sim.std()
        
        # don't attend to other region-specific tokens or non-global prompt ones
        # expand dimensions to match heads
        mask = mask[:, None].repeat(1, numheads, 1, 1)
        mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
        ret.masked_fill_(mask == 0, max_neg_value)
        
        return ret
    
    
class CustomCrossAttentionPosattn2Optimized(CustomCrossAttentionPosattnOptimized, CustomCrossAttentionPosattn2):
    
    def cross_attention_control(self, sim, context, numheads=None):
        # compute mask that ignores everything except the local descriptions
        globalmask = context.global_prompt_mask[:, None].to(sim.dtype)
        assert torch.all(globalmask == 1)
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.cross_attn_masks[sim.shape[1]].to(sim.dtype)

        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        maskmaxes = mask.max(-1, keepdim=True)[0]
        mask = (mask >= (maskmaxes.clamp_min(1e-3) - 1e-4)).float()
        maskext = mask[:, None].repeat(1, numheads, 1, 1)
        maskext = maskext.view(-1, maskext.shape[-2], maskext.shape[-1])
        
        mask2 = (context.layer_ids[:, None] != 0).to(sim.dtype).to(sim.device)
        mask2ext = mask2[:, None].repeat(1, numheads, 1, 1)
        mask2ext = mask2ext.view(-1, mask2ext.shape[-2], mask2ext.shape[-1])
        
        
        # boostmask should contain only tokens that are local: intersection of "mask" and where layer_ids are nonzero
        boostmask = mask * mask2        # selects only those tokens not belonging to any regional description
        a, b = max(0, context.threshold - context.softness / 2), min(context.threshold + context.softness / 2, 1)
        weight = 1 - _threshold_f(context.progress, a, b)[:, None, None]
        boostmask = boostmask * weight
        boostmaskext = boostmask[:, None].repeat(1, numheads, 1, 1)
        
        headweights = self.scheduleweights(context.progress)
        boostmaskext = boostmaskext * headweights[:, :, None, None]
        
        boostmaskext = boostmaskext.view(-1, boostmaskext.shape[-2], boostmaskext.shape[-1])
        
        sim = sim + boostmaskext * context.strength * sim.std()
        
        # do mixture of two distributions: one over region tokens, and one over non-region tokens
        sim = sim * self.scale
        probs_vanilla = sim.softmax(-1)
        prob_mass_of_regions = (probs_vanilla * mask2ext).sum(-1, keepdims=True)
        
        probs_inside_regions = sim.masked_fill(maskext * mask2ext == 0, max_neg_value).softmax(-1)
        probs_outside_regions = sim.masked_fill(mask2ext == 1, max_neg_value).softmax(-1)
        
        outprobs = prob_mass_of_regions * probs_inside_regions + (1 - prob_mass_of_regions) * probs_outside_regions
        
        return outprobs
    
    
def _threshold_f(p, a, b=1): # transitions from 0 at p=a to 1 at p=b using sinusoid curve
    threshold = (a, b)
    b = max(threshold)
    b = min(b, 1)
    a = min(threshold)
    a = min(b, a)
    a = max(a, 0)
    b = max(a, b)
    
    if not isinstance(p, torch.Tensor) and p.dim() == 1 and len(p) > 1:
        print("Assertion failed")
        print(p)
        assert False
    weight = torch.zeros_like(p)
    weight.masked_fill_(p >= b, 1)
    midpoint = (b - a) / 2 + a
    midweight = (torch.sin(math.pi * (p - midpoint) / (b - a)) + 1) * 0.5
    weight = torch.where((p < b) & (p > a), midweight, weight)
    
    return weight
    
    
class CustomCrossAttentionBaselineGlobal(CustomCrossAttentionBaseline):
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        sim.masked_fill_(~mask, max_neg_value)
        return sim
    
    
class DoublecrossBasicTransformerBlock(BasicTransformerBlock):
    threshold = 0.2 
    
    @classmethod
    def convert(cls, m):
        m.__class__ = cls
        m.init_extra()
        return m
    
    def init_extra(self):
        self.attn2l = deepcopy(self.attn2)
        self.attn2.__class__ = CustomCrossAttentionBaselineGlobal
        self.attn2l.__class__ = CustomCrossAttentionBaselineLocalGlobalFallback
        
        self.register_buffer("manual_gate", torch.tensor([1.]))
        self.learned_gate = torch.nn.Parameter(torch.randn(1,) * 1e-2)      # TODO: per-head learned gate
        
        self.norm2l = deepcopy(self.norm2)
        
        for p in self.get_trainable_parameters():
            p.train_param = True
        
    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        if self.training:
            x = self.manual_gate * torch.tanh(self.learned_gate) * self.attn2l(self.norm2l(x), context=context) + x
        else:
            threshold_gate = (context["c_crossattn"][0].progress < self.threshold).float()
            if torch.any(threshold_gate):
                x = threshold_gate[:, None, None] * self.manual_gate * \
                    torch.tanh(self.learned_gate) * self.attn2l(self.norm2l(x), context=context) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def get_trainable_parameters(self):
        params = list(self.attn2l.parameters())
        params += list(self.norm2l.parameters())
        params += [self.learned_gate]
        return params   
    
    
class TokenTypeEmbedding(torch.nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.emb = torch.nn.Embedding(5, embdim)
        self.merge = torch.nn.Sequential(
            torch.nn.Linear(embdim, embdim//2),
            torch.nn.GELU(),
            torch.nn.Linear(embdim//2, embdim)
        )
        self.gateA = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        self.gateB = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        
    def forward(self, tokentypes, contextemb):
        tokentypeemb = self.emb(tokentypes.clamp_min(0))
        ret = self.merge(contextemb + tokentypeemb)
        ret = tokentypeemb * self.gateA + ret * self.gateB
        return ret
    
    
class ProgressEmbedding(torch.nn.Module):
    def __init__(self, embdim) -> None:
        super().__init__()
        self.progress_emb = torch.nn.Sequential(
            torch.nn.Linear(1, embdim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(embdim//2, embdim)
        )
        self.merge = torch.nn.Sequential(
            torch.nn.Linear(embdim, embdim//2),
            torch.nn.GELU(),
            torch.nn.Linear(embdim//2, embdim)
        )
        self.gateA = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        self.gateB = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        
    def forward(self, progress, queries):
        progressemb = self.progress_emb(progress)
        ret = self.merge(progressemb + queries)
        ret = progressemb * self.gateA + ret * self.gateB
        return ret
    
    
class CustomCrossAttentionExt(CustomCrossAttentionBase):
    # DONE: add model extension to be able to tell where is global and local parts of the prompt
        
    def init_extra(self):
        # conditioning on token type (global BOS, global or local)
        self.token_type_emb = TokenTypeEmbedding(self.to_k.in_features)
        # conditioning on progress (0..1)
        self.progress_emb = ProgressEmbedding(self.to_q.in_features)
        
        for p in self.get_trainable_parameters():
            p.train_param = True

    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        if context is not None:
            context = context["c_crossattn"][0]
            contextembs = context.embs
        else:
            assert False        # this shouldn't be used as self-attention
            contextembs = x
            
        typeemb = self.token_type_emb(context.captiontypes, contextembs)
        progressemb = self.progress_emb(context.progress[:, None, None], x)

        q = self.to_q(x + progressemb)
        k = self.to_k(contextembs + typeemb)
        v = self.to_v(contextembs)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            sim = einsum('b i d, b j d -> b i j', q, k)
        
        del q, k

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if context is not None:     # cross-attention
            sim = self.cross_attention_control(sim, context, numheads=h)
        
        # attention
        sim = (sim * self.scale).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        
        # get mask that selects global as well as applicable local prompt
        wf = self.weight_func(sim, context, sim_mask=mask)

        # expand dimensions to match heads
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        # remove the global part from the mask
        wf.masked_fill_(~mask, 0)  # max_neg_value)
        
        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        wfmaxes = wf.max(-1, keepdim=True)[0]
        wf = (wf >= (wfmaxes.clamp_min(1e-3) - 1e-4))
        
        # get mask that selects only global tokens
        gmask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        gmask = repeat(gmask, 'b j -> (b h) () j', h=numheads)
        
        # update stimulation to attend to global tokens when no local tokens are available
        lmask = wf | gmask
        
        sim.masked_fill_(~lmask, max_neg_value)
        return sim
        
    
    def get_trainable_parameters(self):
        params = list(self.to_q.parameters())
        params += list(self.to_k.parameters())
        params += list(self.token_type_emb.parameters())
        params += list(self.progress_emb.parameters())
        return params
    
    
class CustomCrossAttentionExt2(CustomCrossAttentionBase):
    # DONE: add model extension to be able to tell where is global and local parts of the prompt
        
    def init_extra(self):
        # conditioning on token type (global BOS, global or local)
        self.token_type_emb = TokenTypeEmbedding(self.to_k.in_features)
        
        for p in self.get_trainable_parameters():
            p.train_param = True

    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        if context is not None:
            context = context["c_crossattn"][0]
            contextembs = context.embs
        else:
            assert False        # this shouldn't be used as self-attention
            contextembs = x
            
        typeemb = self.token_type_emb(context.captiontypes, contextembs)

        q = self.to_q(x)
        k = self.to_k(contextembs + typeemb)
        v = self.to_v(contextembs)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            sim = einsum('b i d, b j d -> b i j', q, k)
        
        del q, k

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if context is not None:     # cross-attention
            sim = self.cross_attention_control(sim, context, numheads=h)
        
        # attention
        sim = (sim * self.scale).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        
        # get mask that selects global as well as applicable local prompt
        wf = self.weight_func(sim, context, sim_mask=mask)

        # expand dimensions to match heads
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        # remove the global part from the mask
        wf.masked_fill_(~mask, 0)  # max_neg_value)
        
        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        wfmaxes = wf.max(-1, keepdim=True)[0]
        wf = (wf >= (wfmaxes.clamp_min(1e-3) - 1e-4))
        
        # get mask that selects only global tokens
        gmask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        gmask = repeat(gmask, 'b j -> (b h) () j', h=numheads)
        
        # update stimulation to attend to global tokens when no local tokens are available
        lmask = wf | gmask
        
        sim.masked_fill_(~lmask, max_neg_value)
        return sim
    
    def get_trainable_parameters(self):
        params = list(self.to_q.parameters())
        params += list(self.to_k.parameters())
        params += list(self.token_type_emb.parameters())
        return params
    
        
class TokenTypeEmbeddingMinimal(torch.nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.emb = torch.nn.Embedding(10, embdim)
        
    def forward(self, tokentypes):
        tokentypeemb = self.emb(tokentypes.clamp_min(0))
        return tokentypeemb
    
    
class ProgressEmbeddingMinimal(torch.nn.Module):
    def __init__(self, embdim) -> None:
        super().__init__()
        self.progress_emb = torch.nn.Sequential(
            torch.nn.Linear(1, embdim),
            torch.nn.ReLU(),
            torch.nn.Linear(embdim, embdim)
        )
        # self.gateA = torch.nn.Parameter(torch.randn(embdim) * 1e-3)
        
    def forward(self, progress):
        progressemb = self.progress_emb(progress)
        return progressemb
    
    
class DiscretizedProgressEmbed(torch.nn.Module):
    def __init__(self, embdim, embeds=50, steps=1000) -> None:
        super().__init__()
        self.embdim, self.embeds, self.steps = embdim, embeds, steps
        self.emb1 = torch.nn.Embedding(embeds+1, embdim)
        self.emb2 = torch.nn.Embedding(steps // embeds, embdim)
        
    def forward(self, x):
        # if torch.any(x == 1.):
        #     print("x contains a 1")
        xstep = (x * self.steps).round().long().clamp_max(self.steps-1)
        x1 = torch.div(xstep, (self.steps // self.embeds) , rounding_mode="floor")
        x2 = xstep % (self.steps // self.embeds)
        emb1 = self.emb1(x1)
        emb2 = self.emb2(x2)
        return emb1 + emb2
    
    
class ProgressClassifier(torch.nn.Module):      # classifies whether to use global prompt or local prompt for every head given progress
    INITBIAS = 0
    def __init__(self, embdim=512, numheads=8) -> None:
        super().__init__()
        self.embdim, self.numheads, self.numclasses = embdim, numheads, 2
        self.net = torch.nn.Sequential(
            DiscretizedProgressEmbed(embdim),
            torch.nn.GELU(),
            torch.nn.Linear(embdim, embdim),
            torch.nn.GELU(),
            torch.nn.Linear(embdim, self.numclasses * numheads)
        )
        finalbias = self.net[-1].bias
        classbias = torch.tensor([0, self.INITBIAS]).repeat(finalbias.shape[0]//2)
        finalbias.data += classbias
        
    def forward(self, progress):
        out = self.net(progress)        # maps (batsize, 1) to (batsize, numclases * numheads)
        probs = out.view(out.shape[0], self.numheads, self.numclasses).softmax(-1)
        return probs
    
    
class CustomCrossAttentionMinimal(CustomCrossAttentionExt):
    # Minimal cross attention: computes scores based on content independently from scores based on progress and region
    
    def init_extra(self):
        self.progressclassifier = ProgressClassifier(numheads=self.heads)
        for p in self.get_trainable_parameters():
            p.train_param = True
        
    def get_trainable_parameters(self):
        params = list(self.progressclassifier.parameters())
        return params
        
    def forward(self, x, context=None, mask=None):
        h = self.heads
        
        if context is not None:
            context = context["c_crossattn"][0]
            contextembs = context.embs
        else:
            assert False        # this shouldn't be used as self-attention
            contextembs = x
            
        q = self.to_q(x)
        k = self.to_k(contextembs)
        v = self.to_v(contextembs)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            sim = einsum('b i d, b j d -> b i j', q, k)
        
        del q, k

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if context is not None:     # cross-attention
            sim = self.cross_attention_control(sim, context, numheads=h)
        else:
            sim = (sim * self.scale).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def cross_attention_control(self, sim, context, numheads=None):
        #apply mask on sim
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = context.captiontypes >= 2
        mask = repeat(mask, 'b j -> (b h) () j', h=numheads)
        
        # get mask that selects global as well as applicable local prompt
        wf = self.weight_func(sim, context, sim_mask=mask)

        # expand dimensions to match heads
        wf = wf[:, None].repeat(1, numheads, 1, 1)
        wf = wf.view(-1, wf.shape[-2], wf.shape[-1])
        
        # remove the global part from the mask
        wf.masked_fill_(~mask, 0)  # max_neg_value)
        # determine where to use global mask (where no local descriptions are available)
        useglobal = wf.sum(-1) == 0
        
        # get the mask back to 0/1, make sure we only attend to at most one of the local descriptions at once
        wfmaxes = wf.max(-1, keepdim=True)[0]
        wf = (wf >= (wfmaxes.clamp_min(1e-3) - 1e-4))
        
        # get mask that selects only global tokens
        gmask = (context.captiontypes >= 0) & (context.captiontypes < 2)
        gmask = repeat(gmask, 'b j -> (b h) () j', h=numheads)
        
        # update stimulation to attend to global tokens when no local tokens are available
        lmask = wf | (useglobal[:, :, None] & gmask)
        
        gsim = sim.masked_fill(gmask==0, max_neg_value)
        gattn = (gsim * self.scale).softmax(dim=-1)
        
        lsim = sim.masked_fill(lmask==0, max_neg_value)
        lattn = (lsim * self.scale).softmax(dim=-1)
        
        progressclasses = self.progressclassifier(context.progress).view(-1, 2)
        attn = gattn.float() * progressclasses[:, 0][:, None, None] + lattn.float() * progressclasses[:, 1][:, None, None]
        return attn


class ControlPWWLDM(ControlLDM):
    first_stage_key = 'image'
    cond_stage_key = 'all'
    control_key = 'cond_image'
    padlimit = 1  #5
    optiminit_lr = 0.1
    optiminit_numsteps = 20
    threshold_lot = -1
    
    # @torch.no_grad()
    # def get_input(self, batch, k, bs=None, *args, **kwargs):
    #     # takes a batch and outputs image x and conditioning info c  --> keep unchanged
    
    def set_control_drop(self, p=0.):
        self.control_drop = p
    
    def encode_using_text_encoder(self, input_ids):
        outputs = self.cond_stage_model.transformer(input_ids=input_ids, output_hidden_states=self.cond_stage_model.layer=="hidden")
        if self.cond_stage_model.layer == "last":
            text_emb = outputs.last_hidden_state
        elif self.cond_stage_model.layer == "pooled":
            text_emb = outputs.pooler_output[:, None, :]
        else:
            text_emb = outputs.hidden_states[self.layer_idx]
        return text_emb
    
    def optimize_xt(self, x, t, c, i):
        if ("optinit" in self.casmode.chunks and i == 0) or "optall" in self.casmode.chunks:
            c["c_crossattn"][0].compute_attention_losses = True
            with torch.enable_grad():
                orig_x = deepcopy(x)
                x = torch.nn.Parameter(x)
                optimizer = torch.optim.SGD([x], lr=self.optiminit_lr)
                for i in range(self.optiminit_numsteps):
                    optimizer.zero_grad()
                    c["c_crossattn"][0].attention_losses = []
                    _ = self.apply_model(x, t, c)
                    losses = c["c_crossattn"][0].attention_losses
                    loss = sum(losses) / len(losses)
                    del losses
                    loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"Optiminit step {i}, loss: {loss.detach().cpu().item()}")
                        
        return x
    
    def get_learned_conditioning(self, cond):
        # takes conditioning info (cond_key) and preprocesses it to later be fed into LDM
        # returns CustomTextConditioning object
        # called from get_input()
        # must be used with cond_key = "all", then get_input() passes the batch as-is in here
        # DONE: unpack texts, embed them, pack back up and package with cross-attention masks
        
        # this is a non-parallelized implementation
        with torch.no_grad():
            pad_token_id, bos_token_id, modelmaxlen = self.cond_stage_model.tokenizer.pad_token_id, self.cond_stage_model.tokenizer.bos_token_id, self.cond_stage_model.tokenizer.model_max_length 
            device = cond["caption"].device
            tokenids = cond["caption"].cpu()
            layerids = cond["layerids"].cpu()
            encoder_layerids = cond["encoder_layerids"].cpu()
            input_ids = []
            for i in range(len(tokenids)):
                start_j = 0
                for j in range(len(tokenids[0])):
                    layerid = encoder_layerids[i, j].item()
                    next_layerid = encoder_layerids[i, j+1].item() if j+1 < len(tokenids[0]) else -1
                    if next_layerid == -1:
                        break
                    else:     # not padded
                        if next_layerid > layerid:
                            assert next_layerid - layerid == 1
                            input_ids.append(tokenids[i, start_j:j+1])
                            start_j = j+1
                input_ids.append(tokenids[i, start_j:j+1])
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id).to(device)
            # if input_ids.shape[1] < modelmaxlen:
            #     pad_input_ids_block = torch.ones_like(input_ids[:, 0:1]) * pad_token_id
            #     input_ids = torch.cat([input_ids, pad_input_ids_block.repeat(1, modelmaxlen - input_ids.shape[1])], 1)      # concatenate extra paddings to reach 77 length
             
            # 2. encode using text encoder
            text_emb = self.encode_using_text_encoder(input_ids)
            
            # 3. pack text embs back to original format, ensure compatibility with masks
            out_emb = torch.zeros(tokenids.shape[0], tokenids.shape[1], text_emb.shape[2], dtype=text_emb.dtype, device=text_emb.device)
            tokenids_recon = pad_token_id * torch.ones_like(tokenids)
            k = 0
            for i in range(len(tokenids)):
                start_j = 0
                for j in range(len(tokenids[0])):
                    layerid = encoder_layerids[i, j].item()
                    next_layerid = encoder_layerids[i, j+1].item() if j+1 < len(tokenids[0]) else -1
                    if next_layerid == -1:
                        break
                    else:     # not padded
                        if next_layerid > layerid:
                            assert next_layerid - layerid == 1
                            tokenids_recon[i, start_j:j+1] = input_ids[k, :j+1-start_j]
                            out_emb[i, start_j:j+1, :] = text_emb[k, :j+1-start_j, :]
                            start_j = j+1
                            k += 1
                tokenids_recon[i, start_j:j+1] = input_ids[k, :j+1-start_j]
                out_emb[i, start_j:j+1, :] = text_emb[k, :j+1-start_j, :]
                k += 1
            assert torch.all(tokenids == tokenids_recon)
            
        global_prompt_mask = (cond["captiontypes"] < 2) & (cond["captiontypes"] >= 0)
        global_bos_eos_mask = ((cond["caption"] == pad_token_id) | (cond["caption"] == bos_token_id)) & global_prompt_mask
        
        ret = CustomTextConditioning(embs=out_emb,
                                     layer_ids=layerids,
                                     token_ids=tokenids,
                                     global_prompt_mask=global_prompt_mask,
                                     global_bos_eos_mask=global_bos_eos_mask)
        
        ret.threshold = self.threshold
        ret.softness = self.softness
        ret.strength = self.strength
        ret.threshold_lot = self.threshold_lot
        
        ret.captiontypes = cond["captiontypes"]
        
        cross_attn_masks = cond["regionmasks"]    
        cross_attn_masks = {res[0] * res[1]: mask.view(mask.size(0), mask.size(1), -1).transpose(1, 2) for res, mask in cross_attn_masks.items() if res[0] <= 64}
        ret.cross_attn_masks = cross_attn_masks
        
        if self.casmode.name.startswith("legacy"):
            ret.__class__ = CTC_gradio_pww
            legacymethod= self.casmode.name.split("+")[0][len("legacy-"):]
            ret.set_method(legacymethod)
            
        return ret

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        
        # attach progress to cond["c_crossattn"]        # TODO: check that "t" is a tensor of one value per example in the batch
        cond["c_crossattn"][0].progress = 1 - t / self.num_timesteps
        cond["c_crossattn"][0].sigma_t = self.sigmas.to(t.device)[t]

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=None, only_mid_control=self.only_mid_control)
        else:
            scalemult = 1.
            if self.training and hasattr(self, "control_drop") and self.control_drop > 0.:      # control drop during training
                scalemult = 1. if random.random() > self.control_drop else 0.
            # cond["c_crossattn"].on_before_control()
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond)
            control = [c * scale * scalemult for c, scale in zip(control, self.control_scales)]
            if self.global_average_pooling:
                control = [torch.mean(c, dim=(2, 3), keepdim=True) for c in control]
            # cond["c_crossattn"].on_before_controlled()
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond, control=control, only_mid_control=self.only_mid_control)
        return eps
    
    def get_trainable_parameters(self):
        params = []
        saved_param_names = []
        for paramname, param in self.named_parameters():
            if hasattr(param, "train_param") and param.train_param:
                saved_param_names.append(paramname)
                params.append(param)
        
        return params, set(saved_param_names)
    
    def configure_optimizers(self):
        lr = self.learning_rate
        
        params, _ = self.get_trainable_parameters()
        
        for p in self.parameters():
            p.requires_grad = False
        for p in params:
            p.requires_grad = True
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        _, trainable_param_names = self.get_trainable_parameters()
        print(f"Number of parameters in checkpoint: {len(checkpoint['state_dict'])}")
        newstatedict = {}
        for k, v in checkpoint["state_dict"].items():
            if k in trainable_param_names:
                newstatedict[k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = newstatedict
        print(f"Number of trained parameters in checkpoint: {len(checkpoint['state_dict'])}")
        return checkpoint
        # DONE: filter state dict to save only those parameters that have been trained

    @torch.no_grad()
    def get_uncond_batch(self, batch):      # DONE: change regionmasks to fit new prompts
        uncond_cond = deepcopy(batch)       # DONE: change all prompts to "" and re-tokenize
        bos, eos = self.cond_stage_model.tokenizer.bos_token_id, self.cond_stage_model.tokenizer.pad_token_id
        maxlen = (self.padlimit + 1) if self.limitpadding else self.cond_stage_model.tokenizer.model_max_length
        bs = batch["caption"].shape[0]
        device = batch["caption"].device
        
        new_caption3 = torch.ones(bs, maxlen, device=device, dtype=torch.long) * eos
        new_caption3[:, 0] = bos
        new_layerids3 = torch.zeros_like(new_caption3)
        new_captiontypes3 = torch.zeros_like(new_caption3) + 1
        if not self.casmode.use_global_prompt_only:
        # if self.cas_name not in ("cac", "global", "dd") and not self.cas_name.startswith("posattn") and not self.cas_name.startswith("legacy"):
            new_caption3 = torch.cat([new_caption3, new_caption3], 1)
            new_layerids3 = torch.cat([new_layerids3, new_layerids3 + 1], 1)
            new_captiontypes3 = torch.cat([new_captiontypes3, new_captiontypes3 + 1], 1)
        new_captiontypes3[:, 0] = 0
        
        new_regionmasks3 = {k: torch.ones(bs, new_caption3.shape[1], v.shape[2], v.shape[3], device=v.device, dtype=v.dtype) 
                            for k, v in batch["regionmasks"].items()}
        
        uncond_cond["caption"] = new_caption3  #torch.tensor(new_caption).to(device)
        uncond_cond["layerids"] = new_layerids3  #torch.tensor(new_layerids).to(device)
        uncond_cond["encoder_layerids"] = new_layerids3.clone()  #torch.tensor(new_layerids).to(device)
        uncond_cond["captiontypes"] = new_captiontypes3  #torch.tensor(new_captiontypes).to(device)
        uncond_cond["regionmasks"] = new_regionmasks3
                
        return uncond_cond
    
    @torch.no_grad()
    def log_images(self, batch, N=None, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        N = batch["image"].shape[0]
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0]
        # N = min(z.shape[0], N)
        log["reconstruction"] = reconstrimg = self.decode_first_stage(z)  #.clamp(0, 1) * 2.0 - 1.0
        log["control"] = controlimg = c_cat * 2.0 - 1.0
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            n_row = min(z.shape[0], n_row)
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uncond_batch = self.get_uncond_batch(batch)
            _, uc = self.get_input(uncond_batch, self.first_stage_key, bs=N)
            uc_cat, uc_cross = uc["c_concat"][0], uc["c_crossattn"][0]
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning={"c_concat": [uc_cat], "c_crossattn": [uc_cross]},
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = generated_img = x_samples_cfg
            
        log[f"all"] = torch.cat([reconstrimg, controlimg, generated_img], 2)
        del log["reconstruction"]
        del log["control"]
        log["generated"] = log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"]
        del log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"]

        return log
    
    
def convert_model(model, cas_class=None, casmode=None, freezedown=False, simpleencode=False, threshold=-1, strength=0., softness=0.):
    model.__class__ = ControlPWWLDM
    # if simpleencode:
    #     model.__class__ = ControlPWWLDMSimpleEncode
    model.first_stage_key = "image"
    model.control_key = "cond_image"
    model.cond_stage_key = "all"
    model.casmode = casmode
    model.threshold = threshold
    model.strength = strength
    model.softness = softness
    model.threshold_lot = casmode.threshold_lot
    
    if casmode is not None:
        assert cas_class is None
        if casmode.name.startswith("legacy"):
            cas_class = CustomCrossAttentionLegacy
        else:
            cas_class = {"both": CustomCrossAttentionBaselineBoth,
                        "local": CustomCrossAttentionBaselineLocal,
                        "global": CustomCrossAttentionBaselineGlobal,
                        "bothext": CustomCrossAttentionExt,
                        "bothext2": CustomCrossAttentionExt2,
                        "bothminimal": CustomCrossAttentionMinimal,
                        "doublecross": None,
                        "sepswitch": CustomCrossAttentionSepSwitch,
                        "delegated": CustomCrossAttentionDelegated,
                        "cac": CustomCrossAttentionCAC,
                        "dd": CustomCrossAttentionDenseDiffusion,
                        "posattn": CustomCrossAttentionPosattn,
                        "posattn2": CustomCrossAttentionPosattn2,       # with probability mass fix
                        "posattn3": CustomCrossAttentionPosattn3,       # same as 2, sim.max() scaling similar to DenseDiffusion
                        "posattn4": CustomCrossAttentionPosattn4,       # same as 3, + scaling based on region size like in DenseDiffusion
                        "posattn5": CustomCrossAttentionPosattn5,       # same as 2, but using strength as multiplier to attentuate region-specific attention
                        "posattn5a": CustomCrossAttentionPosattn5a,       # same as 2, but using strength as multiplier to attentuate region-specific attention
                        "posattn5b": CustomCrossAttentionPosattn5b,       # same as 2, but rebalanced tokens within a region's description
                        "posattn5c": CustomCrossAttentionPosattn5c,       # same as 2, but rebalanced tokens within a region's description
                        "posattn5u": CustomCrossAttentionPosattn5u,       # same as 2, but uniform everywhere across all tokens within region description
                        "posattn-opt": CustomCrossAttentionPosattnOptimized,
                        "posattn2-opt": CustomCrossAttentionPosattn2Optimized,
                        }[casmode.basename]
        
    if cas_class is None:
        cas_class = CustomCrossAttentionBaseline
        
    print(f"CAS name: {casmode}")
    print(f"CAS class: {cas_class}")
    
    # DONE: replace CrossAttentions that are at attn2 in BasicTransformerBlocks with adapted CustomCrossAttention that takes into account cross-attention masks
    for module in model.model.diffusion_model.modules():
        if isinstance(module, BasicTransformerBlock): # module.__class__.__name__ == "BasicTransformerBlock":
            assert not module.disable_self_attn
            if casmode == "doublecross":
                DoublecrossBasicTransformerBlock.convert(module)
                module.threshold = threshold
            else:
                module.attn2 = cas_class.from_base(module.attn2)
                module.attn2.threshold = threshold
        
    for module in model.control_model.modules():
        if isinstance(module, BasicTransformerBlock): # module.__class__.__name__ == "BasicTransformerBlock":
            assert not module.disable_self_attn
            if casmode == "doublecross":
                DoublecrossBasicTransformerBlock.convert(module)
                module.threshold = threshold
            else:
                module.attn2 = cas_class.from_base(module.attn2)
                module.attn2.threshold = threshold
    
    return model


def get_checkpointing_callbacks(interval=6*60*60, dirpath=None):
    print(f"Checkpointing every {interval} seconds")
    interval_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="interval_delta_epoch={epoch}_step={step}",
        train_time_interval=timedelta(seconds=interval),
        save_weights_only=True,
        save_top_k=-1,
    )
    latest_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="latest_all_epoch={epoch}_step={step}",
        monitor="step",
        mode="max",
        train_time_interval=timedelta(minutes=10),
        save_top_k=1,
    )
    return [interval_checkpoint, latest_checkpoint]


def create_controlnet_pww_model(basemodelname="v1-5-pruned.ckpt", model_name='control_v11p_sd15_seg', casmode="bothext",
                                freezedown=False, simpleencode=False, threshold=-1, strength=0., softness=0.,
                                extendedcontrol=False, loadckpt="", nocontrol=False):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(f'./models/{model_name}.yaml').cpu()
    # load main weights
    model.load_state_dict(load_state_dict(f'./models/{basemodelname}', location='cpu'), strict=False)
    # load controlnet weights
    model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cpu'), strict=False)
    model.base_model_name = basemodelname
    model.controlnet_model_name = model_name
    
    model = convert_model(model, casmode=casmode, freezedown=freezedown, simpleencode=simpleencode, 
                          threshold=threshold, strength=strength, softness=softness)
    model.sigmas = ((1 - model.alphas_cumprod) / model.alphas_cumprod) ** 0.5
    model.sigmas = torch.cat([torch.zeros_like(model.sigmas[0:1]), model.sigmas], 0)
    
    if loadckpt != "":
        for loadckpt_e in loadckpt.split(","):
            if loadckpt_e != "":
                print(f"loading trained parameters from {loadckpt_e}")
                # refparam1a = model.model.diffusion_model.middle_block[1].proj_in.weight.data.clone()
                # refparam2a = deepcopy(model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2l.to_q.weight.data.clone())
                ckpt_state_dict = load_state_dict(loadckpt, location="cpu")
                # testing the partial loading
                model.load_state_dict(ckpt_state_dict, strict=False)
                # refparam1b = model.model.diffusion_model.middle_block[1].proj_in.weight.data.clone()
                # refparam2b = deepcopy(model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2l.to_q.weight.data.clone())
                # assert torch.all(refparam1a == refparam1b)
    return model


class CASMode():
    def __init__(self, name):
        self.chunks = name.split("+")
        self.basename = self.chunks[0]
        self.threshold_lot = -1.
        
        self.localonlytil = False
        for chunk in self.chunks:
            m = re.match(r"lot(\d\.\d+)", chunk)
            if m:
                self.localonlytil = True
                self.threshold_lot = float(m.group(1))
                
        self._use_global_prompt_only = None
        self._augment_global_caption = None
        
    @property
    def name(self):
        return "+".join(self.chunks)
    
    @property
    def use_global_prompt_only(self):
        if self._use_global_prompt_only is not None:
            return self._use_global_prompt_only
        if self.localonlytil:
            return False
        if self.basename.startswith("posattn") or \
           self.basename.startswith("legacy") or \
           self.basename in ("cac", "dd", "global") :
            return True
        else:
            return False
        
    @use_global_prompt_only.setter
    def use_global_prompt_only(self, val:bool):
        self._use_global_prompt_only = val
        
    @property
    def augment_global_caption(self):
        if self._augment_global_caption is not None:
            return self._augment_global_caption
        if "keepprompt" in self.chunks or self.is_test:
            return False
        else:
            if self.name == "doublecross":
                return True
            # TODO
            return True
        
    @property
    def augment_only(self):
        return "augmentonly" in self.chunks or "augment_only" in self.chunks
        
    @augment_global_caption.setter
    def augment_global_caption(self, val:bool):
        self._augment_global_caption = val
        
    @property
    def is_test(self):
        return "test" in self.chunks
        
    @property
    def replace_layerids_with_encoder_layerids(self):
        if self.localonlytil:
            return False
        if self.use_global_prompt_only:
            return False
        else:
            return True
        return ("uselocal" in self.casmodechunks) or (self.casmode in ("cac", "posattn", "posattn2", "posattn3", "dd"))       # use local annotations on global prompt and discard local prompts
        
    def addchunk(self, chunk:str):
        self.chunks.append(chunk)
        return self
    
    def __add__(self, chunk:str):
        return self.addchunk(chunk)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __eq__(self, other:str):
        return self.name.__eq__(other)
    
    def __hash__(self):
        return hash(self.name)
        
