import math
from random import random
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
import pathlib
from pathlib import Path
import torchvision.transforms as T

from typing import Callable, Optional, List, Tuple

from einops import rearrange, repeat

from beartype import beartype

from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from muse_maskgit_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME
from muse_maskgit_pytorch.dna_encoder import EnformerEncoder, OneHotDNAEncoder
from muse_maskgit_pytorch.attend import Attend

from tqdm.auto import tqdm

# Updated muse_maskgit_pytorch.py - Add this import at the top with the other imports
from muse_maskgit_pytorch.dna_encoder import EnformerEncoder, OneHotDNAEncoder
from muse_maskgit_pytorch.improved_dna_encoders import (
    KmerDNAEncoder, MotifDNAEncoder, EfficientDNAEncoder, 
    create_dna_encoder
)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return gate * F.gelu(x)

def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        cross_attend = False,
        scale = 8,
        flash = True,
        dropout = 0.
    ):
        super().__init__()
        self.scale = scale
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.attend = Attend(
            flash = flash,
            dropout = dropout,
            scale = scale
        )

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        assert not (exists(context) ^ self.cross_attend)

        n = x.shape[-2]
        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        if exists(context_mask):
            context_mask = repeat(context_mask, 'b j -> b h i j', h = h, i = n)
            context_mask = F.pad(context_mask, (1, 0), value = True)

        out = self.attend(q, k, v, mask = context_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        flash = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, flash = flash),
                Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend = True, flash = flash),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = LayerNorm(dim)

    def forward(self, x, context=None, context_mask=None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x
            # <- only do cross-attn if we actually have a context
            if context is not None:
                x = cross_attn(x, context=context, context_mask=context_mask) + x
            x = ff(x) + x
        return self.norm(x)

# transformer - it's all we need

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        dim_out = None,
        t5_name = DEFAULT_T5_NAME,
        self_cond = False,
        add_mask_id = False,
        dna_encoder = None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.mask_id = num_tokens if add_mask_id else None

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens + int(add_mask_id), dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.transformer_blocks = TransformerBlocks(dim = dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.dim_out = default(dim_out, num_tokens)
        self.to_logits = nn.Linear(dim, self.dim_out, bias = False)

        # optional DNA encoder
        self.dna_encoder = dna_encoder
        dna_dim = None
        if dna_encoder is not None:
            test_coords = [('chr1', 0, 12800000)]
            with torch.no_grad():
                test_embedding = dna_encoder.encode(test_coords)  # [1, T, E] or [1,1,E]
            dna_dim = test_embedding.shape[-1]

        text_embed_dim = dna_dim if dna_dim is not None else get_encoded_dim(t5_name)

        self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) if text_embed_dim != dim else nn.Identity() 

        # optional self conditioning
        self.self_cond = self_cond
        self.self_cond_to_init_embed = FeedForward(dim)

    # optional (kept so negative prompting etc. won’t crash if used later)
    def encode_text(self, texts: Optional[List[str]]):
        if texts is None:
            return None
        return t5_encode_text(texts, name=self.t5_name)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        if cond_scale == 1:
            return self.forward(*args, return_embed = return_embed, cond_drop_prob = 0., **kwargs)

        logits, embed = self.forward(*args, return_embed = True, cond_drop_prob = 0., **kwargs)

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        text_embed: torch.Tensor,
        neg_text_embed: torch.Tensor,
        cond_scale = 3.,
        return_embed = False,
        **kwargs
    ):
        neg_logits = self.forward(*args, neg_text_embed = neg_text_embed, cond_drop_prob = 0., **kwargs)
        pos_logits, embed = self.forward(*args, return_embed = True, text_embed = text_embed, cond_drop_prob = 0., **kwargs)

        logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def forward(
        self,
        x,
        *,
        return_embed=False,
        dna_coords: Optional[List[tuple]] = None,
        return_logits=False,
        labels=None,
        ignore_index=0,
        self_cond_embed=None,
        cond_drop_prob=0.,
        conditioning_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None
    ):
        device, b, n = x.device, *x.shape
        assert n <= self.seq_len

        # -------------------- build context (DNA/text) if present
        context = None
        context_mask = None

        if self.dna_encoder is not None and dna_coords is not None:
            # keep dna encoder on same device
            self.dna_encoder = self.dna_encoder.to(device)
            context = self.dna_encoder.encode(dna_coords)      # [B, T, E] or [B, 1, E]
            context = context.to(device)
        elif text_embeds is not None:
            context = text_embeds.to(device)
        elif texts is not None:
            context = self.encode_text(texts).to(device)

        if context is not None:
            context = self.text_embed_proj(context)
            context_mask = (context != 0).any(dim=-1)   # [B, T_ctx]

        # -------------------- append conditioning tokens (low-res) as context
        if exists(conditioning_token_ids):
            conditioning_token_ids = rearrange(conditioning_token_ids, 'b ... -> b (...)')   # [B, Lc]
            cond_token_emb = self.token_emb(conditioning_token_ids)                          # [B, Lc, D]
            if context is None:
                context = cond_token_emb
                context_mask = torch.ones(context.shape[:-1], dtype=torch.bool, device=device)
            else:
                context = torch.cat((context, cond_token_emb), dim=-2)
                cond_mask = torch.ones((context.shape[0], conditioning_token_ids.shape[-1]),
                                       dtype=torch.bool, device=device)
                context_mask = torch.cat((context_mask, cond_mask), dim=-1)

        # classifier-free guidance dropout on context
        if cond_drop_prob > 0. and context is not None:
            # drop entire context with prob
            keep = prob_mask_like((b, 1), 1. - cond_drop_prob, device)
            context_mask = context_mask & keep

        # -------------------- token + pos embeddings
        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device=device))

        if self.self_cond:
            if not exists(self_cond_embed):
                self_cond_embed = torch.zeros_like(x)
            x = x + self.self_cond_to_init_embed(self_cond_embed)

        # -------------------- transformer
        embed = self.transformer_blocks(x, context=context, context_mask=context_mask)
        logits = self.to_logits(embed)

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        if self.dim_out == 1:
            loss = F.binary_cross_entropy_with_logits(rearrange(logits, '... 1 -> ...'), labels.float())
        else:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index=ignore_index)

        if not return_logits:
            return loss

        return loss, logits

# self critic wrapper

class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.to_pred = nn.Linear(net.dim, 1)

    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(x, *args, return_embed = True, **kwargs)
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(x, *args, return_embed = True, **kwargs)
        return self.to_pred(embeds)

    def forward(self, x, *args, labels = None, **kwargs):
        _, embeds = self.net(x, *args, return_embed = True, **kwargs)
        logits = self.to_pred(embeds)

        if not exists(labels):
            return logits

        logits = rearrange(logits, '... 1 -> ...')
        return F.binary_cross_entropy_with_logits(logits, labels.float())

# specialized transformers

class MaskGitTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        assert 'add_mask_id' not in kwargs
        super().__init__(*args, add_mask_id = True, **kwargs)

class TokenCritic(Transformer):
    def __init__(self, *args, **kwargs):
        assert 'dim_out' not in kwargs
        super().__init__(*args, dim_out = 1, **kwargs)

# classifier free guidance functions

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device = None):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return uniform(shape, device = device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# noise schedules

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

# main maskgit classes

@beartype
class MaskGit(nn.Module):
    def __init__(
        self,
        image_size,
        transformer: MaskGitTransformer,
        noise_schedule: Callable = cosine_schedule,
        token_critic: Optional[TokenCritic] = None,
        self_token_critic = False,
        vae: Optional[VQGanVAE] = None,
        cond_vae: Optional[VQGanVAE] = None,
        cond_image_size = None,
        cond_drop_prob = 0.5,
        self_cond_prob = 0.9,
        no_mask_token_prob = 0.,
        critic_loss_weight = 1.
    ):
        super().__init__()
        self.vae = vae.copy_for_eval() if exists(vae) else None

        if exists(cond_vae):
            self.cond_vae = cond_vae.eval()
        else:
            self.cond_vae = self.vae

        assert not (exists(cond_vae) and not exists(cond_image_size)), 'cond_image_size must be specified if conditioning'

        self.image_size = image_size
        self.cond_image_size = cond_image_size
        self.resize_image_for_cond_image = exists(cond_image_size)

        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.self_cond = transformer.self_cond
        assert self.vae.codebook_size == self.cond_vae.codebook_size == transformer.num_tokens, 'transformer num_tokens must be set to be equal to the vae codebook size'

        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

        assert not (self_token_critic and exists(token_critic))
        self.token_critic = token_critic

        if self_token_critic:
            self.token_critic = SelfCritic(transformer)

        self.critic_loss_weight = critic_loss_weight

        # self conditioning
        self.self_cond_prob = self_cond_prob

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        self.no_mask_token_prob = no_mask_token_prob

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        texts: Optional[List[str]] = None,
        dna_coords: Optional[List[Tuple[str, int, int]]] = None,
        negative_texts: Optional[List[str]] = None,
        cond_images: Optional[torch.Tensor] = None,
        fmap_size = None,
        temperature = 1.,
        topk_filter_thres = 0.9,
        can_remask_prev_masked = False,
        force_not_use_token_critic = False,
        timesteps = 18,
        cond_scale = 3,
        critic_noise_scale = 1
    ):
        device = next(self.parameters()).device
        fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.image_size))
        seq_len = fmap_size ** 2

        # infer batch size from inputs
        if cond_images is not None:
            batch_size = cond_images.shape[0]
        elif dna_coords is not None:
            batch_size = len(dna_coords)
        elif texts is not None:
            batch_size = len(texts)
        else:
            raise ValueError("Need at least one of cond_images, dna_coords, or texts to determine batch size.")

        ids = torch.full((batch_size, seq_len), self.mask_id, dtype=torch.long, device=device)
        scores = torch.zeros_like(ids, dtype=torch.float32)

        cond_ids = None
        if self.resize_image_for_cond_image:
            assert exists(cond_images), 'conditioning image must be passed for super-res'
            _, cond_ids, _ = self.cond_vae.encode(cond_images)

        # build (optional) context embeddings only if you provided DNA or text
        text_embeds = None
        if self.transformer.dna_encoder is not None and dna_coords is not None:
            text_embeds = self.transformer.dna_encoder.encode(dna_coords)
        elif texts is not None:
            text_embeds = self.transformer.encode_text(texts)

        demask_fn = self.transformer.forward_with_cond_scale

        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic
        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        self_cond_embed = None
        start_temp = temperature

        for step, steps_left in tqdm(zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))), total=timesteps):
            mask_ratio = self.noise_schedule(step)
            k = max(int((mask_ratio * seq_len).item()), 1)
            masked_idx = scores.topk(k, dim=-1).indices
            ids.scatter_(1, masked_idx, self.mask_id)

            logits, embed = demask_fn(
                ids,
                text_embeds=text_embeds,           # can be None
                conditioning_token_ids=cond_ids,   # low-res tokens if provided
                self_cond_embed=self_cond_embed,
                cond_scale=cond_scale,
                return_embed=True
            )
            if self.self_cond:
                self_cond_embed = embed

            logits = top_k(logits, topk_filter_thres)
            temp = start_temp * (steps_left / timesteps)
            pred_ids = gumbel_sample(logits, temperature=temp, dim=-1)

            is_mask = (ids == self.mask_id)
            ids = torch.where(is_mask, pred_ids, ids)

            if use_token_critic:
                sc = token_critic_fn(
                    ids,
                    text_embeds=text_embeds,
                    conditioning_token_ids=cond_ids,
                    cond_scale=cond_scale
                ).squeeze(-1)
                sc = sc + (uniform(sc.shape, device=device) - 0.5) * critic_noise_scale * (steps_left / timesteps)
                scores = sc
            else:
                probs = logits.softmax(-1)
                scores = 1 - probs.gather(2, pred_ids[..., None]).squeeze(-1)
                if not can_remask_prev_masked:
                    scores.masked_fill_(~is_mask, -1e5)

        ids = rearrange(ids, 'b (h w) -> b h w', h=fmap_size)
        return self.vae.decode_from_ids(ids)
    
    

    def forward(
        self,
        images_or_ids: torch.Tensor,
        ignore_index = -1,
        cond_images: Optional[torch.Tensor] = None,
        cond_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        dna_coords: Optional[List[tuple]] = None,
        cond_drop_prob = None,
        train_only_generator = False,
        sample_temperature = None
    ):
        # tokenize if needed

        if images_or_ids.dtype == torch.float:
            assert exists(self.vae), 'vqgan vae must be passed in if training from raw images'
            assert all([height_or_width == self.image_size for height_or_width in images_or_ids.shape[-2:]]), 'the image you passed in is not of the correct dimensions'

            with torch.no_grad():
                _, ids, _ = self.vae.encode(images_or_ids)
        else:
            assert not self.resize_image_for_cond_image, 'you cannot pass in raw image token ids if you want the framework to autoresize image for conditioning super res transformer'
            ids = images_or_ids

        # take care of conditioning image if specified

        if self.resize_image_for_cond_image:
            cond_images_or_ids = F.interpolate(images_or_ids, self.cond_image_size, mode = 'nearest')

        # get some basic variables

        ids = rearrange(ids, 'b ... -> b (...)')

        batch, seq_len, device, cond_drop_prob = *ids.shape, ids.device, default(cond_drop_prob, self.cond_drop_prob)

        # tokenize conditional images if needed

        assert not (exists(cond_images) and exists(cond_token_ids)), 'if conditioning on low resolution, cannot pass in both images and token ids'

        if exists(cond_images):
            assert exists(self.cond_vae), 'cond vqgan vae must be passed in'
            assert all([height_or_width == self.cond_image_size for height_or_width in cond_images.shape[-2:]])

            with torch.no_grad():
                _, cond_token_ids, _ = self.cond_vae.encode(cond_images)

        # prepare mask

        rand_time = uniform((batch,), device = device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min = 1)

        mask_id = self.mask_id
        batch_randperm = torch.rand((batch, seq_len), device = device).argsort(dim = -1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        mask_id = self.transformer.mask_id
        labels = torch.where(mask, ids, ignore_index)

        if self.no_mask_token_prob > 0.:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = torch.where(mask, mask_id, ids)

        # get text embeddings

        if exists(texts):
            text_embeds = self.transformer.encode_text(texts)
            texts = None

        # self conditioning

        self_cond_embed = None

        if self.transformer.self_cond and random() < self.self_cond_prob:
            with torch.no_grad():
                _, self_cond_embed = self.transformer(
                    x,
                    text_embeds = text_embeds,
                    conditioning_token_ids = cond_token_ids,
                    cond_drop_prob = 0.,
                    return_embed = True,
                    dna_coords = dna_coords
                )

                self_cond_embed.detach_()

        # get loss

        ce_loss, logits = self.transformer(
            x,
            text_embeds = text_embeds,
            self_cond_embed = self_cond_embed,
            conditioning_token_ids = cond_token_ids,
            labels = labels,
            cond_drop_prob = cond_drop_prob,
            ignore_index = ignore_index,
            return_logits = True,
            dna_coords = dna_coords
        )

        if not exists(self.token_critic) or train_only_generator:
            return ce_loss

        # token critic loss

        sampled_ids = gumbel_sample(logits, temperature = default(sample_temperature, random()))

        critic_input = torch.where(mask, sampled_ids, x)
        critic_labels = (ids != critic_input).float()

        bce_loss = self.token_critic(
            critic_input,
            text_embeds = text_embeds,
            conditioning_token_ids = cond_token_ids,
            labels = critic_labels,
            cond_drop_prob = cond_drop_prob
        )

        return ce_loss + self.critic_loss_weight * bce_loss

# final Muse class

@torch.no_grad()
def generate_from_dna(maskgit: MaskGit,
                      dna_coords,               # List[tuple]  len == B
                      cond_images=None,         # low-res maps (B,1,H,W) for SR
                      temperature=1.,
                      topk_filter_thres=0.9,
                      timesteps=18,
                      cond_scale=3.,
                      critic_noise_scale=1.,
                      force_not_use_token_critic=False,
                      can_remask_prev_masked=False):
    """
    Replica of MaskGit.generate, but conditioned on DNA windows
    instead of text.  Works for both base and super-res models.
    """
    device   = next(maskgit.parameters()).device
    fmap     = maskgit.vae.get_encoded_fmap_size(maskgit.image_size)
    seq_len  = fmap ** 2
    B        = len(dna_coords)

    # ------------------------------------------------------------------ 0) book-keeping
    ids    = torch.full((B, seq_len), maskgit.mask_id, device=device)
    scores = torch.zeros_like(ids, dtype=torch.float32)

    cond_token_ids = None
    if maskgit.resize_image_for_cond_image:
        assert cond_images is not None, '`cond_images` required for SR model'
        _, cond_token_ids, _ = maskgit.cond_vae.encode(cond_images)

    # choose which forward pass we’ll call (with CFG)
    demask_fn = maskgit.transformer.forward_with_cond_scale
    if exists(maskgit.token_critic) and not force_not_use_token_critic:
        token_critic_fn = maskgit.token_critic.forward_with_cond_scale
        use_token_critic = True
    else:
        use_token_critic = False

    self_cond_embed = None
    start_T = temperature

    for step, t in enumerate(torch.linspace(0, 1, timesteps, device=device)):
        # 1) mask some tokens ------------------------------------------------
        mask_ratio  = maskgit.noise_schedule(t)
        k           = max(int((mask_ratio * seq_len).item()), 1)
        masked_idx  = scores.topk(k, dim=-1).indices
        ids.scatter_(1, masked_idx, maskgit.mask_id)

        # 2) forward pass ----------------------------------------------------
        logits, embed = demask_fn(ids,
                                  dna_coords=dna_coords,
                                  self_cond_embed=self_cond_embed,
                                  conditioning_token_ids=cond_token_ids,
                                  cond_scale=cond_scale,
                                  return_embed=True)
        if maskgit.self_cond:
            self_cond_embed = embed.detach()

        # 3) sample ----------------------------------------------------------
        logits = top_k(logits, topk_filter_thres)
        temp   = start_T * (timesteps - 1 - step) / (timesteps - 1)
        pred   = gumbel_sample(logits, temperature=temp, dim=-1)

        is_mask = ids == maskgit.mask_id
        ids     = torch.where(is_mask, pred, ids)

        # 4) update confidence scores ---------------------------------------
        if use_token_critic:
            sc = token_critic_fn(ids,
                                 dna_coords=dna_coords,
                                 conditioning_token_ids=cond_token_ids,
                                 cond_scale=cond_scale).squeeze(-1)
            sc += (uniform(sc.shape, device=device) - 0.5) * critic_noise_scale
            scores = sc
        else:
            probs  = logits.softmax(-1)
            scores = 1 - probs.gather(2, pred[..., None]).squeeze(-1)
            if not can_remask_prev_masked:
                scores.masked_fill_(~is_mask, -1e5)

    # ------------------------------------------------------------------ decode
    ids = rearrange(ids, 'b (h w) -> b h w', h=fmap)
    return maskgit.vae.decode_from_ids(ids)
# ----------------------------------------------------------------------


@beartype
class Muse(nn.Module):
    def __init__(
        self,
        base: MaskGit,
        superres: MaskGit
    ):
        super().__init__()
        self.base_maskgit = base.eval()

        assert superres.resize_image_for_cond_image
        self.superres_maskgit = superres.eval()

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        cond_scale = 3.,
        temperature = 1.,
        timesteps = 18,
        superres_timesteps = None,
        return_lowres = False,
        return_pil_images = True
    ):
        lowres_image = self.base_maskgit.generate(
            texts = texts,
            cond_scale = cond_scale,
            temperature = temperature,
            timesteps = timesteps
        )

        superres_image = self.superres_maskgit.generate(
            texts = texts,
            cond_scale = cond_scale,
            cond_images = lowres_image,
            temperature = temperature,
            timesteps = default(superres_timesteps, timesteps)
        )
        
        if return_pil_images:
            lowres_image = list(map(T.ToPILImage(), lowres_image))
            superres_image = list(map(T.ToPILImage(), superres_image))            

        if not return_lowres:
            return superres_image

        return superres_image, lowres_image
