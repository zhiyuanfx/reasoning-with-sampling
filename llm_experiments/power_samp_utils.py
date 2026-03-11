import os

from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets


import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from grader_utils.parse_utils import parse_answer
from constants import *

### DESCRIPTION ###
# power sampling to sample from p^{alpha}, where p is the base model
# takes in 1/alpha (temperature) as an argument (default 0.25), and mcmc_power_samp implements sampling from p^{alpha} 


class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    # returns log probs
    @torch.no_grad()
    def next_token(self, prefix):
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]
        output = self.model(prefix_cond)
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)



# returns probabilities (normed)
def normalize(dist):
    probs = F.softmax(dist, dim=-1)
    return probs

# returns sum of logits (product of distributions p*q)
def dist_product(logit_p, logit_q):
    return logit_p+logit_q

# returns logit scaled by temp (temperature scaling p^(1/tau))
def dist_temp_scale(logit_p, temp):
    return logit_p * torch.tensor(1 / temp, dtype=logit_p.dtype, device=logit_p.device)

# low-temperature sampling proposal distribution
@torch.inference_mode()
def naive_temp(p : AutoregressiveSampler, context, temp, seq_len):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]

    idx = tokens.view(unscaled_logits.shape[0], 1, 1)
    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1)
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1)

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm.cpu().numpy(), log_probs_unnorm.cpu().numpy()


def sample_inverse_prob_index(
    log_probs_norm,
    c,
    t,
    jump_size,
    neighbor_blocks=2,
    w_min=0.0,
    w_max=1e9,
):
    """
    Sample a resampling start index idx in full-sequence coordinates.

    Args:
        log_probs_norm: cached token log probs for generated tokens only.
                        log_probs_norm[i] corresponds to token at full position c + i.
        c: length of prompt/context
        t: current full sequence length
        jump_size: block size in tokens
        neighbor_blocks: how many neighboring blocks to consider.
                         <= 0 means all generated prefix.
        w_min, w_max: clipping range for logits based on -log p

    Returns:
        idx: integer in [start_idx, t - 1], using full-sequence indexing
    """
    if t <= c:
        raise ValueError("Cannot sample inverse-prob index when there are no generated tokens.")

    if neighbor_blocks <= 0:
        start_idx = c
    else:
        start_idx = max(c, t - neighbor_blocks * jump_size)

    end_idx = t - 1
    assert start_idx <= end_idx

    start_off = start_idx - c
    end_off = end_idx - c

    local_log_probs = log_probs_norm[start_off:end_off + 1]
    weight_logits = np.clip(-local_log_probs, w_min, w_max).astype(np.float64)

    if not np.all(np.isfinite(weight_logits)):
        probs = np.ones_like(weight_logits, dtype=np.float64) / len(weight_logits)
    else:
        shifted = weight_logits - np.max(weight_logits)
        weights = np.exp(shifted)
        if (not np.all(np.isfinite(weights))) or weights.sum() <= 0:
            probs = np.ones_like(weight_logits, dtype=np.float64) / len(weight_logits)
        else:
            probs = weights / weights.sum()

    sampled_off = np.random.choice(np.arange(start_off, end_off + 1), p=probs)
    idx = c + int(sampled_off)
    return idx


def sample_mcmc_index(
    log_probs_norm,
    c,
    t,
    jump_size,
    index_sample="uniform",
    neighbor_blocks=0,
    w_min=0.0,
    w_max=1e9,
):
    if t <= c:
        raise ValueError("Cannot sample MCMC index when there are no generated tokens.")

    if index_sample == "uniform":
        start_idx = c if neighbor_blocks <= 0 else max(c, t - neighbor_blocks * jump_size)
        return random.randint(start_idx, t - 1)

    if index_sample == "inverse_prob":
        return sample_inverse_prob_index(
            log_probs_norm=log_probs_norm,
            c=c,
            t=t,
            jump_size=jump_size,
            neighbor_blocks=neighbor_blocks,
            w_min=w_min,
            w_max=w_max,
        )

    raise ValueError(f"Unknown index sampling mode: {index_sample}")

@torch.inference_mode()
def mcmc_power_samp(
    p : AutoregressiveSampler,
    context,
    temp,
    mcmc_steps,
    max_new_tokens,
    block_num=16,
    index_sample="uniform",
    neighbor_blocks=0,
    w_min=0.0,
    w_max=1e9,
):
    c = len(context)
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = np.empty(max_new_tokens, dtype=np.float32)
    log_probs_unnorm = np.empty(max_new_tokens, dtype=np.float32)
    filled = 0
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    attempts = 0
    acceptances = 0

    for _ in range(block_num):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        add_len = len(lp_norm)
        log_probs_norm[filled:filled + add_len] = lp_norm
        log_probs_unnorm[filled:filled + add_len] = lp_unnorm
        filled += add_len

        for _ in range(mcmc_steps):
            attempts+=1
            t = len(gen)
            idx = sample_mcmc_index(
                log_probs_norm=log_probs_norm,
                c=c,
                t=t,
                jump_size=jump_size,
                index_sample=index_sample,
                neighbor_blocks=neighbor_blocks,
                w_min=w_min,
                w_max=w_max,
            )
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm[idx-c:s-c]
            log_r = np.sum(target_log_prob_prop) + np.sum(log_prob_cur) - np.sum(target_log_prob_cur) - np.sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                gen = prop
                log_probs_norm[idx-c:s-c] = log_prob_prop
                log_probs_unnorm[idx-c:s-c] = target_log_prob_prop
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            gen_new_len = max(0, (eos_idx + 1) - c)
            acceptance_ratio = acceptances/attempts if attempts > 0 else 0.0
            assert len(gen) == c + gen_new_len
            return gen, log_probs_norm[:gen_new_len], log_probs_unnorm[:gen_new_len], acceptance_ratio

    acceptance_ratio = acceptances/attempts if attempts > 0 else 0.0
    return gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio


def find_subseq_start(seq: list[int], subseq: list[int]) -> int:
    if not subseq or len(subseq) > len(seq):
        return -1
    n = len(subseq)
    first = subseq[0]
    for i, x in enumerate(seq):
        if x == first and seq[i:i+n] == subseq:
            print(f"[info] found delimiter at position {i} in newly generated region")
            return i
    return -1


@torch.inference_mode()
def mcmc_power_samp_truncate(
    p,  # AutoregressiveSampler
    context: list[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 16,
    sample_in_block: bool = False,
    step_delim: str = STEP_DELIM,
):
    """
    Fixed-length blockwise MCMC (same MH as before), but after finishing MH for each block,
    truncate the final winner (current `gen`) at the first occurrence of `step_delim`
    found *within that block's newly added region only*.

    Returns:
      gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio

    Notes:
      - We keep the delimiter tokens in `gen` (truncate to end of delimiter token sequence).
      - `filled` is kept consistent with current gen: filled == len(gen) - c.
    """
    c = len(context)
    gen = context.copy() if context is not None else []

    assert max_new_tokens % block_num == 0
    jump_size = max_new_tokens // block_num

    log_probs_norm = np.empty(max_new_tokens, dtype=np.float32)
    log_probs_unnorm = np.empty(max_new_tokens, dtype=np.float32)
    filled = 0

    # Tokenize delimiter once (robust across templates/spacing)
    delim_ids = p.tokenizer.encode(step_delim, add_special_tokens=False)

    attempts = 0
    acceptances = 0

    for _ in range(block_num):
        prev_t = len(gen)
        # (A) Propose fixed-length extension
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=len(gen) + jump_size)

        add_len = len(lp_norm)
        log_probs_norm[filled:filled + add_len] = lp_norm
        log_probs_unnorm[filled:filled + add_len] = lp_unnorm
        filled += add_len

        # (B) MH inner loop 
        for _ in range(mcmc_steps):
            attempts += 1
            t = len(gen)

            if not sample_in_block:
                idx = random.randint(c, t - 1)
            else:
                idx = random.randint(max(c, t - jump_size), t - 1)

            prop, log_prob_prop, target_log_prob_prop = naive_temp(
                p, gen[:idx], temp=temp, seq_len=t
            )
            s = len(prop)

            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx

            log_prob_cur = log_probs_norm[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm[idx - c : s - c]

            log_r = np.sum(target_log_prob_prop) + np.sum(log_prob_cur) - np.sum(target_log_prob_cur)- np.sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop
                
                txt = p.tokenizer.decode(gen[prev_t:], skip_special_tokens=False)
                if step_delim in txt:
                    print("[info] delimiter appears in decoded new region")
                else:
                    print("[info] delimiter NOT in decoded new region")
                    
                log_probs_norm[idx - c : s - c] = log_prob_prop
                log_probs_unnorm[idx - c : s - c] = target_log_prob_prop

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # (C) Early stop on EOS 
        eos_id = p.tokenizer.eos_token_id
        if eos_id in gen:
            eos_idx = gen.index(eos_id)
            gen = gen[:eos_idx + 1]
            filled = max(0, len(gen) - c)
            acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
            return gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio

        # (D) Truncate final winner for this block 
        if delim_ids:
            new_region = gen[prev_t:]
            pos = find_subseq_start(new_region, delim_ids)
            if pos != -1:
                cut_full_len = prev_t + pos + len(delim_ids)
                if cut_full_len < len(gen):
                    gen = gen[:cut_full_len]
                    filled = max(0, len(gen) - c)

    acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
    return gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio

@torch.inference_mode()
def mcmc_power_samp_one_step(
    p,  # AutoregressiveSampler
    context: list[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 16,
    sample_in_block: bool = False,
    step_delim: str = STEP_DELIM,
):
    """
    Experimental variant:
      - reuse existing naive_temp() unchanged (upper-bound generation)
      - after EACH candidate generation (block extension + each MH proposal):
          truncate the NEWLY GENERATED SUFFIX at first STEP_DELIM occurrence (inclusive)
      - MH accept/reject compares variable-length suffixes using length-normalized sums:
          log_r = avg(target_prop) + avg(prop_cur) - avg(target_cur) - avg(prop_prop)
        where avg(x)=sum(x)/len(x) (0 if len==0)

    Returns:
      gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio
    """
    c = len(context)
    gen = context.copy() if context is not None else []

    assert max_new_tokens % block_num == 0
    jump_size = max_new_tokens // block_num

    # store per-token logprobs for current gen's generated part (positions c:)
    log_probs_norm = np.empty(max_new_tokens, dtype=np.float32)
    log_probs_unnorm = np.empty(max_new_tokens, dtype=np.float32)
    filled = 0

    delim_ids = p.tokenizer.encode(step_delim, add_special_tokens=False)
    eos_id = p.tokenizer.eos_token_id

    attempts = 0
    acceptances = 0

    for _ in range(block_num):
        # define "block start" for optional in-block sampling (best-effort under truncation)
        block_start_full = len(gen)

        # ---- (A) Extend current state, then truncate suffix immediately
        old_len = len(gen)
        prop_full, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=old_len + jump_size)

        # newly generated suffix (beyond prefix old_len)
        suffix_tokens = prop_full[old_len:]
        lp_norm = np.asarray(lp_norm, dtype=np.float32)
        lp_unnorm = np.asarray(lp_unnorm, dtype=np.float32)

        # truncate suffix at first delim occurrence (inclusive)
        if delim_ids and len(suffix_tokens) > 0:
            pos = find_subseq_start(suffix_tokens, delim_ids)
            if pos != -1:
                cut = pos + len(delim_ids)
                if cut < len(suffix_tokens):
                    suffix_tokens = suffix_tokens[:cut]
                    lp_norm = lp_norm[:cut]
                    lp_unnorm = lp_unnorm[:cut]

        # apply update
        gen = gen + suffix_tokens
        add_len = len(suffix_tokens)
        if add_len > 0:
            log_probs_norm[filled:filled + add_len] = lp_norm
            log_probs_unnorm[filled:filled + add_len] = lp_unnorm
            filled += add_len

        # EOS early stop
        if eos_id in gen:
            eos_idx = gen.index(eos_id)
            gen = gen[:eos_idx + 1]
            filled = max(0, len(gen) - c)
            acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
            return gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio

        # ---- (B) MH inner loop (variable-length due to truncation)
        for _ in range(mcmc_steps):
            attempts += 1
            t = len(gen)
            if t <= c:
                continue

            if not sample_in_block:
                idx = random.randint(c, t - 1)
            else:
                lo = max(c, block_start_full)
                if lo > t - 1:
                    lo = c
                idx = random.randint(lo, t - 1)

            # propose from prefix gen[:idx], upper-bound length t
            prop_full, prop_lp_norm, prop_lp_unnorm = naive_temp(p, gen[:idx], temp=temp, seq_len=t)

            # suffix from idx onward
            new_suffix_tokens = prop_full[idx:]
            prop_lp_norm = np.asarray(prop_lp_norm)
            prop_lp_unnorm = np.asarray(prop_lp_unnorm)

            # truncate proposed suffix at delim (inclusive)
            if delim_ids and len(new_suffix_tokens) > 0:
                pos = find_subseq_start(new_suffix_tokens, delim_ids)
                if pos != -1:
                    cut = pos + len(delim_ids)
                    if cut < len(new_suffix_tokens):
                        new_suffix_tokens = new_suffix_tokens[:cut]
                        prop_lp_norm = prop_lp_norm[:cut]
                        prop_lp_unnorm = prop_lp_unnorm[:cut]

            # proposed full seq after truncation
            prop_trunc = gen[:idx] + new_suffix_tokens

            # current suffix probs slice (idx -> end)
            cur_suffix_len = len(gen) - idx
            cur_start = idx - c
            cur_end = cur_start + cur_suffix_len  # should equal filled

            cur_lp_norm = log_probs_norm[cur_start:cur_end]
            cur_lp_unnorm = log_probs_unnorm[cur_start:cur_end]

            # length-normalized sums (avg). if empty => 0
            if prop_lp_norm.shape[0] > 0:
                prop_prop_avg = float(np.sum(prop_lp_norm) / prop_lp_norm.shape[0])
                prop_targ_avg = float(np.sum(prop_lp_unnorm) / prop_lp_unnorm.shape[0])
            else:
                prop_prop_avg = 0.0
                prop_targ_avg = 0.0

            if cur_lp_norm.shape[0] > 0:
                cur_prop_avg = float(np.sum(cur_lp_norm) / cur_lp_norm.shape[0])
                cur_targ_avg = float(np.sum(cur_lp_unnorm) / cur_lp_unnorm.shape[0])
            else:
                cur_prop_avg = 0.0
                cur_targ_avg = 0.0

            log_r = prop_targ_avg + cur_prop_avg - cur_targ_avg - prop_prop_avg

            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                gen = prop_trunc

                # update stored probs: keep up to idx, replace suffix with proposed suffix probs
                prop_suffix_len = int(prop_lp_norm.shape[0])
                new_filled = max(0, len(gen) - c)

                if prop_suffix_len > 0:
                    log_probs_norm[cur_start:cur_start + prop_suffix_len] = prop_lp_norm
                    log_probs_unnorm[cur_start:cur_start + prop_suffix_len] = prop_lp_unnorm

                # set filled to match new gen
                filled = min(cur_start + prop_suffix_len, new_filled)

                # EOS early stop
                if eos_id in gen:
                    eos_idx = gen.index(eos_id)
                    gen = gen[:eos_idx + 1]
                    filled = max(0, len(gen) - c)
                    acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
                    return gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
    filled = max(0, len(gen) - c)
    return gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio

# alpha = infty power sampling; temp is for proposal distribution
@torch.inference_mode()
def max_swap(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    # print(f'Temp: {temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = np.empty(max_new_tokens, dtype=np.float32)
    log_probs_unnorm = np.empty(max_new_tokens, dtype=np.float32)
    filled = 0


    # print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    # print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in range(block_num):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        add_len = len(lp_norm)
        log_probs_norm[filled:filled + add_len] = lp_norm
        log_probs_unnorm[filled:filled + add_len] = lp_unnorm
        filled += add_len

        for _ in range(mcmc_steps):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) - sum(target_log_prob_cur)

            if log_r > 0:
                acceptances+=1
                gen = prop
                log_probs_norm[idx-c:s-c] = log_prob_prop
                log_probs_unnorm[idx-c:s-c] = target_log_prob_prop
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            gen_new_len = max(0, (eos_idx + 1) - c)
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm[:gen_new_len], log_probs_unnorm[:gen_new_len], acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio

# power sampling with autoregressive mcmc
def mcmc_power_samp_old(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16):
    c = len(context)
    # print(f'alpha: {1/temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    # print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    # print(jump_size)
    attempts = 0
    acceptances = 0


    for _ in range(block_num):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in range(mcmc_steps):
            attempts+=1
            t = len(gen)
            idx = random.randint(c, t-1)
            # llm query takes the burden of time
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio


def format_prompt(question, model, tokenizer, cot=True, semantic_block=False):
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str+=COT if not semantic_block else COT_STEP
        else:
            format_str+=BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str+=COT if not semantic_block else COT_STEP
        else:
            format_str+=BASE

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT if not semantic_block else COT_STEP
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT if not semantic_block else COT_STEP
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str+=COT if not semantic_block else COT_STEP
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str+=COT if not semantic_block else COT_STEP
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    return format_str
