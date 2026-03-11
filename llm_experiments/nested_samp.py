from contextlib import nullcontext
from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import transformers

from constants import *
from power_samp_utils import AutoregressiveSampler, naive_temp, sample_inverse_prob_index


@torch.inference_mode()
def nested_power_samp(
    p: AutoregressiveSampler,
    context,
    temp,
    mcmc_steps,
    max_new_tokens,
    block_num=16,
    mode="block",
    neighbor_blocks=2,
    w_min=0.0,
    w_max=1e9,
):
    """
    Nested-sampling-style solver with two modes:
        - mode="block": always resample from current block boundary
        - mode="inverse_prob": sample resampling start index according to clipped -log p weights

    Returns same signature as mcmc_power_samp:
        gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    For this nested solver, acceptance / index weighting uses log_probs_norm from naive_temp().
    We still cache log_probs_unnorm from naive_temp() so the return values match the
    semantics used by mcmc_power_samp().
    """
    if temp <= 0:
        raise ValueError("temp must be > 0.")
    if temp >= 1:
        raise ValueError("nested_power_samp currently assumes temp < 1 so that temp/(1-temp) is well-defined.")

    c = len(context)
    gen = []
    if context is not None:
        gen = context.copy()

    log_probs_norm = np.empty(max_new_tokens, dtype=np.float32)
    log_probs_unnorm = np.empty(max_new_tokens, dtype=np.float32)  # kept for return-signature compatibility
    filled = 0

    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)

    attempts = 0
    acceptances = 0

    # alpha = 1 / temp, so 1 / (alpha - 1) = temp / (1 - temp)
    accept_scale = temp / (1.0 - temp)

    for _ in range(block_num):
        # Step 1: initial extension for this block
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size + len(gen))
        add_len = len(lp_norm)

        log_probs_norm[filled:filled + add_len] = lp_norm
        log_probs_unnorm[filled:filled + add_len] = lp_unnorm
        filled += add_len

        # Step 2: nested refinement updates
        for _ in range(mcmc_steps):
            attempts += 1
            t = len(gen)

            if t <= c:
                break

            if mode == "block":
                idx = max(c, t - jump_size)
            elif mode == "inverse_prob":
                idx = sample_inverse_prob_index(
                    log_probs_norm=log_probs_norm,
                    c=c,
                    t=t,
                    jump_size=jump_size,
                    neighbor_blocks=neighbor_blocks,
                    w_min=w_min,
                    w_max=w_max,
                )
            else:
                raise ValueError(f"Unknown nested sampling mode: {mode}")

            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)

            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx

            log_prob_cur = log_probs_norm[idx - c:s - c]

            # Since prefix before idx is identical, comparing suffix log probs is enough.
            # We implement the rule literally:
            #   log p(x') > (temp / (1-temp)) * log p(x)
            if np.sum(log_prob_prop) > accept_scale * np.sum(log_prob_cur):
                acceptances += 1
                gen = prop
                log_probs_norm[idx - c:s - c] = log_prob_prop
                log_probs_unnorm[idx - c:s - c] = target_log_prob_prop

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            gen_new_len = max(0, (eos_idx + 1) - c)
            acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
            assert len(gen) == c + gen_new_len
            return gen, log_probs_norm[:gen_new_len], log_probs_unnorm[:gen_new_len], acceptance_ratio

    acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
    return gen, log_probs_norm[:filled], log_probs_unnorm[:filled], acceptance_ratio
