#!/usr/bin/env python

# Created on 2018/12/18
# Author: Kaituo XU

import argparse
import os

import librosa
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch

from data import AudioDataLoader, AudioDataset
from pit_criterion import cal_loss
from tasnet import TasNet


parser = argparse.ArgumentParser('Evaluate separation performance using TasNet')
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str, required=True,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')


def evaluate(args):
    total_sisnr = 0
    total_sdr = 0
    total_cnt = 0
    # Load model
    model = TasNet.load_model(args.model_path)
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    dataset = AudioDataset(args.data_dir, args.batch_size,
                           sample_rate=args.sample_rate, L=model.L)
    data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)

    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            # Get batch data
            padded_mixture, mixture_lengths, padded_source = data
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            # Forward
            estimate_source = model(padded_mixture, mixture_lengths)  # [B, C, K, L]
            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            # Remove padding and flat
            mixture = remove_pad_and_flat(padded_mixture, mixture_lengths)
            source = remove_pad_and_flat(padded_source, mixture_lengths)
            estimate_source = remove_pad_and_flat(estimate_source, mixture_lengths)
            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                # src_ref = np.stack([s1, s2], axis=0)
                # src_est = np.stack([recon_s1_sig, recon_s2_sig], axis=0)
                src_anchor = np.stack([mix, mix], axis=0)
                sisnr1 = get_SISNR(src_ref[0], src_est[0])
                sisnr2 = get_SISNR(src_ref[1], src_est[1])
                sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
                sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
                # sisnr1 = get_SISNR(s1, recon_s1_sig)
                # sisnr2 = get_SISNR(s2, recon_s2_sig)
                print("sisnr1: {0:.2f}, sisnr2: {1:.2f}".format(sisnr1, sisnr2))
                print("sdr1: {0:.2f}, sdr2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[0]))

                total_sisnr += sisnr1 + sisnr2
                total_sdr += (sdr[0]-sdr0[0]) + (sdr[1]-sdr0[0])
                total_cnt += 2
    print("Average sisnr improvement: {0:.2f}".format(total_sisnr / total_cnt))
    print("Average sdr improvement: {0:.2f}".format(total_sdr / total_cnt))

            
def remove_pad_and_flat(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, K, L] or [B, K, L]
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 4:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 4: # [B, C, K, L]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 3:  # [B, K, L]
            results.append(input[:length].view(-1).cpu().numpy())
    return results


def get_SISNR(ref_sig, out_sig, eps=1e-8):
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    evaluate(args)
