#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pickle
import sys

sys.path.insert(0, 'stylegan3')

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

ORIGINAL_REPO_URL = 'https://github.com/NVlabs/stylegan3'
TITLE = 'NVlabs/stylegan3'
DESCRIPTION = f'This is a demo for {ORIGINAL_REPO_URL}.'
SAMPLE_IMAGE_DIR = 'https://huggingface.co/spaces/hysts/StyleGAN3/resolve/main/samples'
ARTICLE = f'''## Generated images
- truncation: 0.7
### AFHQv2
- size: 512x512
- seed: 0-99
![AFHQv2 samples]({SAMPLE_IMAGE_DIR}/afhqv2.jpg)
### FFHQ
- size: 1024x1024
- seed: 0-99
![FFHQ samples]({SAMPLE_IMAGE_DIR}/ffhq.jpg)
### FFHQ-U
- size: 1024x1024
- seed: 0-99
![FFHQ-U samples]({SAMPLE_IMAGE_DIR}/ffhq-u.jpg)
### MetFaces
- size: 1024x1024
- seed: 0-99
![MetFaces samples]({SAMPLE_IMAGE_DIR}/metfaces.jpg)
### MetFaces-U
- size: 1024x1024
- seed: 0-99
![MetFaces-U samples]({SAMPLE_IMAGE_DIR}/metfaces-u.jpg)
'''

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def make_transform(translate: tuple[float, float], angle: float) -> np.ndarray:
    mat = np.eye(3)
    sin = np.sin(angle / 360 * np.pi * 2)
    cos = np.cos(angle / 360 * np.pi * 2)
    mat[0][0] = cos
    mat[0][1] = sin
    mat[0][2] = translate[0]
    mat[1][0] = -sin
    mat[1][1] = cos
    mat[1][2] = translate[1]
    return mat


def generate_z(z_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.random.RandomState(seed).randn(
        1, z_dim)).to(device)


@torch.inference_mode()
def generate_image(model_name: str, seed: int, truncation_psi: float,
                   tx: float, ty: float, angle: float,
                   model_dict: dict[str, nn.Module],
                   device: torch.device) -> np.ndarray:
    model = model_dict[model_name]
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))

    z = generate_z(model.z_dim, seed, device)
    label = torch.zeros([1, model.c_dim], device=device)

    mat = make_transform((tx, ty), angle)
    mat = np.linalg.inv(mat)
    model.synthesis.input.transform.copy_(torch.from_numpy(mat))

    out = model(z, label, truncation_psi=truncation_psi)
    out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return out[0].cpu().numpy()


def load_model(file_name: str, device: torch.device) -> nn.Module:
    path = hf_hub_download('hysts/StyleGAN3',
                           f'models/{file_name}',
                           use_auth_token=TOKEN)
    with open(path, 'rb') as f:
        model = pickle.load(f)['G_ema']
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.z_dim)).to(device)
        label = torch.zeros([1, model.c_dim], device=device)
        model(z, label)
    return model


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    model_names = {
        'AFHQv2-512-R': 'stylegan3-r-afhqv2-512x512.pkl',
        'FFHQ-1024-R': 'stylegan3-r-ffhq-1024x1024.pkl',
        'FFHQ-U-256-R': 'stylegan3-r-ffhqu-256x256.pkl',
        'FFHQ-U-1024-R': 'stylegan3-r-ffhqu-1024x1024.pkl',
        'MetFaces-1024-R': 'stylegan3-r-metfaces-1024x1024.pkl',
        'MetFaces-U-1024-R': 'stylegan3-r-metfacesu-1024x1024.pkl',
        'AFHQv2-512-T': 'stylegan3-t-afhqv2-512x512.pkl',
        'FFHQ-1024-T': 'stylegan3-t-ffhq-1024x1024.pkl',
        'FFHQ-U-256-T': 'stylegan3-t-ffhqu-256x256.pkl',
        'FFHQ-U-1024-T': 'stylegan3-t-ffhqu-1024x1024.pkl',
        'MetFaces-1024-T': 'stylegan3-t-metfaces-1024x1024.pkl',
        'MetFaces-U-1024-T': 'stylegan3-t-metfacesu-1024x1024.pkl',
    }

    model_dict = {
        name: load_model(file_name, device)
        for name, file_name in model_names.items()
    }

    func = functools.partial(generate_image,
                             model_dict=model_dict,
                             device=device)
    func = functools.update_wrapper(func, generate_image)

    gr.Interface(
        func,
        [
            gr.inputs.Radio(list(model_names.keys()),
                            type='value',
                            default='FFHQ-1024-R',
                            label='Model'),
            gr.inputs.Number(default=0, label='Seed'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi'),
            gr.inputs.Slider(-1, 1, step=0.05, default=0, label='Translate X'),
            gr.inputs.Slider(-1, 1, step=0.05, default=0, label='Translate Y'),
            gr.inputs.Slider(-180, 180, step=5, default=0, label='Angle'),
        ],
        gr.outputs.Image(type='numpy', label='Output'),
        theme=args.theme,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
