from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

current_dir = pathlib.Path(__file__).parent
submodule_dir = current_dir / "stylegan3"
sys.path.insert(0, submodule_dir.as_posix())


class Model:
    MODEL_NAME_DICT = {
        "AFHQv2-512-R": "stylegan3-r-afhqv2-512x512.pkl",
        "FFHQ-1024-R": "stylegan3-r-ffhq-1024x1024.pkl",
        "FFHQ-U-256-R": "stylegan3-r-ffhqu-256x256.pkl",
        "FFHQ-U-1024-R": "stylegan3-r-ffhqu-1024x1024.pkl",
        "MetFaces-1024-R": "stylegan3-r-metfaces-1024x1024.pkl",
        "MetFaces-U-1024-R": "stylegan3-r-metfacesu-1024x1024.pkl",
        "AFHQv2-512-T": "stylegan3-t-afhqv2-512x512.pkl",
        "FFHQ-1024-T": "stylegan3-t-ffhq-1024x1024.pkl",
        "FFHQ-U-256-T": "stylegan3-t-ffhqu-256x256.pkl",
        "FFHQ-U-1024-T": "stylegan3-t-ffhqu-1024x1024.pkl",
        "MetFaces-1024-T": "stylegan3-t-metfaces-1024x1024.pkl",
        "MetFaces-U-1024-T": "stylegan3-t-metfacesu-1024x1024.pkl",
    }

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._download_all_models()
        self.model_name = "FFHQ-1024-R"
        self.model = self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> nn.Module:
        file_name = self.MODEL_NAME_DICT[model_name]
        path = hf_hub_download("hysts/StyleGAN3", f"models/{file_name}")
        with open(path, "rb") as f:
            model = pickle.load(f)["G_ema"]
        model.eval()
        model.to(self.device)
        return model

    def set_model(self, model_name: str) -> None:
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _download_all_models(self):
        for name in self.MODEL_NAME_DICT.keys():
            self._load_model(name)

    @staticmethod
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

    def generate_z(self, seed: int) -> torch.Tensor:
        seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
        z = np.random.RandomState(seed).randn(1, self.model.z_dim)
        return torch.from_numpy(z).float().to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return tensor.cpu().numpy()

    def set_transform(self, tx: float, ty: float, angle: float) -> None:
        mat = self.make_transform((tx, ty), angle)
        mat = np.linalg.inv(mat)
        self.model.synthesis.input.transform.copy_(torch.from_numpy(mat))

    @torch.inference_mode()
    def generate(self, z: torch.Tensor, label: torch.Tensor, truncation_psi: float) -> torch.Tensor:
        return self.model(z, label, truncation_psi=truncation_psi)

    def generate_image(self, seed: int, truncation_psi: float, tx: float, ty: float, angle: float) -> np.ndarray:
        self.set_transform(tx, ty, angle)

        z = self.generate_z(seed)
        label = torch.zeros([1, self.model.c_dim], device=self.device)

        out = self.generate(z, label, truncation_psi)
        out = self.postprocess(out)
        return out[0]

    def set_model_and_generate_image(
        self, model_name: str, seed: int, truncation_psi: float, tx: float, ty: float, angle: float
    ) -> np.ndarray:
        self.set_model(model_name)
        return self.generate_image(seed, truncation_psi, tx, ty, angle)
