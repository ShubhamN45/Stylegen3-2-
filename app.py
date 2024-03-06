#!/usr/bin/env python

from __future__ import annotations

import gradio as gr
import numpy as np

from model import Model

DESCRIPTION = "# [StyleGAN3](https://github.com/NVlabs/stylegan3)"


def get_sample_image_url(name: str) -> str:
    sample_image_dir = "https://huggingface.co/spaces/hysts/StyleGAN3/resolve/main/samples"
    return f"{sample_image_dir}/{name}.jpg"


def get_sample_image_markdown(name: str) -> str:
    url = get_sample_image_url(name)
    size = 512 if name == "afhqv2" else 1024
    seed = "0-99"
    return f"""
    - size: {size}x{size}
    - seed: {seed}
    - truncation: 0.7
    ![sample images]({url})"""


model = Model()

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        with gr.TabItem("App"):
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(
                        label="Model", choices=list(model.MODEL_NAME_DICT.keys()), value="FFHQ-1024-R"
                    )
                    seed = gr.Slider(label="Seed", minimum=0, maximum=np.iinfo(np.uint32).max, step=1, value=0)
                    psi = gr.Slider(label="Truncation psi", minimum=0, maximum=2, step=0.05, value=0.7)
                    tx = gr.Slider(label="Translate X", minimum=-1, maximum=1, step=0.05, value=0)
                    ty = gr.Slider(label="Translate Y", minimum=-1, maximum=1, step=0.05, value=0)
                    angle = gr.Slider(label="Angle", minimum=-180, maximum=180, step=5, value=0)
                    run_button = gr.Button()
                with gr.Column():
                    result = gr.Image(label="Result", elem_id="result")

        with gr.TabItem("Sample Images"):
            with gr.Row():
                model_name2 = gr.Dropdown(
                    [
                        "afhqv2",
                        "ffhq",
                        "ffhq-u",
                        "metfaces",
                        "metfaces-u",
                    ],
                    value="afhqv2",
                    label="Model",
                )
            with gr.Row():
                text = get_sample_image_markdown(model_name2.value)
                sample_images = gr.Markdown(text)

    run_button.click(
        fn=model.set_model_and_generate_image,
        inputs=[
            model_name,
            seed,
            psi,
            tx,
            ty,
            angle,
        ],
        outputs=result,
        api_name="run",
    )
    model_name2.change(
        fn=get_sample_image_markdown,
        inputs=model_name2,
        outputs=sample_images,
        queue=False,
        api_name=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
