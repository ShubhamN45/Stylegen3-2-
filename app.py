#!/usr/bin/env python

from __future__ import annotations

import argparse

import gradio as gr
import numpy as np

from model import Model

TITLE = '# NVlabs/stylegan3'
DESCRIPTION = '''This is an unofficial demo for [https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan3).

Expected execution time on Hugging Face Spaces: 50s
'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.stylegan3" />'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def get_sample_image_url(name: str) -> str:
    sample_image_dir = 'https://huggingface.co/spaces/hysts/StyleGAN3/resolve/main/samples'
    return f'{sample_image_dir}/{name}.jpg'


def get_sample_image_markdown(name: str) -> str:
    url = get_sample_image_url(name)
    size = 512 if name == 'afhqv2' else 1024
    seed = '0-99'
    return f'''
    - size: {size}x{size}
    - seed: {seed}
    - truncation: 0.7
    ![sample images]({url})'''


def main():
    args = parse_args()
    model = Model(args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        with gr.Tabs():
            with gr.TabItem('App'):
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            model_name = gr.Dropdown(list(
                                model.MODEL_NAME_DICT.keys()),
                                                     value='FFHQ-1024-R',
                                                     label='Model')
                            seed = gr.Slider(0,
                                             np.iinfo(np.uint32).max,
                                             step=1,
                                             value=0,
                                             label='Seed')
                            psi = gr.Slider(0,
                                            2,
                                            step=0.05,
                                            value=0.7,
                                            label='Truncation psi')
                            tx = gr.Slider(-1,
                                           1,
                                           step=0.05,
                                           value=0,
                                           label='Translate X')
                            ty = gr.Slider(-1,
                                           1,
                                           step=0.05,
                                           value=0,
                                           label='Translate Y')
                            angle = gr.Slider(-180,
                                              180,
                                              step=5,
                                              value=0,
                                              label='Angle')
                            run_button = gr.Button('Run')
                    with gr.Column():
                        result = gr.Image(label='Result', elem_id='result')

            with gr.TabItem('Sample Images'):
                with gr.Row():
                    model_name2 = gr.Dropdown([
                        'afhqv2',
                        'ffhq',
                        'ffhq-u',
                        'metfaces',
                        'metfaces-u',
                    ],
                                              value='afhqv2',
                                              label='Model')
                with gr.Row():
                    text = get_sample_image_markdown(model_name2.value)
                    sample_images = gr.Markdown(text)

        gr.Markdown(FOOTER)

        model_name.change(fn=model.set_model, inputs=model_name, outputs=None)
        run_button.click(fn=model.set_model_and_generate_image,
                         inputs=[
                             model_name,
                             seed,
                             psi,
                             tx,
                             ty,
                             angle,
                         ],
                         outputs=result)
        model_name2.change(fn=get_sample_image_markdown,
                           inputs=model_name2,
                           outputs=sample_images)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
