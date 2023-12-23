from functools import partial
import os

import torch
import numpy as np
import gradio as gr
import gdown


WEBSITE = """
<div class="embed_hidden">
<h1 style='text-align: center'> MoMask: Generative Masked Modeling of 3D Human Motions </h1>
<h2 style='text-align: center'>
<a href="https://ericguo5513.github.io" target="_blank"><nobr>Chuan Guo*</nobr></a> &emsp;
<a href="https://yxmu.foo/" target="_blank"><nobr>Yuxuan Mu*</nobr></a> &emsp;
<a href="https://scholar.google.com/citations?user=w4e-j9sAAAAJ&hl=en" target="_blank"><nobr>Muhammad Gohar Javed*</nobr></a> &emsp;
<a href="https://sites.google.com/site/senwang1312home/" target="_blank"><nobr>Sen Wang</nobr></a> &emsp;
<a href="https://www.ece.ualberta.ca/~lcheng5/" target="_blank"><nobr>Li Cheng</nobr></a>
</h2>
<h2 style='text-align: center'>
<nobr>arXiv 2023</nobr>
</h2>
<h3 style="text-align:center;">
<a target="_blank" href="https://arxiv.org/abs/2312.00063"> <button type="button" class="btn btn-primary btn-lg"> Paper </button></a> &ensp;
<a target="_blank" href="https://github.com/EricGuo5513/momask-codes"> <button type="button" class="btn btn-primary btn-lg"> Code </button></a> &ensp;
<a target="_blank" href="https://ericguo5513.github.io/momask/"> <button type="button" class="btn btn-primary btn-lg"> Webpage </button></a> &ensp;
<a target="_blank" href="https://ericguo5513.github.io/source_files/momask_2023_bib.txt"> <button type="button" class="btn btn-primary btn-lg"> BibTex </button></a>
</h3>
<h3> Description </h3>
<p>
This space illustrates <a href='https://ericguo5513.github.io/momask/' target='_blank'><b>MoMask</b></a>, a method for text-to-motion generation.
</p>
</div>
"""

EXAMPLES = [
    "A person is walking slowly",
    "A person is walking in a circle",
    "A person is jumping rope",
    "Someone is doing a backflip",
    "A person is doing a moonwalk",
    "A person walks forward and then turns back",
    "Picking up an object",
    "A person is swimming in the sea",
    "A human is squatting",
    "Someone is jumping with one foot",
    "A person is chopping vegetables",
    "Someone walks backward",
    "Somebody is ascending a staircase",
    "A person is sitting down",
    "A person is taking the stairs",
    "Someone is doing jumping jacks",
    "The person walked forward and is picking up his toolbox",
    "The person angrily punching the air",
]

# Show closest text in the training


# css to make videos look nice
# var(--block-border-color); TODO
CSS = """
.retrieved_video {
    position: relative;
    margin: 0;
    box-shadow: var(--block-shadow);
    border-width: var(--block-border-width);
    border-color: #000000;
    border-radius: var(--block-radius);
    background: var(--block-background-fill);
    width: 100%;
    line-height: var(--line-sm);
}
}
"""


DEFAULT_TEXT = "A person is "

def generate(
    text, uid, motion_length=0, seed=351540, repeat_times=4,
):
    os.system(f'python gen_t2m.py --gpu_id 0 --seed {seed} --ext {uid} --repeat_times {repeat_times} --motion_length {motion_length} --text_prompt {text}')
    datas = []
    for n in repeat_times:
        data_unit = {
            "url": f"./generation/{uid}/animations/0/sample0_repeat{n}_len196_ik.mp4"
            }
        datas.append(data_unit)
    return datas


# HTML component
def get_video_html(data, video_id, width=700, height=700):
    url = data["url"]
    # class="wrap default svelte-gjihhp hide"
    # <div class="contour_video" style="position: absolute; padding: 10px;">
    # width="{width}" height="{height}"
    video_html = f"""
<video class="retrieved_video" width="{width}" height="{height}" preload="auto" muted playsinline onpause="this.load()"
autoplay loop disablepictureinpicture id="{video_id}">
  <source src="{url}" type="video/mp4">
  Your browser does not support the video tag.
</video>
"""
    return video_html


def generate_component(generate_function, text):
    if text == DEFAULT_TEXT or text == "" or text is None:
        return [None for _ in range(4)]

    datas = generate_function(text, )
    htmls = [get_video_html(data, idx) for idx, data in enumerate(datas)]
    return htmls


if not os.path.exists("checkpoints/t2m"):
    os.system("bash prepare/download_models_demo.sh")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOADING

# DEMO
theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")
generate_and_show = partial(generate_component, generate)

with gr.Blocks(css=CSS, theme=theme) as demo:
    gr.Markdown(WEBSITE)
    videos = []

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Column(scale=2):
                text = gr.Textbox(
                    show_label=True,
                    label="Text prompt",
                    value=DEFAULT_TEXT,
                )
            with gr.Column(scale=1):
                gen_btn = gr.Button("Generate", variant="primary")
                clear = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=2):

            def generate_example(text):
                return generate_and_show(text)

            examples = gr.Examples(
                examples=[[x, None, None] for x in EXAMPLES],
                inputs=[text],
                examples_per_page=20,
                run_on_click=False,
                cache_examples=False,
                fn=generate_example,
                outputs=[],
            )

    i = -1
    # should indent
    for _ in range(1):
        with gr.Row():
            for _ in range(4):
                i += 1
                video = gr.HTML()
                videos.append(video)

    # connect the examples to the output
    # a bit hacky
    examples.outputs = videos

    def load_example(example_id):
        processed_example = examples.non_none_processed_examples[example_id]
        return gr.utils.resolve_singleton(processed_example)

    examples.dataset.click(
        load_example,
        inputs=[examples.dataset],
        outputs=examples.inputs_with_examples,  # type: ignore
        show_progress=False,
        postprocess=False,
        queue=False,
    ).then(fn=generate_example, inputs=examples.inputs, outputs=videos)

    gen_btn.click(
        fn=generate_and_show,
        inputs=[text],
        outputs=videos,
    )
    text.submit(
        fn=generate_and_show,
        inputs=[text],
        outputs=videos,
    )

    def clear_videos():
        return [None for x in range(4)] + [DEFAULT_TEXT]

    clear.click(fn=clear_videos, outputs=videos + [text])

demo.launch(share=True)
