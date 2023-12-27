from functools import partial
import os

import torch
import numpy as np
import gradio as gr
import random
import shutil

print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

import os
from os.path import join as pjoin

import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator

from options.hgdemo_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

from gen_t2m import load_vq_model, load_res_model, load_trans_model, load_len_estimator

clip_version = 'ViT-B/32'

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
🔥🔥🔥 This space presents an interactive demo for <a href='https://ericguo5513.github.io/momask/' target='_blank'><b>MoMask</b></a>, a method for text-to-motion generation!!! It generates human motions (skeletal animations) based on your descriptions. To gain a better understanding of our work, you could try the provided examples first. 🔥🔥🔥
</p>
<p>
🚀🚀🚀 In addition, we provide a link to download the generated human skeletal motion in <b>BVH</b> file format, compatible with CG software such as Blender!!! 🚀🚀🚀
</p>
<p>
😁😁😁 If you find this demo interesting, we would appreciate your star on our <a href="https://github.com/EricGuo5513/momask-codes" target="_blank">github</a>. More details could be found on our <a href='https://ericguo5513.github.io/momask/' target='_blank'>webpage</a>. 🫶🫶🫶
</p>
<p>
If you have any issues on this space or feature requests, we warmly welcome you to contact us through our <a href="https://github.com/EricGuo5513/momask-codes/issues" target="_blank">github repo</a> or <a href="mailto:ymu3@ualberta.ca?subject =[MoMask]Feedback&body = Message">email</a>.
</p>
</div>
"""
WEBSITE_bottom = """
<p>
We thanks <a href="https://huggingface.co/spaces/Mathux/TMR" target="_blank">TMR</a> for this cool space template.
</p>
</div>
"""

EXAMPLES = [
   "A person is running on a treadmill.", "The person did a kick spin to the left.",
   "The person takes 4 steps backwards.", "A person jumps up and then lands.",
   "The person was pushed but did not fall.", "The person does a salsa dance.",
   "This person kicks with his right leg then jabs several times.",
   "A person stands for few seconds and picks up his arms and shakes them.",
   "A person walks in a clockwise circle and stops where he began.",
   "A man bends down and picks something up with his right hand.",
   "A person walks with a limp, their left leg gets injured.",
   "A person repeatedly blocks their face with their right arm.",
   "A figure streches it hands and arms above its head.",
   "The person holds his left foot with his left hand, puts his right foot up and left hand up too.",
   "A person stands, crosses left leg in front of the right, lowering themselves until they are sitting, both hands on the floor before standing and uncrossing legs.",
   "The man walked forward, spun right on one foot and walked back to his original position.",
   "A man is walking forward then steps over an object then continues walking forward.",
   "This person takes 4 steps forward staring with his right foot.",
]

# Show closest text in the training


# css to make videos look nice
# var(--block-border-color); TODO
CSS = """
.generate_video {
    position: relative;
    margin-left: auto;
    margin-right: auto;
    box-shadow: var(--block-shadow);
    border-width: var(--block-border-width);
    border-color: #000000;
    border-radius: var(--block-radius);
    background: var(--block-background-fill);
    width: 25%;
    line-height: var(--line-sm);
}
}
"""


DEFAULT_TEXT = "A person is "


if not os.path.exists("checkpoints/t2m"):
    os.system("bash prepare/download_models_demo.sh")

##########################
######Preparing demo######
##########################
parser = EvalT2MOptions()
opt = parser.parse()
fixseed(opt.seed)
opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
dim_pose = 263
root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
model_dir = pjoin(root_dir, 'model')
model_opt_path = pjoin(root_dir, 'opt.txt')
model_opt = get_opt(model_opt_path, device=opt.device)

######Loading RVQ######
vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
vq_opt = get_opt(vq_opt_path, device=opt.device)
vq_opt.dim_pose = dim_pose
vq_model, vq_opt = load_vq_model(vq_opt)

model_opt.num_tokens = vq_opt.nb_code
model_opt.num_quantizers = vq_opt.num_quantizers
model_opt.code_dim = vq_opt.code_dim

######Loading R-Transformer######
res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
res_opt = get_opt(res_opt_path, device=opt.device)
res_model = load_res_model(res_opt, vq_opt, opt)

assert res_opt.vq_name == model_opt.vq_name

######Loading M-Transformer######
t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

#####Loading Length Predictor#####
length_estimator = load_len_estimator(model_opt)

t2m_transformer.eval()
vq_model.eval()
res_model.eval()
length_estimator.eval()

res_model.to(opt.device)
t2m_transformer.to(opt.device)
vq_model.to(opt.device)
length_estimator.to(opt.device)

opt.nb_joints = 22
mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
def inv_transform(data):
    return data * std + mean

kinematic_chain = t2m_kinematic_chain
converter = Joint2BVHConvertor()
cached_dir = './cached'
uid = 12138
animation_path = pjoin(cached_dir, f'{uid}')
os.makedirs(animation_path, exist_ok=True)

@torch.no_grad()
def generate(
    text, uid, motion_length=0, use_ik=True, seed=10107, repeat_times=1,
):
    # fixseed(seed)
    prompt_list = []
    length_list = []
    est_length = False
    prompt_list.append(text)
    if motion_length == 0:
        est_length = True
    else:
        length_list.append(motion_length)

    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        text_embedding = t2m_transformer.encode_text(prompt_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
        token_lens = Categorical(probs).sample()  # (b, seqlen)
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list
    datas = []
    for r in range(repeat_times):
        mids = t2m_transformer.generate(captions, token_lens,
                                        timesteps=opt.time_steps,
                                        cond_scale=opt.cond_scale,
                                        temperature=opt.temperature,
                                        topk_filter_thres=opt.topkr,
                                        gsample=opt.gumbel_sample)
        mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        pred_motions = vq_model.forward_decoder(mids)
        pred_motions = pred_motions.detach().cpu().numpy()
        data = inv_transform(pred_motions)
        ruid = random.randrange(99999)
        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            animation_path = pjoin(cached_dir, f'{uid}')
            shutil.rmtree(animation_path)
            os.makedirs(animation_path, exist_ok=True)
            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
            bvh_path = pjoin(animation_path, "sample_repeat%d.bvh" % (r))
            save_path = pjoin(animation_path, "sample_repeat%d_%d.mp4"%(r, ruid))
            if use_ik:
                print("Using IK")
                _, joint = converter.convert(joint, filename=bvh_path, iterations=100)
            else:
                _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)
            plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
            np.save(pjoin(animation_path, "sample_repeat%d.npy"%(r)), joint)
        data_unit = {
            "url": pjoin(animation_path, "sample_repeat%d_%d.mp4"%(r, ruid))
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
<h2 style='text-align: center'>
<a href="file/{pjoin(animation_path, "sample_repeat0.bvh")}" download="sample.bvh"><b>BVH Download</b></a>
</h2>
<video class="generate_video" width="{width}" height="{height}" style="center" preload="auto" muted playsinline onpause="this.load()"
autoplay loop disablepictureinpicture id="{video_id}">
  <source src="file/{url}" type="video/mp4">
  Your browser does not support the video tag.
</video>
"""
    return video_html

def generate_component(generate_function, text, motion_len='0', postprocess='IK'):
    if text == DEFAULT_TEXT or text == "" or text is None:
        return [None for _ in range(1)]
    # uid = random.randrange(99999)
    try:
        motion_len = max(0, min(int(float(motion_len) * 20), 196))
    except:
        motion_len = 0
    use_ik = postprocess == 'IK'
    datas = generate_function(text, uid, motion_len, use_ik)
    htmls = [get_video_html(data, idx) for idx, data in enumerate(datas)]
    return htmls


# LOADING

# DEMO
theme = gr.themes.Default(primary_hue="blue", secondary_hue="gray")
generate_and_show = partial(generate_component, generate)

with gr.Blocks(css=CSS, theme=theme) as demo:
    gr.Markdown(WEBSITE)
    videos = []

    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(
                show_label=True,
                label="Text prompt",
                value=DEFAULT_TEXT,
            )
            with gr.Row():
                with gr.Column(scale=1):
                    motion_len = gr.Textbox(
                        show_label=True,
                        label="Motion length (<10s)",
                        value=0,
                        info="Specify the motion length; 0 to use the default auto-setting.",
                    )
                with gr.Column(scale=1):
                    use_ik = gr.Radio(
                        ["Raw", "IK"],
                        label="Post-processing",
                        value="IK",
                        info="Use basic inverse kinematic (IK) for foot contact locking",
                    )
            gen_btn = gr.Button("Generate", variant="primary")
            clear = gr.Button("Clear", variant="secondary")
            gr.Markdown(
                        f"""
                            
                        """
                    )

        with gr.Column(scale=2):

            def generate_example(text):
                return generate_and_show(text)

            examples = gr.Examples(
                examples=[[x, None, None] for x in EXAMPLES],
                inputs=[text],
                examples_per_page=10,
                run_on_click=False,
                cache_examples=False,
                fn=generate_example,
                outputs=[],
            )

    i = -1
    # should indent
    for _ in range(1):
        with gr.Row():
            for _ in range(1):
                i += 1
                video = gr.HTML()
                videos.append(video)
    gr.Markdown(WEBSITE_bottom)
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
        inputs=[text, motion_len, use_ik],
        outputs=videos,
    )
    text.submit(
        fn=generate_and_show,
        inputs=[text, motion_len, use_ik],
        outputs=videos,
    )

    def clear_videos():
        return [None for x in range(1)] + [DEFAULT_TEXT]

    clear.click(fn=clear_videos, outputs=videos + [text])

demo.launch()
