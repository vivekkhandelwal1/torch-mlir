# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch
from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

import math
import sys

from torchvision import utils as tv_utils
from torchvision.transforms import functional as TF
from tqdm.notebook import trange, tqdm
from IPython import display

sys.path.append('/home/vivek/work/02_07/v-diffusion-pytorch')

from CLIP import clip
from diffusion import get_model, sampling, utils

# ==============================================================================

# Load the models

model = get_model('cc12m_1_cfg')()
_, side_y, side_x = model.shape
model.load_state_dict(torch.load('/home/vivek/work/02_07/v-diffusion-pytorch/checkpoints/cc12m_1_cfg.pth', map_location='cpu'))
model = model.half().cuda().eval().requires_grad_(False)
clip_model = clip.load(model.clip_model, jit=False, device='cpu')[0]

# Settings

# The text prompt
prompt = 'New York City, oil on canvas'

#The strength of the text conditioning (0 means don't condition on text, 1 means sample images that match the text about as well as the images match the text captions in the training set, 3+ is recommended).
weight = 5

# Sample this many images.
n_images = 4

# Specify the number of diffusion timesteps (default is 50, can lower for faster but lower quality sampling).
steps = 50

# The random seed. Change this to sample different images.
seed = 0

# Display progress every this many timesteps.
display_every =   10

target_embed = clip_model.encode_text(clip.tokenize(prompt)).float().cuda()

class CC12M1Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_images = n_images
        self.side_y = side_y
        self.side_x = side_x
        self.steps = steps
        torch.manual_seed(seed)

    @export
    @annotate_args([
        None,
    ])

    def cfg_model_fn(self, x, t):
        """The CFG wrapper function."""
        n = x.shape[0]
        x_in = x.repeat([2, 1, 1, 1])
        t_in = t.repeat([2])
        clip_embed_repeat = target_embed.repeat([n, 1])
        clip_embed_in = torch.cat([torch.zeros_like(clip_embed_repeat), clip_embed_repeat])
        v_uncond, v_cond = model(x_in, t_in, clip_embed_in).chunk(2, dim=0)
        v = v_uncond + (v_cond - v_uncond) * weight
        return v


    def display_callback(self, info):
        if info['i'] % display_every == 0:
            nrow = math.ceil(info['pred'].shape[0]**0.5)
            grid = tv_utils.make_grid(info['pred'], nrow, padding=0)
            tqdm.write(f'Step {info["i"]} of {steps}:')
            display.display(utils.to_pil_image(grid))
            tqdm.write(f'')

    def forward(self):
        x = torch.randn([self.n_images, 3, self.side_y, self.side_x], device='cuda')
        t = torch.linspace(1, 0, self.steps + 1, device='cuda')[:-1]
        step_list = utils.get_spliced_ddpm_cosine_schedule(t)
        outs = sampling.plms_sample(self.cfg_model_fn, x, step_list, {}, callback=self.display_callback)
        tqdm.write('Done!')
        for i, out in enumerate(outs):
            filename = f'out_{i}.png'
            utils.to_pil_image(out).save(filename)
            display.display(display.Image(filename))


@register_test_case(module_factory=lambda: CC12M1Model())
def CC12M1Model_basic(module, tu: TestUtils):
    module.forward()
