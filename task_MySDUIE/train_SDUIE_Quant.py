import os
import random

os.environ['WANDB_MODE'] = 'dryrun'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gc
import lpips
import numpy as np
import torchvision.transforms.functional as TTF
import torch.nn.functional as TNF
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
import wandb

import sys
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig

p = "src/"
sys.path.append(p)
import torch.nn as nn
import copy

from diffusers import DDPMScheduler
import os
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms

torch.set_num_threads(2)


def parse_args_paired_training(input_args=None):
    parser = argparse.ArgumentParser()
    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=440, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo",
                        help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None, )
    parser.add_argument("--variant", type=str, default=None, )
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=1000000, )
    parser.add_argument("--checkpointing_steps", type=int, default=500, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0, )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help=(
                            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
                        ),
                        )
    parser.add_argument("--report_to", type=str, default="wandb",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )

    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--stage", type=str)
    parser.add_argument("--train_hazy_dir", type=str)
    parser.add_argument("--train_clear_dir", type=str)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.constant_(m.weight, 0.00001)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class SkipConv(torch.nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SkipConv, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(ch_in, ch_in, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        )

    def forward(self, x):
        return self.model(x)


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)

    feature_map = None
    if not self.ignore_skip:
        # up
        for idx, up_block in enumerate(self.up_blocks):
            # add skip
            sample = sample
            sample = up_block(sample, latent_embeds)

    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    return sample


def my_vae_decode(self, z):
    if self.post_quant_conv is not None:
        z = self.post_quant_conv(z)
    dec = self.decoder(z)
    return dec


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8,
                 lora_rank_vae=4, stage=None):
        super().__init__()

        self.stage = stage

        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        vae.decode = my_vae_decode.__get__(vae, vae.__class__)
        vae.decoder.skip_conv_1 = SkipConv(512, 512).cuda()
        vae.decoder.skip_conv_2 = SkipConv(256, 512).cuda()
        vae.decoder.skip_conv_3 = SkipConv(128, 512).cuda()
        vae.decoder.skip_conv_4 = SkipConv(128, 256).cuda()
        vae.decoder.ignore_skip = True

        vae_B = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae_B.encoder.forward = my_vae_encoder_fwd.__get__(vae_B.encoder, vae_B.encoder.__class__)
        vae_B.decoder.forward = my_vae_decoder_fwd.__get__(vae_B.decoder, vae_B.decoder.__class__)
        vae_B.decode = my_vae_decode.__get__(vae_B, vae_B.__class__)
        vae_B.decoder.skip_conv_1 = SkipConv(512, 512).cuda()
        vae_B.decoder.skip_conv_2 = SkipConv(256, 512).cuda()
        vae_B.decoder.skip_conv_3 = SkipConv(128, 512).cuda()
        vae_B.decoder.skip_conv_4 = SkipConv(128, 256).cuda()
        vae_B.decoder.ignore_skip = True

        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        self.lora_rank_vae = lora_rank_vae

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                          target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",
                                         target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            # VAE_B的加载
            vae_lora_config_B = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian",
                                         target_modules=sd["vae_lora_target_modules"])
            vae_B.add_adapter(vae_lora_config_B, adapter_name="vae_skip")
            _sd_vae_B = vae_B.state_dict()
            for k in sd["state_dict_vae_B"]:
                _sd_vae_B[k] = sd["state_dict_vae_B"][k]
            vae_B.load_state_dict(_sd_vae_B)

            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        if pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            vae.decoder.skip_conv_1.apply(init_weights)
            vae.decoder.skip_conv_2.apply(init_weights)
            vae.decoder.skip_conv_3.apply(init_weights)
            vae.decoder.skip_conv_4.apply(init_weights)

            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                  # "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                                  "to_k", "to_q", "to_v", "to_out.0",
                                  ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                                         target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

            target_modules_vae_B = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                    # "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                                    "to_k", "to_q", "to_v", "to_out.0",
                                    ]
            vae_lora_config_B = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                                           target_modules=target_modules_vae_B)
            vae_B.add_adapter(vae_lora_config_B, adapter_name="vae_skip")

            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                                          target_modules=target_modules_unet
                                          )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        vae_B.to("cuda")
        self.unet, self.vae, self.vae_B = unet, vae, vae_B
        self.vae.decoder.gamma = 1
        self.vae.decoder.stage = stage
        # self.vae.decoder.generator = Generator(in_ch=256, num_res=6).cuda()

        self.vae_B.decoder.gamma = 1
        self.vae_B.decoder.stage = stage

        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.vae_B.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vae_B.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.vae_B.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

        for n, _p in self.vae_B.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

        self.vae_B.decoder.skip_conv_1.requires_grad_(True)
        self.vae_B.decoder.skip_conv_2.requires_grad_(True)
        self.vae_B.decoder.skip_conv_3.requires_grad_(True)
        self.vae_B.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t, ct_B=None, if_rec=None, prompt=None, prompt_tokens=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        if if_rec:
            if ct_B == None:
                # 进行A的重建
                encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
                model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, ).sample
                x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
                x_denoised = x_denoised.to(model_pred.dtype)
                self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
                output = self.vae.decode(x_denoised / self.vae.config.scaling_factor)
                output = output.clamp(-1, 1)
                return output
            else:
                # 进行B的重建
                encoded_control = self.vae_B.encode(ct_B).latent_dist.sample() * self.vae_B.config.scaling_factor
                model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, ).sample
                x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
                x_denoised = x_denoised.to(model_pred.dtype)
                self.vae_B.decoder.incoming_skip_acts = self.vae_B.encoder.current_down_blocks
                output_B = self.vae_B.decode(x_denoised / self.vae_B.config.scaling_factor)
                output_B = output_B.clamp(-1, 1)
                return output_B
        else:
            if ct_B == None:
                encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
                model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, ).sample
                x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
                x_denoised = x_denoised.to(model_pred.dtype)
                self.vae_B.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
                output = self.vae_B.decode(x_denoised / self.vae_B.config.scaling_factor)

                # 添加残差连接
                # output = output + (c_t * 2 - 1) / 3
                output = output.clamp(-1, 1)

                return output

            elif c_t == None:
                encoded_control = self.vae_B.encode(ct_B).latent_dist.sample() * self.vae_B.config.scaling_factor
                model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, ).sample
                x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
                x_denoised = x_denoised.to(model_pred.dtype)
                self.vae.decoder.incoming_skip_acts = self.vae_B.encoder.current_down_blocks
                output = self.vae.decode(x_denoised / self.vae.config.scaling_factor)
                output = output.clamp(-1, 1)

                return output

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        sd["state_dict_vae_B"] = {k: v for k, v in self.vae_B.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])

    elif image_prep == "resize_256_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])

    elif image_prep == "resize_128":
        T = transforms.Compose([
            transforms.Resize((128, 128), interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
        ])

    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_320"]:
        T = transforms.Compose([
            transforms.Resize((320, 320), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, train_hazy_dir, train_clear_dir, split, image_prep, tokenizer, mode):
        super().__init__()
        self.split = split
        # 💡 新增：直接定义固定提示词
        self.fixed_caption = "a clear image"

        if split == "train":
            self.input_folder = os.path.join(dataset_folder, train_hazy_dir)
            self.output_folder = os.path.join(dataset_folder, train_clear_dir)
            self.input_folder_unpair = os.path.join(dataset_folder, "train_A_unpair")
            self.output_folder_unpair = os.path.join(dataset_folder, "train_B_unpair")
            # 💡 删除了所有 captions_X = ... 的 json 路径定义
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            # 💡 同样删除了 test 的 json 路径定义

            # 💡 删掉了所有 with open(captions_X, "r") 逻辑

        if self.split == "train":
            # 💡 新增：通过扫描非成对文件夹获取文件名列表，用 sorted 排序
            self.img_names_A_unpair = sorted(os.listdir(self.input_folder_unpair))
            self.img_names_B_unpair = sorted(os.listdir(self.output_folder_unpair))

        # 💡 新增：通过扫描基础文件夹获取成对的文件名列表
        self.img_names_A = sorted(os.listdir(self.input_folder))
        self.img_names_B = sorted(os.listdir(self.output_folder))

        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

        self.mode = mode

    def __len__(self):
        return len(self.img_names_A)

    def __getitem__(self, idx):

        img_name_A = self.img_names_A[idx]
        # 是否读取成对数据
        # v = random.randint(1, len(self.img_names_B) - 1)
        img_name_B = self.img_names_B[idx]

        if self.split == "train":
            # 读取非成对的B
            v = random.randint(0, len(self.img_names_B_unpair) - 1)
            img_name_B_unpair = self.img_names_B_unpair[v]
            output_img_unpair = Image.open(os.path.join(self.output_folder_unpair, img_name_B_unpair))
            output_t_unpair = self.T(output_img_unpair)
            output_t_unpair = TTF.to_tensor(output_t_unpair)
            output_t_unpair = TTF.normalize(output_t_unpair, mean=[0.5], std=[0.5])

            v = random.randint(0, len(self.img_names_A_unpair) - 1)
            img_t_unpair = self.img_names_A_unpair[v]
            img_t_unpair = Image.open(os.path.join(self.input_folder_unpair, img_t_unpair))
            img_t_unpair = self.T(img_t_unpair)
            img_t_unpair = TTF.to_tensor(img_t_unpair)

        input_img = Image.open(os.path.join(self.input_folder, img_name_A))
        img_t = self.T(input_img)
        img_t = TTF.to_tensor(img_t)

        output_img = Image.open(os.path.join(self.output_folder, img_name_B))
        output_t = self.T(output_img)
        output_t = TTF.to_tensor(output_t)
        output_t = TTF.normalize(output_t, mean=[0.5], std=[0.5])

        # 修改：直接使用固定的字符串
        caption = self.fixed_caption

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        if self.mode == "train":
            return {
                "output_pixel_values": output_t,
                "a_unpair":img_t_unpair,
                "b_unpair": output_t_unpair,
                "conditioning_pixel_values": img_t,
                "caption": caption,
                "input_ids": input_ids
            }
        else:
            return {
                "output_pixel_values": output_t,
                "conditioning_pixel_values": img_t,
                "caption": caption,
                "input_ids": input_ids
            }


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        if args.stage == "S1":
            net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
                                        stage=args.stage)
            # net_pix2pix = Pix2Pix_Turbo(pretrained_name=None, pretrained_path=args.model_path, stage="S1")
            # net_pix2pix = Pix2Pix_Turbo(pretrained_name="", pretrained_path=args.pretrained_path, stage=args.stage)
        elif args.stage == "S2":
            # net_pix2pix = Pix2Pix_Turbo(pretrained_name=None, pretrained_path=None, stage="S2")
            # net_pix2pix.load_state_dict(torch.load(args.model_path, map_location="cuda"))
            # net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
            #                             stage=args.stage)
            pass

        else:
            raise ValueError("Wrong stage !!!!!")
        net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_distorted = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type,
                                                             device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_disc_distorted = net_disc_distorted.cuda()
    net_disc_distorted.requires_grad_(True)
    net_disc_distorted.cv_ensemble.requires_grad_(False)
    net_disc_distorted.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    for n, _p in net_pix2pix.vae_B.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    #
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
                    list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
                    list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
                    list(net_pix2pix.vae.decoder.skip_conv_4.parameters())
    #
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae_B.decoder.skip_conv_1.parameters()) + \
                    list(net_pix2pix.vae_B.decoder.skip_conv_2.parameters()) + \
                    list(net_pix2pix.vae_B.decoder.skip_conv_3.parameters()) + \
                    list(net_pix2pix.vae_B.decoder.skip_conv_4.parameters())

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                  eps=args.adam_epsilon, )
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes,
                                 num_cycles=args.lr_num_cycles, power=args.lr_power, )

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
                                       betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                       eps=args.adam_epsilon, )
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)

    optimizer_disc_distorted = torch.optim.AdamW(net_disc_distorted.parameters(), lr=args.learning_rate,
                                                 betas=(args.adam_beta1, args.adam_beta2),
                                                 weight_decay=args.adam_weight_decay,
                                                 eps=args.adam_epsilon, )
    lr_scheduler_disc_distorted = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc_distorted,
                                                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                                num_training_steps=args.max_train_steps * accelerator.num_processes,
                                                num_cycles=args.lr_num_cycles, power=args.lr_power)

    dataset_train = PairedDataset(dataset_folder=args.dataset_folder,
                                  train_hazy_dir=args.train_hazy_dir, train_clear_dir=args.train_clear_dir,
                                  image_prep=args.train_image_prep, split="train", tokenizer=net_pix2pix.tokenizer,
                                  mode="train")
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                           num_workers=args.dataloader_num_workers)
    dataset_val = PairedDataset(dataset_folder=args.dataset_folder,
                                train_hazy_dir=None, train_clear_dir=None,
                                image_prep=args.test_image_prep, split="test", tokenizer=net_pix2pix.tokenizer,
                                mode="val")
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, optimizer_disc_distorted, dl_train, lr_scheduler, lr_scheduler_disc, lr_scheduler_disc_distorted = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, optimizer_disc_distorted, dl_train, lr_scheduler,
        lr_scheduler_disc, lr_scheduler_disc_distorted
    )
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_disc_distorted.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
                        disable=not accelerator.is_local_main_process, )

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for name, module in net_disc_distorted.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # Hyperparameters
    Basic_Skip = 100
    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):

        for step, batch in enumerate(dl_train):

            l_acc = [net_pix2pix, net_disc, net_disc_distorted]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                x_src_unpair = batch["a_unpair"]
                # x_tgt_unpair = batch["b_unpair"]

                x_tgt_as_input = (x_tgt + 1) / 2
                # x_tgt_unpair_as_input = (x_tgt_unpair + 1) / 2
                x_src_unpair_as_label = x_src_unpair * 2 - 1
                x_src_as_label = x_src * 2 - 1

                B, C, H, W = x_src.shape
                # forward pass

                # 有监督去雾损失
                distorted_to_clear = net_pix2pix(x_src, prompt_tokens=batch["input_ids"])
                loss_dehazing_pixel_wise = TNF.mse_loss(distorted_to_clear.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                accelerator.backward(loss_dehazing_pixel_wise, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # 自我重建损失，成对数据的hazy
                output = net_pix2pix(x_src, if_rec=True, prompt_tokens=batch["input_ids"])
                loss = TNF.mse_loss(output.float(), x_src_as_label.float(), reduction="mean") * args.lambda_l2
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # 自我重建损失，非成对数据hazy
                output_A_unpair = net_pix2pix(c_t=x_src_unpair, if_rec=True, prompt_tokens=batch["input_ids"])
                loss = TNF.mse_loss(output_A_unpair.float(), x_src_unpair_as_label.float(), reduction="mean") * args.lambda_l2
                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                if global_step < Basic_Skip:
                    # 自我重建损失，成对数据clear
                    output_B = net_pix2pix(c_t=None, ct_B=x_tgt_as_input, if_rec=True, prompt_tokens=batch["input_ids"])
                    loss = TNF.mse_loss(output_B.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                    accelerator.backward(loss, retain_graph=False)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Generator loss: fool the discriminator
                """
                # 交叉判别损失：去雾过程
                distorted_to_clear = net_pix2pix(x_src, prompt_tokens=batch["input_ids"])
                lossG = net_disc(distorted_to_clear, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # 重建判别损失：成对clear图像
                if global_step < Basic_Skip:
                    output_B = net_pix2pix(c_t=None, ct_B=x_tgt_as_input, if_rec=True, prompt_tokens=batch["input_ids"])
                    lossG = net_disc(output_B, for_G=True).mean() * args.lambda_gan
                    accelerator.backward(lossG)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # 重建判别损失：成对hazy图像
                output_distorted = net_pix2pix(x_src, if_rec=True, prompt_tokens=batch["input_ids"])
                lossG = net_disc_distorted(output_distorted, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # 重建判别损失：非成对hazy图像
                output_A_unpair = net_pix2pix(c_t=x_src_unpair, ct_B=None, if_rec=True,
                                              prompt_tokens=batch["input_ids"])
                lossG = net_disc_distorted(output_A_unpair, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """
                # 交叉判别损失：去雾过程
                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc(distorted_to_clear.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                # 重建判别损失：成对hazy图像
                # real image
                lossD_real = net_disc_distorted(x_src.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc_distorted.parameters(), args.max_grad_norm)
                optimizer_disc_distorted.step()
                lr_scheduler_disc_distorted.step()
                optimizer_disc_distorted.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc_distorted(output_distorted.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc_distorted.parameters(), args.max_grad_norm)
                optimizer_disc_distorted.step()
                optimizer_disc_distorted.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake

                # 重建判别损失：非成对hazy图像
                # real image
                lossD_real = net_disc_distorted(x_src_unpair_as_label.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc_distorted.parameters(), args.max_grad_norm)
                optimizer_disc_distorted.step()
                lr_scheduler_disc_distorted.step()
                optimizer_disc_distorted.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc_distorted(output_A_unpair.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc_distorted.parameters(), args.max_grad_norm)
                optimizer_disc_distorted.step()
                optimizer_disc_distorted.zero_grad(set_to_none=args.set_grads_to_none)

                if global_step < Basic_Skip:
                    # 重建判别损失：clear图像
                    # real image
                    lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                    accelerator.backward(lossD_real.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    lr_scheduler_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                    # fake image
                    lossD_fake = net_disc(output_B.detach(), for_real=False).mean() * args.lambda_gan
                    accelerator.backward(lossD_fake.mean())
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss.detach().item()
                    # logs["loss_lpips"] = loss_lpips.detach().item()

                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1 and args.stage == "S1":
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        if args.track_val_fid:
                            pass
                            # os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            x_tgt_as_input = (x_tgt + 1) / 2

                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():

                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(
                                    net_pix2pix)(x_src, prompt_tokens=batch_val["input_ids"].cuda())

                                rec = accelerator.unwrap_model(
                                    net_pix2pix)(x_src, if_rec=True, prompt_tokens=batch_val["input_ids"].cuda())

                                rec_B = accelerator.unwrap_model(
                                    net_pix2pix)(c_t=None, ct_B=x_tgt_as_input, if_rec=True, prompt_tokens=batch_val["input_ids"].cuda())

                                x_src = x_src * 2 - 1
                                x_tgt_pred = torch.concatenate(tensors=[x_src, x_tgt,
                                                                        x_tgt_pred, rec, rec_B], dim=3)

                                l_l2.append(loss.item())
                                # l_lpips.append(loss_lpips.item())

                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                # outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                outf = os.path.join(args.output_dir, "eval", f"val_{global_step}_" + str(step) + ".jpg")
                                output_pil.save(outf)

                        logs["val/l2"] = np.mean(l_l2)
                        # logs["val/lpips"] = np.mean(l_lpips)
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
