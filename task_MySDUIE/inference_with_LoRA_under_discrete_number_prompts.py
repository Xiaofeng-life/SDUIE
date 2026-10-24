import os
import sys
# p = "/home/congxiaofeng/CXF_Code/img2img-turbo-main/"
# sys.path.append(p)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from diffusers import DDPMScheduler


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


import torch
import math


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

        vae.decoder.ignore_skip = True

        vae_B = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        vae_B.encoder.forward = my_vae_encoder_fwd.__get__(vae_B.encoder, vae_B.encoder.__class__)
        vae_B.decoder.forward = my_vae_decoder_fwd.__get__(vae_B.decoder, vae_B.decoder.__class__)
        vae_B.decode = my_vae_decode.__get__(vae_B, vae_B.__class__)

        vae.decoder = None
        vae_B.encoder = None

        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        self.lora_rank_vae = lora_rank_vae

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")

            # VAE的加载
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

            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian",
                                          target_modules=sd["unet_lora_target_modules"])
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        vae_B.to("cuda")
        self.unet, self.vae, self.vae_B = unet, vae, vae_B
        # self.vae.decoder.gamma = 1
        # self.vae.decoder.stage = stage
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

    def forward(self, c_t, ct_B=None, if_rec=None, prompt=None, prompt_tokens=None):

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, ).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        x_denoised = x_denoised.to(model_pred.dtype)
        self.vae_B.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output = self.vae_B.decode(x_denoised / self.vae_B.config.scaling_factor)
        output = output.clamp(-1, 1)
        return output


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

            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]

            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                  # "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                                  "to_k", "to_q", "to_v", "to_out.0",
                                  ]

            target_modules_vae_B = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                                    # "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                                    "to_k", "to_q", "to_v", "to_out.0",
                                    ]

            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

            print("Loading success !!!!!!!!!!!!!!!!!!!!!" * 5)

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

        if ct_B == None:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc, ).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae_B.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output = self.vae_B.decode(x_denoised / self.vae_B.config.scaling_factor)

            output = output.clamp(-1, 1)

            return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str,  help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--pretrained_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    parser.add_argument("--if_cat", type=str, default="False")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = os.listdir(args.input_dir)
    all_level_names = ["dp00", "dp01", "dp02", "dp03", "dp04", "dp05", "dp06", "dp07", "dp08", "dp09", "dp10"]
    all_level_prompts = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    # initialize the model
    # 加载LoRA模型
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.pretrained_path, stage="S1")

    model.set_eval()
    if args.use_fp16:
        model.half()

    for idx, factor in enumerate(all_level_prompts):
        for file in all_files:
            input_image = Image.open(os.path.join(args.input_dir, file)).convert('RGB')
            # make sure that the input image is a multiple of 8
            # input_image = Image.open(args.input_image).convert('RGB')
            new_width = input_image.width - input_image.width % 8
            new_height = input_image.height - input_image.height % 8
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
            input_image = input_image.resize((512, 512), Image.LANCZOS)
            bname = file

            # translate the image
            with torch.no_grad():
                c_t = F.to_tensor(input_image).unsqueeze(0).cuda()
                if args.use_fp16:
                    c_t = c_t.half()
                output_image = model(c_t, prompt="Enhance this image by {} level.".format(all_level_prompts[idx]))

                output_image = output_image * 0.5 + 0.5
            if args.if_cat == "True":
                output_image = torch.concatenate(tensors=[c_t, output_image], dim=3)

            print("before: ", torch.mean(c_t), ", after: ", torch.mean(output_image))

            output_pil = transforms.ToPILImage()(output_image[0].cpu())
            # output_pil = output_pil.resize((256, 256), Image.LANCZOS)
            # save the output image
            output_pil.save(os.path.join(args.output_dir, bname[:-4] + "_" + all_level_names[idx] + ".jpg"))
