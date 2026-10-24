# SDUIE: Semi-Supervised Diffusion for Underwater Image Enhancement with Quant-Text Dual Control

This repository contains the official implementation of our CVPR 2026 paper:

## 📖 Abstract
Underwater images often exhibit dominant blue-green hues due to wavelength-dependent light attenuation.
While existing enhancement methods have achieved promising performance, they typically overlook the subjective nature of visual preferences. 
To address this gap, we propose SDUIE, a level-aware Semi-supervised Diffusion framework for Underwater Image Enhancement that enables dual control through both quantitative and textual inputs.
SDUIE-Quant allows continuous, numerical adjustment of enhancement levels via low-rank adaptation weight merging within a dual-branch diffusion model. 
This model comprises a supervised branch trained on synthetic underwater-terrestrial pairs and a self-supervised branch designed to preserve the natural hues of real-world underwater scenes.
Building on this, SDUIE-Text introduces intuitive, language-guided control by aligning semantic prompts with visual enhancement effects, leveraging the learned fusion weights.
This dual-modality design offers both precise control and flexible, user-preferred enhancement. 
Experimental results demonstrate that SDUIE achieves state-of-the-art results while better preserving the aesthetic qualities often missed by conventional methods. 
The source code is in https://github.com/Xiaofeng-life/SDUIE.

## Install Python Dependencies

All required Python packages are listed in the `requirements.txt` file at the project root. Choose the installation method that best fits your environment.

Create and activate a virtual environment in the project root:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux / macOS
   
```

## Training

After installing the Python dependencies (see [Install Python Dependencies](#install-python-dependencies)), you can run the training scripts.
This project uses a two‑stage training pipeline:

1. **Stage 1 – Quantization‑aware training (`train_SDUIE_Quant.py`)**  
This stage trains the model with quantization awareness. It is the first step and produces a checkpoint file (e.g., `model_10001.pkl`).

2. **Stage 2 – Discrete text‑guided training (`train_SDUIE_Text.py`)**  
   This stage fine‑tunes the model using discrete text guidance. It **depends on the checkpoint** generated in Stage 1.


### Dataset Preparation
Place your training data under the folders specified by `--dataset_folder` and `--train_hazy_dir` / `--train_clear_dir`.

Dataset link: [Google Drive](https://drive.google.com/file/d/1pNFAO8TkiuylJychgibnvhb_fdELet7f/view?usp=drive_link)

### Running Stage 1 – Quantization‑aware Training
```angular2html
CUDA_VISIBLE_DEVICES=0 python task_MySDUIE/train_SDUIE_Quant.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="results/MySDUIE/train_SDUIE_Quant" \
    --dataset_folder="results/MySDUIE/data/V15_3" \
    --learning_rate 0.0001 \
    --resolution=512 \
    --train_batch_size=1 \
    --train_image_prep "resize_320" \
    --test_image_prep "resize_320" \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --eval_freq 1000 --num_training_epochs 1000 --num_samples_eval 200 \
    --pretrained_path "" \
    --stage S1 \
    --train_hazy_dir "train_A" --train_clear_dir "train_B"
```

If the GPU is 4090, run the code
```angular2html
export NCCL_P2P_DISABLE=1 export NCCL_IB_DISABLE=1
```

- --pretrained_model_name_or_path uses the SD‑Turbo base model.

- --pretrained_path is left empty because this is the first stage.

- After training, the best checkpoint (e.g., model_10001.pkl) will be saved under --output_dir/checkpoints/.


### Running Stage 2 – Discrete Text‑guided Training
Important: Before running this stage, ensure that Stage 1 has completed and the checkpoint path is correctly set.

```angular2html
CUDA_VISIBLE_DEVICES=5 python task_MySDUIE/train_SDUIE_Text.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="results/MySDUIE/train_SDUIE_Text" \
    --dataset_folder="results/MySDUIE/data/train_discrete" \
    --learning_rate 0.0001 \
    --resolution=512 \
    --train_batch_size=1 \
    --train_image_prep "resize_320" \
    --test_image_prep "resize_320" \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --eval_freq 1000 --num_training_epochs 1000 --num_samples_eval 200 \
    --pretrained_path "results/MySDUIE/train_SDUIE_Quant/checkpoints/model_from_stage1.pkl" \
    --stage S1 \
    --train_hazy_dir "train_A" --train_clear_dir "train_B"
```

- The key difference is --pretrained_path, which now points to the checkpoint from Stage 1.

- You may need to change the checkpoint filename if your training produced a different one.

### Notes on Arguments

- --dataset_folder, --train_hazy_dir, --train_clear_dir: adjust to your actual data paths.

- --output_dir: where logs and checkpoints will be saved.

- --eval_freq and --num_training_epochs control validation frequency and total epochs.

- --enable_xformers_memory_efficient_attention reduces VRAM usage.


## 🔬 Inference

### Stage 1: SDUIE-Quant Inference

After training SDUIE-Quant, you can perform inference on underwater images using **LoRA interpolation**, 
which enables continuous control over enhancement intensity by interpolating between the enhancement and hue-preservation LoRA weights.
The fusion factor `α ∈ [0, 1]` controls the enhancement level:

```angular2html
CUDA_VISIBLE_DEVICES=4 python task_MySDUIE/inference_paired_batch_with_LoRA_interpolation.py --pretrained_path "results/MySDUIE/train_SDUIE_Quant/checkpoints/model_10001.pkl" \
    --input_dir "results/MySDUIE/data/UCCS/" \
    --prompt "a clear image" \
    --output_dir "results/MySDUIE/train_SDUIE_Quant/UCCS_interpolation/"
```

### Stage 2: SDUIE-Text Inference

After training SDUIE-Text, you can perform semantic-level enhancement control using multi-prompt inference, 
which generates multiple enhanced outputs for each input image using a pre-defined set of prompts covering different enhancement levels (from `"one"` to `"ten"`). 
This is particularly useful for:


```angular2html
CUDA_VISIBLE_DEVICES=2 python task_MySDUIE/inference_with_LoRA_under_discrete_number_prompts.py --pretrained_path "results/MySDUIE/train_SDUIE_Text/checkpoints/model_5001.pkl" \
    --input_dir "results/MySDUIE/data/UCCS/" \
    --output_dir "results/MySDUIE/train_SDUIE_Text/UCCS_differ_prompts/"
```


**Supported Level Keywords**: `one`, `two`, `three`, `four`, `five`, `six`, `seven`, `eight`, `nine`, `ten`


### Other Option: Training and Test on UIEB 
We provide the evaluation code for the UIEB dataset. 
Notably, UIEB is not well-suited for the controlled experiments presented in this paper.

```angular2html
CUDA_VISIBLE_DEVICES=2 python task_MySDUIE/train_SDUIE_Quant.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="results/MySDUIE/train_SDUIE_Quant_UIEB" \
    --dataset_folder="results/MySDUIE/data/UIEB_for_SDUIE" \
    --learning_rate 0.0001 \
    --resolution=512 \
    --train_batch_size=1 \
    --train_image_prep "resize_320" \
    --test_image_prep "resize_320" \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --eval_freq 1000 --num_training_epochs 1000 --num_samples_eval 200 \
    --pretrained_path "" \
    --stage S1 \
    --train_hazy_dir "train_A" --train_clear_dir "train_B"
```
The inference command is as follows:
```angular2html
CUDA_VISIBLE_DEVICES=4 python task_MySDUIE/inference_paired_batch_with_LoRA_interpolation.py --pretrained_path "results/MySDUIE/train_SDUIE_Quant_UIEB/checkpoints/model_10001.pkl" \
    --input_dir "results/MySDUIE/data/UCCS/" \
    --prompt "a clear image" \
    --output_dir "results/MySDUIE/train_SDUIE_Quant_UIEB/UCCS_interpolation/"
```

After running the inference, select the images with the .p10 extension from the folder;
these represent the final level of enhancement results.




## Citation
If you find this work useful for your research, please cite our paper:


```bibtex
@InProceedings{Cong_2026_CVPR,
    author    = {Cong, Xiaofeng and Zhang, Yu-Xin and Shen, Hao and Jin, Yeying and Hou, Junming and Gui, Jie},
    title     = {SDUIE: Semi-Supervised Diffusion for Underwater Image Enhancement with Quant-Text Dual Control},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {37423-37433}
}
```



## 🙏 Acknowledgements

We gratefully acknowledge the contributors of the [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [🤗 Diffusers](https://github.com/huggingface/diffusers) libraries, which provided the foundational framework for our implementation.

Our work is also built upon the insights from [img2img-turbo](https://github.com/GaParmar/img2img-turbo), which pioneered one-step image-to-image translation with Stable Diffusion turbo and demonstrated the effectiveness of adversarial learning for efficient inference.