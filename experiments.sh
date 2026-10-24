

# Inference
CUDA_VISIBLE_DEVICES=4 python task_MySDUIE/inference_with_LoRA_interpolation_Spherical.py --pretrained_path "results/MySDUIE/train_SDUIE_Quant/checkpoints/model_10001.pkl" \
    --input_dir "results/MySDUIE/data/UCCS/" \
    --prompt "a clear image" \
    --output_dir "results/MySDUIE/train_SDUIE_Quant/UCCS_interpolation/"
