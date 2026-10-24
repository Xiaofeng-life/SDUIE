import os
import shutil


def organize_images_by_suffix(input_dir, output_dir):
    """
    将输入目录下的图像，根据文件名末尾的 _pXX 后缀，
    分别复制到输出目录下的 11 个独立子文件夹中。

    :param input_dir: 存放原始所有图像的混合目录 (即你代码中的 args.output_dir)
    :param output_dir: 准备存放 11 个子文件夹的目标根目录
    """
    # 定义你之前的 11 个后缀名称
    factors_names = ["p00", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10"]

    # 确保目标总目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 在目标目录下提前创建好 11 个子文件夹
    for name in factors_names:
        os.makedirs(os.path.join(output_dir, name, "images"), exist_ok=True)

    # 遍历输入目录下的所有文件
    all_files = os.listdir(input_dir)
    copied_count = 0

    for file in all_files:
        # 确保处理的是文件，且是 jpg 图像
        if not os.path.isfile(os.path.join(input_dir, file)) or not file.lower().endswith('.jpg'):
            continue

        # 提取文件名（不含扩展名），例如 "image_p01"
        filename_without_ext = os.path.splitext(file)[0]

        # 根据你保存的代码：bname[:-4] + "_" + all_your_factors_names[idx] + ".jpg"
        # 文件的末尾 3 个字符刚好是 p00, p01 ... p10
        suffix = filename_without_ext[-3:]

        # 检查提取出的后缀是否在我们定义的 11 个分类中
        if suffix in factors_names:
            src_path = os.path.join(input_dir, file)
            dst_path = os.path.join(output_dir, suffix, "images",  file)

            # 执行复制操作（如果想变成移动，可以改为 shutil.move）
            shutil.copy(src_path, dst_path)
            copied_count += 1

    print(f"处理完成！共成功分类复制了 {copied_count} 张图像到 11 个子文件夹中。")

# ----------------------------------------
# 使用示例：
# ----------------------------------------
if __name__ == "__main__":
    # 你的生成图像混合在一起的目录
    generated_dir = "E:/CXF_PaperWriting/GeneralPaperWriting/01_PHD_papers/NeurIPS2025/CVPR2026/SDUIE/results/MySDUIE/train_SDUIE_Quant/UCCS_interpolation"
    # 你希望分类存放的新目录（也可以和 generated_dir 设为同一个，它会自动在里面建子文件夹）
    target_dir = "E:/CXF_PaperWriting/GeneralPaperWriting/01_PHD_papers/NeurIPS2025/CVPR2026/SDUIE/results/MySDUIE/train_SDUIE_Quant/UCCS_interpolation_splits"

    organize_images_by_suffix(generated_dir, target_dir)