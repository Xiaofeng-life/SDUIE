import os
import shutil

# ========= 这里修改成你dp00~dp10所在的根目录 =========
root_dir = r"E:\CXF_PaperWriting\GeneralPaperWriting\01_PHD_papers\NeurIPS2025\CVPR2026\SDUIE\results\MySDUIE\train_SDUIE_Text\UCCS"
# 目标文件夹，放在dp同级，名字叫all_images
target_dir = os.path.join(root_dir)
os.makedirs(target_dir, exist_ok=True)

# 支持的图片后缀
img_suffix = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# 遍历dp00,dp01...dp10
for folder_name in os.listdir(root_dir):
    dp_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(dp_path):
        continue
    # 拼接 images子文件夹路径
    images_src = os.path.join(dp_path, "images")
    if not os.path.exists(images_src):
        print(f"警告：{images_src} 不存在，跳过")
        continue

    # 遍历images里所有图片
    for img_name in os.listdir(images_src):
        if img_name.lower().endswith(img_suffix):
            src_file = os.path.join(images_src, img_name)
            dst_file = os.path.join(target_dir, img_name)

            # 重名处理：加上dp前缀防止覆盖
            if os.path.exists(dst_file):
                dst_file = os.path.join(target_dir, f"{folder_name}_{img_name}")

            shutil.copy2(src_file, dst_file)
            print(f"复制: {src_file} -> {dst_file}")

print("===== 全部复制完成 =====")
