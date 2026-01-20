import os
import random

# ===== 主流程 =====
work_path = "./dataset/VQA/inter_merged"
result_txt_path = '/data1/cx/pred_VQA/result/random/area_result.txt'
results = ""

# 确保输出目录存在
os.makedirs(os.path.dirname(result_txt_path), exist_ok=True)

for scene_id in os.listdir(work_path):
    scene_path = os.path.join(work_path, scene_id)
    if not os.path.isdir(scene_path):
        continue

    for img_file in os.listdir(scene_path):
        if img_file.endswith('.jpg') and "-ls-" in img_file:
            try:
                img_name = img_file.replace(".jpg", "")
                
                print(f"Processing image {img_name} ...")
                
                # 生成0-1之间的随机数
                random_value = random.random()
                # >0.5输出1，否则输出0
                label = '1' if random_value > 0.5 else '0'
                
                line = f"{scene_id} {img_name} {label}\n"
                print(f"Result: {line.strip()} (random value: {random_value:.4f})")

                results += line

            except Exception as e:
                print(f"Error processing {scene_id}/{img_file}: {e}")

# 保存结果
with open(result_txt_path, "w") as f:
    f.write(results)

print("处理完成，结果写入:", result_txt_path)