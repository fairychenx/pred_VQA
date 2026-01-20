from openai import OpenAI
import os
import time
import base64

# ===== 配置 =====
client = OpenAI(
    api_key="sk-no-key",  # 本地部署不需要真实 key
    base_url="http://localhost:8080/v1"  # llama-server 地址
)

system_intel = '''
The provided photo is mosaiced with two images, with the bird's-eye view (BEV) on the left and the front perspective view (PV) on the right. In BEV, the red lines represent lane boundaries. 
Normally, the lane segment is considered as not in the intersection area when it is in front of the area or at rear of the area.
'''

work_path = "./dataset/VQA/inter_merged"
result_txt_path = '/data1/cx/pred_VQA/internvl_area_3times.txt'

# TTS（多次采样）开关与参数
use_tts = True        # True = 启用多次采样+投票；False = 单次推理
num_samples = 3       # 每张图的调用次数（采样次数）
sample_temperature = 0.7  # 采样时的 temperature（提高随机性以得到多样回答）
per_call_delay = 0.2  # 每次调用间隔（秒），避免短时间请求过快


# ===== 工具函数 =====
def encode_image(image_path):
    """将图片转成 base64（字符串）"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_local_model_once(prompt, image_b64, temperature=0.0):
    """一次模型调用，返回模型文本回答（strip后）"""
    prompt_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
    ]

    response = client.chat.completions.create(
        model="InternVL3-8B-Instruct",
        messages=[
            {"role": "system", "content": system_intel},
            {"role": "user", "content": prompt_content}
        ],
        max_tokens=100,
        temperature=temperature
    )

    return response.choices[0].message.content.strip()


def ask_local_model_tta_vote(prompt, image_path, num_samples=3, temperature=0.7):
    """
    在同一张图上多次调用模型（不改变图），收集回答并多数投票。
    返回 (final_label_str, raw_responses_list)
    final_label_str 为 "Yes" 或 "No"
    """
    image_b64 = encode_image(image_path)
    raw_responses = []

    for i in range(num_samples):
        try:
            res = ask_local_model_once(prompt, image_b64, temperature=temperature)
            raw_responses.append(res)
        except Exception as e:
            # 出错时记录一个特殊字符串，继续后续采样
            raw_responses.append(f"__ERROR__:{e}")
        time.sleep(per_call_delay)

    # 统计以 "Yes" 开头的回答（不区分大小写）
    yes_count = sum(1 for r in raw_responses if isinstance(r, str) and r.lower().lstrip().startswith("yes"))
    no_count = sum(1 for r in raw_responses if isinstance(r, str) and r.lower().lstrip().startswith("no"))

    # 若有至少一半以上是 Yes 则最终判 Yes；否则判 No。
    # 注意：如果多数无法决出（比如都是错误或既无 Yes 也无 No），默认判 No。
    if yes_count > no_count:
        final = "Yes"
    else:
        final = "No"

    return final, raw_responses


# ===== 主流程 =====
def main():
    results = ""

    for scene_id in os.listdir(work_path):
        scene_path = os.path.join(work_path, scene_id)
        if not os.path.isdir(scene_path):
            continue

        for img_file in os.listdir(scene_path):
            if img_file.endswith('.jpg') and "-ls-" in img_file:
                img_name = img_file.replace(".jpg", "")
                img_path = os.path.join(scene_path, img_file)

                try:
                    print(f"处理图片 {scene_id}/{img_name} ...")

                    prompt = "You are an expert in analyzing lane structure in the image. Let's determine if the green segment patch is in the intersection area. Please reply in a brief sentence starting with 'Yes' or 'No'."

                    if use_tts:
                        final_result, raw_responses = ask_local_model_tta_vote(
                            prompt, img_path, num_samples=num_samples, temperature=sample_temperature
                        )
                        print("Raw responses:", raw_responses)
                        result = final_result
                    else:
                        # 单次调用（temperature 设为 0 保持确定性）
                        b64 = encode_image(img_path)
                        result = ask_local_model_once(prompt, b64, temperature=0.0)

                    print("API final response:", result)

                    # 生成标签：以 'Yes' 开头视为 1，否则为 0
                    label = '1' if isinstance(result, str) and result.lower().lstrip().startswith("yes") else '0'
                    line = f"{scene_id} {img_name} {label}\n"
                    print("Result:", line.strip())

                    results += line

                except Exception as e:
                    print(f"处理 {scene_id}/{img_file} 出错: {e}")
                    time.sleep(2)  # 避免请求过快

    # 保存结果
    os.makedirs(os.path.dirname(result_txt_path), exist_ok=True)
    with open(result_txt_path, "w") as f:
        f.write(results)

    print("处理完成，结果写入:", result_txt_path)


if __name__ == "__main__":
    main()
