import os
import argparse
import base64
import time
import requests

# ↓↓↓ 1. 换成 SmolVLM 的地址
BASE_URL = "http://localhost:8080/v1"   # 注意 /v1 后缀
API_KEY    = "dummy"                   # 本地模型无所谓，占位即可

system_intel = '''
In the provided bird's-eye view (BEV), the green and blue lane segments are highlighted. The white arrow represents the driving direction of the green segment, and the magenta arrow points from the mid point of the green segment to that of the blue segment. At the top right corner of the image, two arrows with the same directions are drawn in white and magenta, which can be referred to.
The red lines in the photos are lane boundaries, and the part of black is gap segments or the context.
Normally or when the case is confusing, two directions with deviation of less than 45 degrees are considered as compatible. 
'''

# ==== TTS 配置 ====
use_tts = True       # True 启用多次推理+投票
num_samples = 3      # 每张图推理次数
sample_temperature = 0.7
per_call_delay = 0.2


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def ask_smolvlm(prompt: str, image_base64: str, temperature: float = 0.5) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "InternVL3-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_intel},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }}
            ]}
        ],
        "temperature": temperature,
        "max_tokens": 512
    }
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions",
                             headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print("[SmolVLM] 调用失败:", e)
        return "__ERROR__"


def ask_smolvlm_tta_vote(prompt: str, image_base64: str,
                         num_samples: int = 5, temperature: float = 0.7):
    responses = []
    for i in range(num_samples):
        res = ask_smolvlm(prompt, image_base64, temperature=temperature)
        responses.append(res)
        time.sleep(per_call_delay)

    yes_count = sum(1 for r in responses if isinstance(r, str) and ("yes" in r.lower() or "是" in r.lower()))
    no_count = len(responses) - yes_count

    if yes_count > no_count:
        final_label = "1"
        final = "Yes"
    else:
        final_label = "0"
        final = "No"

    return final, final_label, responses


# ---------------- 主逻辑 ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', help='text prompt path', default='/data1/cx/pred_VQA/vec_text_prompt.txt')
    parser.add_argument('--visual', help='visual prompt path', default='/data1/cx/pred_VQA/dataset/VQA/vector')
    parser.add_argument('--output', help='output result path', default='/data1/cx/pred_VQA/improved_result/test_internvl_3times/vec_result1.txt')
    parser.add_argument('--key', help='openai api key')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    with open(args.txt, 'r') as f:
        txt_prompt = f.read()
    exec(txt_prompt)

    work_path = args.visual
    result_path = args.output
    results = ""
    for scene in range(10001, 10150):
        scene = str(scene)
        scene_path = os.path.join(work_path, scene)
        if not os.path.exists(scene_path):
            continue
        imgs = os.listdir(scene_path)
        if len(imgs) == 0:
            line = scene + '\n'
            results += line
            continue
        for img in imgs:
            try:
                img_name = img.split('.png')[0]
                img_path = os.path.join(scene_path, img)
                base64_image = encode_image(img_path)

                if use_tts:
                    final, label, raw_responses = ask_smolvlm_tta_vote(
                        txt_prompt, base64_image,
                        num_samples=num_samples,
                        temperature=sample_temperature
                    )
                    if args.verbose:
                        print("Raw responses:", raw_responses)
                        print(f"最终投票结果: {final}")
                else:
                    result = ask_smolvlm(txt_prompt, base64_image)
                    label = '1' if ('yes' in result.lower() or '是' in result.lower()) else '0'
                    if args.verbose:
                        print(f"模型响应: {result}")

                line = f"{scene} {img_name} {label}\n"
                if args.verbose:
                    print(line)
                results += line
            except Exception as e:
                print(f'Error processing scene {scene}: {e}')
                results += f"{scene} {img_name} 0\n"
                time.sleep(2)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding='utf-8') as f:
        f.write(results)
    print(f"结果已保存到: {result_path}")
