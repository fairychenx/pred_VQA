import os
import argparse
import base64
import time
import requests
from collections import Counter

# ↓↓↓ SmolVLM 配置
BASE_URL = "http://localhost:8080/v1"
API_KEY = "dummy"  # 本地模型无所谓

system_intel = '''
In the provided bird's-eye view (BEV), the red lines in the photos are lane boundaries that are only for references. Color blocks highlighted are different segments of lanes. 
The colors of the blocks come from green and blue.
'''

# ==== TTS 配置 ====
use_tts = True        # True 启用多次推理 + 多数投票
num_samples = 3       # 每张图推理次数
sample_temperature = 0.7
per_call_delay = 0.2

# ==== Few-shot 示例 ====
few_shots = [
    {
        "path": "/data1/cx/pred_VQA/dataset/VQA/leftright/10008/315969525849927212-ls-3-4.png",
        "answer": "The green segment is on the left of the blue segment."
    },
    {
        "path": "/data1/cx/pred_VQA/dataset/VQA/leftright/10002/315971495049927216-ls-0-1.png",
        "answer": "The green segment is on the right of the blue segment."
    },
]

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_smolvlm(prompt: str, image_base64: str, fewshots=None, temperature: float = 0.5) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": system_intel}]

    # 添加 few-shot
    if fewshots:
        for fs in fewshots:
            fs_b64 = encode_image(fs["path"])
            messages.append({"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fs_b64}"}} 
            ]})
            messages.append({"role": "assistant", "content": fs["answer"]})

    # 添加目标问题
    messages.append({"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}} 
    ]})

    payload = {
        "model": "InternVL3-8B-Instruct",
        "messages": messages,
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


def ask_smolvlm_tta_vote(prompt: str, image_base64: str, fewshots=None,
                         num_samples: int = 5, temperature: float = 0.7):
    responses = []
    labels = []

    for _ in range(num_samples):
        res = ask_smolvlm(prompt, image_base64, fewshots=fewshots, temperature=temperature)
        responses.append(res)

        if "left" in res.lower() and "right" not in res.lower():
            labels.append("1")
        elif "right" in res.lower() and "left" not in res.lower():
            labels.append("2")
        else:
            labels.append("0")

        time.sleep(per_call_delay)

    # 多数投票
    counter = Counter(labels)
    final_label = counter.most_common(1)[0][0]

    return final_label, responses


# ---------------- 主逻辑 ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', help='text prompt path', default='/data1/cx/pred_VQA/lr_text_prompt.txt')
    parser.add_argument('--visual', help='visual prompt path', default='./dataset/VQA/leftright')
    parser.add_argument('--output', help='output result path', default='/data1/cx/pred_VQA/improved_result/testtime_fewshot/lr_result.txt')
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
            results += scene + '\n'
            continue
        for img in imgs:
            try:
                img_name = img.split('.png')[0]
                img_path = os.path.join(scene_path, img)
                base64_image = encode_image(img_path)

                if use_tts:
                    label, raw_responses = ask_smolvlm_tta_vote(
                        txt_prompt, base64_image, fewshots=few_shots,
                        num_samples=num_samples,
                        temperature=sample_temperature
                    )
                    if args.verbose:
                        print("Raw responses:", raw_responses)
                        print(f"投票结果标签: {label}")
                else:
                    result = ask_smolvlm(txt_prompt, base64_image, fewshots=few_shots)
                    if "left" in result.lower() and "right" not in result.lower():
                        label = "1"
                    elif "right" in result.lower() and "left" not in result.lower():
                        label = "2"
                    else:
                        label = "0"
                    if args.verbose:
                        print(f"模型响应: {result}")

                line = f"{scene} {img_name} {label}\n"
                if args.verbose:
                    print(line)
                results += line
            except Exception as e:
                print(f"Error processing scene {scene}: {e}")
                results += f"{scene} {img_name} 0\n"
                time.sleep(2)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(results)
    print(f"结果已保存到: {result_path}")
