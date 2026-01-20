import os
import argparse
import base64
import time
import requests

# ================== 配置 ==================
BASE_URL = "http://localhost:8080/v1"
API_KEY = "dummy"  # 本地模型无所谓，占位即可

# 系统信息
system_intel = '''
The red lines in the photos are lane boundaries. Two segments in different lanes don't have any connection relationship. Only two segments in the same lane end to end adjacent are considered as directly connected.
'''

# few-shot 示例
few_shots = [
    {
        "path": "/data1/cx/pred_VQA/dataset/VQA/connection/10001/315966078549927213-ls-0-1.png",
        "answer": "No,the green patch is not connected with the blue patch."
    },
    {
        "path": "/data1/cx/pred_VQA/dataset/VQA/connection/10004/315970822649927215-ls-0-3.png",
        "answer": "Yes,the green patch is directly connected with the blue patch."
    },
]

# ==== TTS 配置 ====
use_tts = True       # True 启用多次推理+投票，False 单次
num_samples = 3      # 每张图推理次数
sample_temperature = 0.7
per_call_delay = 0.2
# ==========================================

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_smolvlm(prompt: str, image_base64: str, temperature: float = 0.5) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "InternVL3-8B-Instruct",
        "messages": [
            {"role": "system", "content": system_intel},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print("[SmolVLM] 调用失败:", e)
        return "__ERROR__"


def ask_smolvlm_fewshot(prompt: str, target_img_path: str, fewshots: list,
                         num_samples: int = 3, temperature: float = 0.7):
    """
    few-shot + TTA 投票
    """
    # 构建消息列表
    messages = [{"role": "system", "content": system_intel}]

    # 添加 few-shot 样例
    for fs in fewshots:
        fs_b64 = encode_image(fs["path"])
        messages.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fs_b64}"}}
        ]})
        messages.append({"role": "assistant", "content": fs["answer"]})

    # 添加目标图片
    target_b64 = encode_image(target_img_path)
    messages.append({"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}"}}
    ]})

    # 多重采样投票
    candidates = []
    votes = {"yes": 0, "no": 0}
    for _ in range(num_samples):
        out = ask_smolvlm(prompt="You are an expert in determining adjacent lane segments in the image. Let's determine if the the green segment is directly connected with the blue segmemt. Please reply in a brief sentence starting with 'Yes' or 'No'.", image_base64=target_b64, temperature=temperature)
        ans = "yes" if (out.lower().lstrip().startswith("yes") or "是" in out) else "no"
        candidates.append({"raw": out, "ans": ans})
        votes[ans] += 1
        time.sleep(per_call_delay)

    # 最终投票结果
    final = "yes" if votes["yes"] > votes["no"] else "no"
    return final, candidates, votes


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", help="text prompt path", default="/data1/cx/pred_VQA/conn_text_prompt.txt")
    parser.add_argument("--visual", help="visual prompt path", default="/data1/cx/pred_VQA/dataset/VQA/connection")
    parser.add_argument("--output", help="output result path", default="/data1/cx/pred_VQA/improved_result/testtime_fewshot/conn_result.txt")
    parser.add_argument("--key", help="openai api key")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # 读取文本 prompt
    with open(args.txt, "r") as f:
        txt_prompt = f.read()
    exec(txt_prompt)
    txt_prompt="You are an expert in determining adjacent lane segments in the image. Let's determine if the the green segment is directly connected with the blue segmemt. Please reply in a brief sentence starting with 'Yes' or 'No'."
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
            results += scene + "\n"
            continue

        for img in imgs:
            try:
                img_name = img.split(".png")[0]
                img_path = os.path.join(scene_path, img)

                if use_tts:
                    final, candidates, votes = ask_smolvlm_fewshot(
                        txt_prompt, img_path, few_shots,
                        num_samples=num_samples, temperature=sample_temperature
                    )
                    if args.verbose:
                        for i, c in enumerate(candidates, 1):
                            print(f"  cand[{i}]: ans={c['ans']}, raw={c['raw'][:120]}")
                        print("投票结果:", votes, "=> 最终:", final)
                else:
                    final, candidates, votes = ask_smolvlm_fewshot(txt_prompt, img_path, few_shots, num_samples=1, temperature=0.0)

                label = "1" if final == "yes" else "0"
                line = f"{scene} {img_name} {label}\n"
                if args.verbose:
                    print(f"模型响应: {final}")
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
