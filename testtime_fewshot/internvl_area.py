# internvl_area_fewshot_v2.py
from openai import OpenAI
import os, time, base64, json, re, copy

client = OpenAI(api_key="sk-no-key", base_url="http://localhost:8080/v1")

system_intel = '''
The provided photo is mosaiced with two images, with the bird's-eye view (BEV) on the left and the front perspective view (PV) on the right. In BEV, the red lines represent lane boundaries. 
Normally, the lane segment is considered as not in the intersection area when it is in front of the area or at rear of the area.
'''

work_path = "/data1/cx/pred_VQA/dataset/VQA/inter_merged"
result_txt_path = '/data1/cx/pred_VQA/improved_result/testtime_fewshot/area_result.txt'
wait_between_images = 1.0

# few-shot 示例
few_shots = [
    {
        "path": "/data1/cx/pred_VQA/dataset/VQA/inter_merged/10002/315971488049927216-ls-5.jpg",
        "answer": "No, the green segment patch not in the intersection area."
    },
    {
        "path": "/data1/cx/pred_VQA/dataset/VQA/inter_merged/10000/315969916349927220-ls-0.jpg",
        "answer": "Yes, the green segment patch is in the intersection area."
    }
]

# 参数
TRIES = 3               # 修正时的独立调用次数（多重 -> 3）
TEMPERATURE = 0.0       # 改为 0 保持稳定


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_yes(text):
    """
    从输出中提取 yes/no
    """
    if not text:
        return None
    if re.search(r'\byes\b', text, re.I):
        return "yes"
    if re.search(r'\bno\b', text, re.I):
        return "no"
    return None


def call_model_once(messages, max_tokens=200, temperature=0.0):
    resp = client.chat.completions.create(
        model="InternVL3-8B-Instruct",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()


def process_one_image(img_path, fewshots):
    # 初始会话
    messages = [{"role":"system", "content": system_intel}]

    # 添加 few-shot 样例 (user + assistant)
    for idx, fs in enumerate(fewshots, start=1):
        fs_b64 = encode_image(fs["path"])
        messages.append({"role":"user", "content":[
            {"type":"text", "text": "You are an expert in analyzing lane structure in the image. Let's determine if the green segment patch is in the intersection area. Please reply in a brief sentence starting with 'Yes' or 'No'."},
            {"type":"image_url", "image_url":{"url": f"data:image/jpeg;base64,{fs_b64}"}}
        ]})
        messages.append({"role":"assistant", "content": fs["answer"]})

    # 目标问题
    target_b64 = encode_image(img_path)
    messages.append({"role":"user", "content":[
        {"type":"text", "text": "You are an expert in analyzing lane structure in the image. Let's determine if the green segment patch is in the intersection area. Please reply in a brief sentence starting with 'Yes' or 'No'."},
        {"type":"image_url", "image_url":{"url": f"data:image/jpeg;base64,{target_b64}"}}
    ]})

    # 多重采样 (3 次)
    candidates, votes = [], {"yes":0, "no":0}
    for i in range(TRIES):
        out = call_model_once(messages, max_tokens=100, temperature=TEMPERATURE)
        ans = extract_yes(out)
        candidates.append({"raw": out, "ans": ans})
        if ans in votes:
            votes[ans] += 1

    # 投票结果
    if votes["yes"] > votes["no"]:
        final = "yes"
    elif votes["no"] > votes["yes"]:
        final = "no"
    else:
        final = candidates[0]["ans"]  # 平局时取第一个结果

    return {
        "candidates": candidates,
        "votes": votes,
        "final": final
    }


def main():
    results = ""
    os.makedirs(os.path.dirname(result_txt_path), exist_ok=True)

    for scene_id in sorted(os.listdir(work_path)):
        scene_path = os.path.join(work_path, scene_id)
        if not os.path.isdir(scene_path):
            continue
        for img_file in sorted(os.listdir(scene_path)):
            if not (img_file.endswith(".jpg") and "-ls-" in img_file):
                continue
            img_name = img_file.replace(".jpg", "")
            img_path = os.path.join(scene_path, img_file)
            try:
                print(f"\n处理图片 {scene_id}/{img_name} ...")
                res = process_one_image(img_path, few_shots)

                for i,c in enumerate(res["candidates"], start=1):
                    print(f"  cand[{i}]: ans={c['ans']}, raw={c['raw'][:120]}")

                print("投票结果:", res["votes"], "=> 最终:", res["final"])

                label = '1' if res["final"] == 'yes' else '0'
                line = f"{scene_id} {img_name} {label}\n"
                results += line

                time.sleep(wait_between_images)
            except Exception as e:
                print("处理出错:", e)
                time.sleep(2)

    with open(result_txt_path, "w") as f:
        f.write(results)
    print("全部处理完，结果写入:", result_txt_path)


if __name__ == "__main__":
    main()
