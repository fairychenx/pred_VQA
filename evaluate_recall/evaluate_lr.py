import os

result_path = '/data1/cx/pred_VQA/result/qwen2/lr_result.txt'
annotation_path = '/data1/cx/pred_VQA/dataset/lr_annotation.txt'

with open(result_path, "r") as f:
    results = f.read()
    results = results.split('\n')

with open(annotation_path, "r") as f:
    annotation = f.read()
    annotation = annotation.split('\n')


def load_dict(content):

    scenes = set()
    value = {}
    for line in content:
        line = line.split(' ')
        scene = line[0]
        if scene not in scenes:
            value[scene] = {}
        scenes.add(scene)
        if len(line) > 1:
            value[scene][line[1]] = line[2]

    return value

num_img = 0
label = load_dict(annotation)
prediction = load_dict(results)


tp = 0
num_0 = 0
num_1 = 0
num_2 = 0
# 为每个类别计算TP和FN
tp_class = {'0': 0, '1': 0, '2': 0}
fn_class = {'0': 0, '1': 0, '2': 0}
for scene in prediction:
    if len(prediction[scene]) != 0:
        for img in prediction[scene]:
            pred = prediction[scene][img]
            true_label = label[scene].get(img)
            if true_label == '0':
                num_0 += 1
                num_img +=1
            if true_label == '1':
                num_1 += 1
                num_img +=1
            if true_label == '2':
                num_2 += 1
                num_img +=1
            if pred == true_label:
                tp += 1
                tp_class[pred] += 1
            else:
                # 预测错误：真实类别的FN+1
                if true_label in fn_class:
                    fn_class[true_label] += 1

# 计算每个类别的召回率
recall_class = {}
for cls in ['0', '1', '2']:
    tp_c = tp_class[cls]
    fn_c = fn_class[cls]
    recall_class[cls] = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0

# 计算宏平均召回率
macro_recall = sum(recall_class.values()) / len(recall_class)

print(f"number of img: {num_img}")
print(f"number of 0s: {num_0}")
print(f"number of 1s: {num_1}")
print(f"number of 2s: {num_2}")
print(f"number of tps: {tp}")
print(f"Accuracy: {tp / num_img}")
print(f"Recall (class 0): {recall_class['0']}")
print(f"Recall (class 1): {recall_class['1']}")
print(f"Recall (class 2): {recall_class['2']}")
print(f"Macro Recall: {macro_recall}")