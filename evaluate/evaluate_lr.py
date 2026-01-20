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
for scene in prediction:
    if len(prediction[scene]) != 0:
        for img in prediction[scene]:
            if label[scene].get(img) == '0':
                num_0 += 1
                num_img +=1
            if label[scene].get(img) == '1':
                num_1 += 1
                num_img +=1
            if label[scene].get(img) == '2':
                num_2 += 1
                num_img +=1
            if prediction[scene][img] == label[scene].get(img):
                tp += 1

print(f"number of img: {num_img}")
print(f"number of 0s: {num_0}")
print(f"number of 1s: {num_1}")
print(f"number of 2s: {num_2}")
print(f"number of tps: {tp}")
print(f"Accuracy: {tp / num_img}")