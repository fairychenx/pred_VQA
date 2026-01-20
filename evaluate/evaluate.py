import os
import argparse

# ==================== 配置区域：所有任务的结果路径和标注路径 ====================
# 方便统一修改所有路径
RESULT_PATHS = {
    'area_result': '/data1/cx/pred_VQA/result/drivemm/area_result.txt',
    'conn_result': '/data1/cx/pred_VQA/result/drivemm/conn_result.txt',
    'lr_result': '/data1/cx/pred_VQA/result/drivemm/lr_result.txt',
    'vec_result': '/data1/cx/pred_VQA/result/drivemm/vec_result.txt',
    
    'area_annotation': '/data1/cx/pred_VQA/dataset/area_annotation.txt',   
    'conn_annotation': '/data1/cx/pred_VQA/dataset/conn_annotation.txt',
    'lr_annotation': '/data1/cx/pred_VQA/dataset/lr_annotation.txt',
    'vec_annotation': '/data1/cx/pred_VQA/dataset/vec_annotation.txt',
}
# ============================================================================


def load_dict(content, col=None):
    """
    加载字典数据
    Args:
        content: 文件内容列表
        col: 列索引，如果为None则默认使用第2列（索引2）
    """
    scenes = set()
    value = {}
    for line in content:
        line = line.split(' ')
        scene = line[0]
        if scene not in scenes:
            value[scene] = {}
        scenes.add(scene)
        if len(line) > 1:
            # 如果col为None，使用默认值2（用于lr任务）
            col_idx = col if col is not None else 2
            value[scene][line[1]] = line[col_idx]
    return value


def evaluate_area(result_path=None, annotation_path=None):
    """评估area任务"""
    if result_path is None:
        result_path = RESULT_PATHS['area_result']
    if annotation_path is None:
        annotation_path = RESULT_PATHS['area_annotation']
    
    print("=" * 50)
    print("Evaluating Area Task")
    print("=" * 50)
    
    with open(result_path, "r") as f:
        results = f.read()
        results = results.split('\n')

    with open(annotation_path, "r") as f:
        annotation = f.read()
        annotation = annotation.split('\n')

    num_img = 0
    label = load_dict(annotation, 2)
    prediction = load_dict(results, 2)

    num_0 = 0
    num_1 = 0
    tp = 0
    for scene in prediction:
        if len(prediction[scene]) != 0:
            for img in prediction[scene]:
                if label[scene].get(img) == '0':
                    num_0 += 1
                if label[scene].get(img) == '1':
                    num_1 += 1
                if prediction[scene][img] == label[scene].get(img):
                    tp += 1
                num_img += 1

    print(f"number of img: {num_img}")
    print(f"number of 0s: {num_0}")
    print(f"number of 1s: {num_1}")
    print(f"number of tps: {tp}")
    print(f"Accuracy: {tp / num_img}")
    print()


def evaluate_conn(result_path=None, annotation_path=None):
    """评估conn任务"""
    if result_path is None:
        result_path = RESULT_PATHS['conn_result']
    if annotation_path is None:
        annotation_path = RESULT_PATHS['conn_annotation']
    
    print("=" * 50)
    print("Evaluating Conn Task")
    print("=" * 50)
    
    with open(result_path, "r") as f:
        results = f.read()
        results = results.split('\n')

    with open(annotation_path, "r") as f:
        annotation = f.read()
        annotation = annotation.split('\n')

    num_img = 0
    label = load_dict(annotation, 2)
    prediction = load_dict(results, 2)

    num_0 = 0
    num_1 = 0
    tp = 0
    for scene in prediction:
        if len(prediction[scene]) != 0:
            for img in prediction[scene]:
                if label.get(scene) != None:
                    if label[scene].get(img) == '0':
                        num_0 += 1
                        num_img += 1
                    if label[scene].get(img) == '1':
                        num_1 += 1
                        num_img += 1
                    if prediction[scene][img] == label[scene].get(img):
                        tp += 1

    print(f"number of img: {num_img}")
    print(f"number of 0s: {num_0}")
    print(f"number of 1s: {num_1}")
    print(f"number of tps: {tp}")
    print(f"Accuracy: {tp / num_img}")
    print()


def evaluate_lr(result_path=None, annotation_path=None):
    """评估lr任务"""
    if result_path is None:
        result_path = RESULT_PATHS['lr_result']
    if annotation_path is None:
        annotation_path = RESULT_PATHS['lr_annotation']
    
    print("=" * 50)
    print("Evaluating LR Task")
    print("=" * 50)
    
    with open(result_path, "r") as f:
        results = f.read()
        results = results.split('\n')

    with open(annotation_path, "r") as f:
        annotation = f.read()
        annotation = annotation.split('\n')

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
                    num_img += 1
                if label[scene].get(img) == '1':
                    num_1 += 1
                    num_img += 1
                if label[scene].get(img) == '2':
                    num_2 += 1
                    num_img += 1
                if prediction[scene][img] == label[scene].get(img):
                    tp += 1

    print(f"number of img: {num_img}")
    print(f"number of 0s: {num_0}")
    print(f"number of 1s: {num_1}")
    print(f"number of 2s: {num_2}")
    print(f"number of tps: {tp}")
    print(f"Accuracy: {tp / num_img}")
    print()


def evaluate_vec(result_path=None, annotation_path=None):
    """评估vec任务"""
    if result_path is None:
        result_path = RESULT_PATHS['vec_result']
    if annotation_path is None:
        annotation_path = RESULT_PATHS['vec_annotation']
    
    print("=" * 50)
    print("Evaluating Vec Task")
    print("=" * 50)
    
    with open(result_path, "r") as f:
        results = f.read()
        results = results.split('\n')

    with open(annotation_path, "r") as f:
        annotation = f.read()
        annotation = annotation.split('\n')

    num_img = 0
    label = load_dict(annotation, 2)
    prediction = load_dict(results, 2)

    num_p = 0
    check = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for scene in prediction:
        if len(prediction[scene]) != 0:
            for img in prediction[scene]:
                if label.get(scene) != None:
                    if label[scene].get(img) == "1":
                        num_p += 1
                    if prediction[scene][img] == "1" and label[scene].get(img) == "1":
                        tp += 1
                        num_img += 1
                    if prediction[scene][img] == "0" and label[scene].get(img) == "0":
                        tn += 1
                        num_img += 1
                    if prediction[scene][img] == "1" and label[scene].get(img) == "0":
                        fp += 1
                        num_img += 1
                    if prediction[scene][img] == "0" and label[scene].get(img) == "1":
                        fn += 1
                        num_img += 1
                    if prediction[scene][img] == label[scene].get(img):
                        check += 1

    print(f"number of img: {num_img}")
    print(f"number of checks: {check}")
    print(f"number of ps: {num_p}")
    print(f"number of tps: {tp}")
    print(f"number of tns: {tn}")
    print(f"number of fps: {fp}")
    print(f"number of fns: {fn}")
    print(f"Accuracy: {check / num_img}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate all tasks: area, conn, lr, vec')
    parser.add_argument('--area_result', help='area result path', 
                        default=RESULT_PATHS['area_result'])
    parser.add_argument('--area_annotation', help='area annotation path',
                        default=RESULT_PATHS['area_annotation'])
    parser.add_argument('--conn_result', help='conn result path',
                        default=RESULT_PATHS['conn_result'])
    parser.add_argument('--conn_annotation', help='conn annotation path',
                        default=RESULT_PATHS['conn_annotation'])
    parser.add_argument('--lr_result', help='lr result path',
                        default=RESULT_PATHS['lr_result'])
    parser.add_argument('--lr_annotation', help='lr annotation path',
                        default=RESULT_PATHS['lr_annotation'])
    parser.add_argument('--vec_result', help='vec result path',
                        default=RESULT_PATHS['vec_result'])
    parser.add_argument('--vec_annotation', help='vec annotation path',
                        default=RESULT_PATHS['vec_annotation'])
    args = parser.parse_args()

    # 依次评估四个任务
    evaluate_area(args.area_result, args.area_annotation)
    evaluate_conn(args.conn_result, args.conn_annotation)
    evaluate_lr(args.lr_result, args.lr_annotation)
    evaluate_vec(args.vec_result, args.vec_annotation)
    
    print("=" * 50)
    print("All evaluations completed!")
    print("=" * 50)

