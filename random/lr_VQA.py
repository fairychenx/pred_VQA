import os
import random



def main():
    
    work_path = "./dataset/VQA/leftright"
    result_path = "/data1/cx/pred_VQA/result/random/lr_result.txt"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
    results = ""

    for scene in range(10000, 10150):
        print('------------------------')
        print(f'Processing scene {scene}...')
        print('------------------------')
        scene = str(scene)
        scene_path = os.path.join(work_path, scene)
        if not os.path.exists(scene_path):
          continue
        imgs = os.listdir(scene_path)
        if len(imgs) == 0:
            line = scene + '\n'
            print(line)
            results += line
            continue
        for img in imgs:
            try:
                img_name = img.split('.png')[0]
                print(f'Processing image {img_name}...')
                
                # 生成0-1之间的随机数，用于决定输出0、1或2
                random_value = random.random()
                # 将[0,1)分成三等分：[0,0.33)->0, [0.33,0.67)->1, [0.67,1)->2
                if random_value < 0.33:
                    label = '0'
                elif random_value < 0.67:
                    label = '1'
                else:
                    label = '2'
                
                line = scene + " " + img_name + " " + label + "\n"
                print(f"Result: {line.strip()} (random value: {random_value:.4f})")
                results += line
            except Exception as e:
                print(f'Error processing scene {scene}: {e}')

    with open(result_path, "w") as f:
        f.write(results)
    
    print(f"处理完成，结果写入: {result_path}")
    return


if __name__ == '__main__':
    main()
