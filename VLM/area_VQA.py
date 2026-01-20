from openai import OpenAI
import os
import base64
import json
from IPython.display import Markdown, display
import time
import sys
sys.path.append("/home/ps/cx/neural-symbolic/API/VQA")
from find_parallel_lanes import search_para_segment
from find_egolane_id import ploygon_filter_ids



api_key="sk-7zZMomkPoQGrnxe1Xqar5O17AfaqShEMALVqucYCIuzr6dr5"
client = OpenAI(
    api_key=api_key,
    base_url="https://aizex.top/v1",
    )


system_intel = '''
The provided photo is mosaiced with two images, with the bird's-eye view (BEV) on the left and the front perspective view (PV) on the right. In BEV, the red lines represent lane boundaries. 
Normally, the lane segment is considered as not in the intersection area when it is in front of the area or at rear of the area.
'''

# ===== 工具函数 =====
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ask_GPT4(prompt, image_base64):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": [
                {"type": "text", "text": system_intel},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                 }
            ]}
        ],
        max_tokens=100,
        temperature=0
    )
    return response.choices[0].message.content

# ===== 主流程 =====
work_path = "./dataset/VQA/inter_merged"
result_txt_path = './result/gpt-4o/area_result.txt'
results = ""

for scene_id in os.listdir(work_path):
    scene_path = os.path.join(work_path, scene_id)
    if not os.path.isdir(scene_path):
        continue

    for img_file in os.listdir(scene_path):
        if img_file.endswith('.jpg') and "-ls-" in img_file:
            try:
                img_name = img_file.replace(".jpg", "")
                img_path = os.path.join(scene_path, img_file)

                print(f"Encoding image {img_name} ...")
                base64_image = encode_image(img_path)

                prompt = """
You are an expert in analyzing lane structure in the image.
Let's determine if the green segment patch is in the intersection area.
Please reply in a brief sentence starting with "Yes" or "No".
                """

                print("Calling deepseekv3 API...")
                result = ask_GPT4(prompt, base64_image)
                print("API response:", result)

                label = '1' if "Yes" in result else '0'
                line = f"{scene_id} {img_name} {label}\n"
                print("Result:", line.strip())

                results += line

            except Exception as e:
                print(f"Error processing {scene_id}/{img_file}: {e}")
                time.sleep(2)  # 避免 API 速率限制

# 保存结果
with open(result_txt_path, "w") as f:
    f.write(results)

print("处理完成，结果写入:", result_txt_path)


# if __name__ == '__main__':
#     try:
#         print('Encoding image...')
#         base64_image = encode_image("/DATA_EDS2/wanggl/datasets/Mosaic_BEV_PV/10000/315969916349927220-ls-0.jpg")
#         print('Image encoded.')
# #         prompt = f'''
# # In the provided photograph, is the green patch vertically directly connected with the blue patch ignoring the proper gap?
# #         '''
#       #  Please reply starting with "Yes" or "No"
#         prompt = f'''
# You are an expert in analyzing lane structure in the image. Let's determine if the the green segment patch is in the intersection area. Please reply in a brief sentence starting with "Yes" or "No".
#         '''
#         # Call GPT-4 API
#         print('Calling GPT-4 API...')
#         result = ask_GPT4(prompt, base64_image)
#         print('API call successful.')
#         print(result)
#         # if "left" in result and "right" not in result:
#         #     print(1)
#         # elif "right" in result and "left" not in result:
#         #     print(2)
#         # else:
#         #     print(0)
#     except Exception as e:
#         # print(f'Error processing scene {scene}: {e}')
#         time.sleep(2)  # Optionally add a delay to avoid rate limiting



# Process each sample
# for idx, sample in enumerate(samples):
#     try:
#         print('------------------------')
#         print(f'Processing sample {idx}...')
#         print('------------------------')

#         # image_path = sample['image_name']
#         # lane_coordinates = sample['lane_coordinates']
#         # traffic_coordinates = sample['traffic_element_coordinates']
#         # attribute = attribute_mapping[sample['attribute']]  # Map attribute number to description

#         # Generate prompt
#         prompt = f'''
# Based on the provided image where the traffic element is highlighted with a green anchor box at coordinates {traffic_coordinates} with attribute '{attribute}' , and the lane line is marked with a green line flanked by two red lines at coordinates {lane_coordinates}, please analyze whether there is a topological relationship between the designated traffic element and the lane line. Use the established rules to guide your analysis and express your reasoning in DSL format.

# '''

#         # Encode the image
#         print('Encoding image...')
#         newimgpath = '/DATA_EDS2/wanggl/datasets/Opentest_719test2' + '/' + image_path
#         base64_image = encode_image(newimgpath)
#         print('Image encoded.')

#         # Call GPT-4 API
#         print('Calling GPT-4 API...')
#         result = ask_GPT4(system_intel, prompt, base64_image)
#         print('API call successful.')

#         # Save the result with the image name as key
#         image_name = os.path.basename(image_path)
#         results[image_name] = result

#     except Exception as e:
#         print(f'Error processing sample {idx}: {e}')
#         failed_requests.append((idx, str(e)))
#         print('Moving to next sample...')
#         time.sleep(2)  # Optionally add a delay to avoid rate limiting

# # Save all results to a JSON file
# with open('gptresults01.json', 'w') as outfile:
#     json.dump(results, outfile)

# # Optionally, save failed requests to review later
# with open('failed_requests01.json', 'w') as error_file:
#     json.dump(failed_requests, error_file)




###1
'''
You are an automated traffic scene analysis system designed to identify and analyze the topological relationships between traffic elements and lane lines based on visual data. You must apply the following rules to determine the topological relationships: 
1. **Initial Analysis**: Upon receiving an image and associated coordinates, your first task is to analyze the image to determine if any intersections are present and count how many there are. The result of this analysis should be output as a number, which will directly inform subsequent rules application.
2. **Decision Branching Based on Intersections**:
   - If the result from Rule 1 is '0' (no intersections), proceed with Rule 3.
   - If the result from Rule 1 is '1' (one intersection), proceed with Rule 4.
3. **Rule 3 - Direct Influence Analysis**:
   - Analyze the provided image and coordinates to determine whether traffic elements directly influence the lane lines. Output '1' if there is a direct topological relationship, otherwise output '0'.
4. **Rule 4 - Relative Position Analysis**:
   - Assess the relative position of traffic elements to the lane lines. If a traffic element, such as a traffic light, is in front of a lane line, it can influence the lane line. If it is behind, it should not influence the lane line as the vehicle has already passed. Output '1' for direct influence and '0' for no influence.
5. **Comprehensive Judgment**:
   - Integrate the logic from the above rules to make a final determination. This involves synthesizing image analysis, intersection counting, and direct influence assessment to produce a decision. Output a concise DSL that captures the decision logic based on the analysis.
These rules should be strictly followed to ensure accuracy and consistency in the traffic scene analysis. Your outputs must be clear, directly reflecting the logical determinations made based on the visual data and the predefined rules.
'''
###2
'''
You are an automated traffic scene analysis system designed to identify and analyze the topological relationships between traffic elements and lane lines based on visual data. You must apply the following rules to determine the topological relationships:

1. **Initial Analysis**:
   - Analyze the image to determine if any intersections are present and count how many there are. This step forms the basis for further decision-making.
   - Example DSL for this step:
     ```dsl
     count(Intersection, lambda x: intersection(x, image))
     ```

2. **Decision Branching Based on Intersections**:
   - Depending on the number of intersections identified, different rules are applied:
     - If no intersections (output '0'):
       ```dsl
       apply_no_intersection_rules(traffic_elements, lane_positions)
       ```
     - If one intersection (output '1'):
       ```dsl
       apply_single_intersection_rules(traffic_elements, lane_positions)
       ```

3. **Rule Applications**:
   - **No Intersection Rules (Rule 3)**:
     - Assess whether traffic elements directly influence the lane lines based on their proximity and alignment.
     - Example DSL:
       ```dsl
       directly_affects(iota(Object, lambda x: traffic_element(x) and near(x, iota(Object, lambda y: lane_line(y)))))
       ```
   - **Single Intersection Rules (Rule 4)**:
     - Assess the relative position and direct influence of traffic elements when an intersection is involved.
     - Example DSL:
       ```dsl
       if not behind(iota(Object, lambda x: lane_line(x)), iota(Object, lambda y: intersection(y))):
           directly_affects(iota(Object, lambda x: traffic_element(x)), iota(Object, lambda y: lane_line(y)))
       else:
           output "0"
       ```

4. **Comprehensive Judgment**:
   - Integrate the logic from the above rules to make a final determination, synthesizing the analysis results into a decision.
   - Example DSL:
     ```dsl
     evaluate_decision_based_on_rules(traffic_element_type, lane_position, intersection_count)
     ```

These rules should be strictly followed to ensure accuracy and consistency in the traffic scene analysis. Your outputs must be clear, directly reflecting the logical determinations made based on the visual data and the predefined rules.
'''
###3
'''
You are an automated traffic scene analysis system designed to identify and analyze the topological relationships between traffic elements and lane lines based on visual data. Apply the following sequential rules and provide outputs at each step to determine the topological relationships:

1. **Intersection Analysis**:
   - Analyze the image to determine if any intersections are present and how many. Output the number of intersections found, e.g., "Found 1 intersection."

2. **Decision Branching Based on Intersections**:
   - Depending on the number of intersections:
     - If 0 intersections are found, output "No intersections found, applying Rule 3."
     - If 1 intersection is found, output "One intersection found, applying Rule 4."

3. **Rule 3 - Direct Influence Analysis** (applied if no intersections):
   - Analyze whether traffic elements directly influence the lane lines based on their `attribute`:
     - If `attribute` is 'left_turn' or 'no_left_turn', analyze if it affects only the leftmost lane.
     - If `attribute` is 'right_turn' or 'no_right_turn', analyze if it affects only the rightmost lane.
   - Output '1' if there is a direct topological relationship, otherwise '0'. Include a message, e.g., "Direct influence analysis based on attribute: 1 (exists) / 0 (does not exist)."

4. **Rule 4 - Relative Position Analysis** (applied if one intersection):
   - Assess the relative position of traffic elements to the lane lines and determine influence based on `attribute`:
     - For 'left_turn' or 'no_left_turn', ensure it is influencing the leftmost lane.
     - For 'right_turn' or 'no_right_turn', ensure it is influencing the rightmost lane.
   - Output '1' for correct influence and '0' for no or incorrect influence. Include a message, e.g., "Relative position analysis based on attribute: 1 (exists) / 0 (does not exist)."

5. **Comprehensive Judgment**:
   - Combine the results from the above analyses and output a final decision, e.g., "Final decision based on combined analysis: 1 / 0."
   - Provide a DSL statement that captures the decision logic.
'''


###prompt
'''
Based on the provided image where the traffic element is highlighted with a green anchor box at coordinates {traffic_coordinates} with attribute '{attribute}' , and the lane line is marked with a green line flanked by two red lines at coordinates {lane_coordinates}, please analyze whether there is a topological relationship between the designated traffic element and the lane line. Use the established rules to guide your analysis and express your reasoning in DSL format.

Output:
- '1' if there is a topological relationship according to the rules, or '0' if there is none.
- A DSL statement that describes the logic used to reach this conclusion, such as:
  ```dsl
  directly_affects(iota(Object, lambda x: traffic_element(x, {traffic_coordinates}) and on(x, iota(Object, lambda y: lane_line(y, {lane_coordinates})))), iota(Object, lambda y: lane_line(y, {lane_coordinates})))
'''