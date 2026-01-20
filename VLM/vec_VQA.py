from openai import OpenAI
import os
import base64
import time

api_key="sk-7zZMomkPoQGrnxe1Xqar5O17AfaqShEMALVqucYCIuzr6dr5"
client = OpenAI(
    api_key=api_key,
    base_url="https://aizex.top/v1",
    )

system_intel = '''
In the provided bird's-eye view (BEV), the green and blue lane segments are highlighted. The white arrow represents the driving direction of the green segment, and the magenta arrow points from the mid point of the green segment to that of the blue segment. At the top right corner of the image, two arrows with the same directions are drawn in white and magenta, which can be referred to.
The red lines in the photos are lane boundaries, and the part of black is gap segments or the context.
Normally or when the case is confusing, two directions with deviation of less than 45 degrees are considered as compatible. 
'''

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to call GPT-4 API
def ask_GPT4(prompt, image_base64):
    response = client.chat.completions.create(model='gpt-4o',
    messages=[
              {"role": "system", "content": [
                    {"type": "text", "text": system_intel},
                ]},
              {"role": "user", "content":[{
                    "type":"text",
                    "text":prompt
                    },
                    {
                    "type":"image_url",
                    "image_url":{
                        "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                    ]
                }
                ],
    max_tokens=100,
    temperature=0)
    return response.choices[0].message.content

def main():
  work_path = "./dataset/VQA/vector/"
  result_path = "./dataset/gpt4o/vec_result.txt"
  results = ""

  #   # results = {}
  #   # failed_requests = []

  for scene in range(10001, 10003):
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
      for img in imgs:
          try:
              img_name = img.split('.png')[0]
              img_path = os.path.join(scene_path, img)
              print('Encoding image...')
              base64_image = encode_image(img_path)
              print('Image encoded.')
              prompt = f'''
  You are an expert in determining the relationships between lane segments in the image. Let's determine if the directions of the two arrows match. Please reply in a brief sentence starting with "Yes" or "No".
          '''
              # Call GPT-4 API
              print('Calling gemini-2.5-flash API...')
              result = ask_GPT4(prompt, base64_image)
              print('API call successful.')
              print(result)
              if "Yes" in result :
                  label = '1'
              else:
                  label = '0'

              line = scene + " " + img_name + " " + label + "\n"
              print(line)
              results += line
          except Exception as e:
              print(f'Error processing scene {scene}: {e}')
              print('Moving to next sample...')
              time.sleep(2)  # Optionally add a delay to avoid rate limiting

  with open(result_path, "w") as f:
      f.write(results)


if __name__ == '__main__':
  main()