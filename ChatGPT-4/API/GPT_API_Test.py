from openai import OpenAI
import os
import base64
import csv

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "Your API key"))
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
Dataset_path=('Dataset/test2')
image_folders = os.listdir(Dataset_path)
image_names=[]
image_files = []
image_base64_urls = []

for image_folder in image_folders:
    for filename in os.listdir(image_folder):
        if filename.endswith(('jpg', 'png', 'jpeg')):
            file_path = os.path.join(image_folder, filename)
            image_files.append(file_path)
            image_names.append(filename)
            image_base64_urls.append(encode_image(file_path))
requests = [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Let's say you're a dermatologist, and as a student, I'd like you to answer my proposed multiple-choice question: Do you think the abnormal part in this dermoscopic image indicates melanoma, melanocytic nevi, or benign keratosis-like lesions? Please choose one of the three answers."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64_url}"
                        },
                    },
                ],
            }
    for image_base64_url in image_base64_urls
]
csv_file = 'API_result//tem0_response_data.csv'
file_exists = os.path.isfile(csv_file)
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Image_id', 'Image_name', 'Response1', 'Response2'])

    for image_id, (image_name, request) in enumerate(zip(image_names, requests), start=1):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                request
            ],
            #max_tokens = 200,
            temperature=0
        )
        response1 = response.choices[0].message.content
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "assistant", "content": response1},
                {"role": "user",
                 "content": "Answer with the shortest possible phrase: what type of disease is identified in this content?"}
            ],
            temperature=0
        )
        response2 = response.choices[0].message.content
        writer.writerow([image_id, image_name, response1, response2])
