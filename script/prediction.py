from PIL import Image
import torch
import os 

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Class images to classify
labels=["person", "vehicle", "person with knife", "on fire","person with gun", "graffiti","person graffiting", "unknown"]
image_folder_path='./images/'
for image in os.listdir(image_folder_path):
    url = image_folder_path+image
    
    print("Image name: {}\n".format(image))
    image_format= str(image).split(".")[1]
    orig_image_name= str(image).split(".")[0]
    image = Image.open(url)
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    pred_index= int(torch.argmax(probs))
    print(labels[pred_index], pred_index, float(torch.max(probs).item()*1e2))
    try:
        image.save("./predictedImages/{}_{}.{}".format(labels[pred_index],orig_image_name, image_format))
    except Exception as e:
        print(e)
        