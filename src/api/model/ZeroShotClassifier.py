from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2

# Load the CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

class UrbanEnviromentClassifier:
    def __init__(self, model_name="openai/clip-vit-large-patch14-336", processor_name="openai/clip-vit-large-patch14-336"):
        self.model = model  # Assign the loaded model to the instance
        self.processor = processor  # Assign the loaded processor to the instance
        self.device = device  # Set device (GPU if available, otherwise CPU)
        self.labels = ["person", "vehicle", "person with knife", "on fire", "person with gun", "graffiti", "person graffiting", "unknown"]

    def clean_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, image):
        predClass = "Other"
        predProb = 0.0
        try:
            # Convert OpenCV image to PIL image
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True).to(self.device)

            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            pred_index = int(torch.argmax(probs))
            predClass = self.labels[pred_index]
            predProb = float(torch.max(probs).item() * 100)
        except Exception as e:
            print("Error: " + str(e))
            pass
        
        # Clean the cache on the GPU memory
        self.clean_gpu_memory()
        
        return {"predClass": str(predClass), "predClassProb": str(predProb)}
