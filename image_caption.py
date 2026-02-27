from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load pretrained model and processors
print("Loading model... Please wait.")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model= BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image
image_path = "sample.jpg"

# <-- Put your image name here
image = Image.open(image_path).convert("RGB")

# Process image
inputs = processor(image, return_tensors="pt")

# Generate caption
print("Generating caption...")
output = model.generate(**inputs)

# Decode caption
caption = processor.decode(output[0],
                                 skip_special_tokens = True)

print("\n Generated Caption:")
print(caption)
