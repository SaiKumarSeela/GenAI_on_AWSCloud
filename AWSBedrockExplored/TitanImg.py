import base64
import boto3
import json
import os
import random

prompt_data = """
 Provide me a 4k hd image of a beach, also use a blue sky rainy season an 
 cinematic display
"""
bedrock = boto3.client(service_name="bedrock-runtime", region_name="ap-south-1")

# Set the model ID, e.g., Titan Image Generator G1.
model_id = "amazon.titan-image-generator-v1"
native_request = {"textToImageParams":{"text": prompt_data},"taskType":"TEXT_IMAGE","imageGenerationConfig":{"cfgScale":8,"seed":0,"width":1024,"height":1024,"numberOfImages":3}}

# Convert the native request to JSON.
request = json.dumps(native_request)

# Invoke the model with the request.
response = bedrock.invoke_model(modelId=model_id, body=request)

# Decode the response body.
model_response = json.loads(response["body"].read())

# Extract the image data.
base64_image_data = model_response["images"][0]

# Save the generated image to a local folder.
i, output_dir = 1, "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
while os.path.exists(os.path.join(output_dir, f"image_{i}.png")):
    i += 1

image_data = base64.b64decode(base64_image_data)

image_path = os.path.join(output_dir, f"image_{i}.png")
with open(image_path, "wb") as file:
    file.write(image_data)

print(f"The generated image has been saved to {image_path}.")



