import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

IMAGE_SIZE = 224
RESCALE = 1 / 255.0
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

def preprocess_image(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to numpy array
    image_array = np.array(image, dtype=np.float32)
    
    # Rescale
    image_array = image_array * RESCALE
    
    # Normalize
    image_array = (image_array - MEAN) / STD
    
    # Change from HWC to CHW
    image_array = np.transpose(image_array, (2, 0, 1))
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def run_inference(model_path, image_path):
    # Load model
    session = ort.InferenceSession(model_path)
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    
    return outputs[0]

if __name__ == "__main__":
    model_path = "model.onnx"
    image_path = "test_image.jpg"
    
    result = run_inference(model_path, image_path)
    print(f"Output shape: {result.shape}")
    print(f"Predictions: {result}")
