# LivenessModels-ONNX

[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![C#](https://img.shields.io/badge/C%23-239120?style=flat-square&logo=c-sharp&logoColor=white)](https://docs.microsoft.com/en-us/dotnet/csharp/)
[![Java](https://img.shields.io/badge/Java-ED8B00?style=flat-square&logo=java&logoColor=white)](https://www.oracle.com/java/)
[![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Model Size](https://img.shields.io/badge/Model%20Size-~327,5MB-blue?style=flat-square)](https://drive.google.com/file/d/1V71oWHVlj3_umCgOpAsnrEbVHdVW0rZP/view)
[![Framework](https://img.shields.io/badge/Framework-Vision%20Transformer-orange?style=flat-square)](https://arxiv.org/abs/2010.11929)

A production-ready ONNX model for face liveness detection using Vision Transformer architecture. This model can distinguish between real and spoofed faces, making it suitable for anti-spoofing applications in facial recognition systems.

## Table of Contents

- [Overview](#overview)
- [Model Information](#model-information)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Metadata](#model-metadata)
- [Preprocessing Configuration](#preprocessing-configuration)
- [Multi-Language Implementation](#multi-language-implementation)
  - [Python](#python)
  - [C#](#c)
  - [Java](#java)
  - [C++](#c-1)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository provides a production-ready ONNX model for face liveness detection. The model is based on Vision Transformer (ViT) architecture and has been optimized for cross-platform deployment with ONNX Runtime.

### Key Features

- **Cross-platform compatibility**: Runs on Windows, macOS, and Linux
- **Multi-language support**: Python, C#, Java, and C++
- **Production-ready**: Optimized ONNX format for efficient inference
- **Easy integration**: Simple API for quick deployment
- **Real-time capable**: Suitable for real-time applications
- **Embedded metadata**: Built-in label mappings for easy interpretation

## Model Information

| Property | Value |
|----------|-------|
| **Architecture** | Vision Transformer (ViT) |
| **Input Size** | 224 x 224 x 3 |
| **Output Classes** | 2 (real, spoof) |
| **Model Format** | ONNX |
| **Model Size** | 327,5 MB |
| **Precision** | FP32 |
| **Framework** | PyTorch to ONNX |

### Model Download

Download the pretrained model from Google Drive:[![Model Size](https://img.shields.io/badge/Click-here-blue?style=flat-square)](https://drive.google.com/file/d/1V71oWHVlj3_umCgOpAsnrEbVHdVW0rZP/view)


## Installation

### Python
```bash
pip install onnxruntime opencv-python numpy pillow onnx
```

### C#
```bash
dotnet add package Microsoft.ML.OnnxRuntime
dotnet add package SixLabors.ImageSharp
```

### Java
```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.16.3</version>
</dependency>
```

### C++
```bash
# Download ONNX Runtime dari GitHub releases
# Link: https://github.com/microsoft/onnxruntime/releases
```


## Quick Start

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load the model
session = ort.InferenceSession("liveness_vit_with_meta.onnx")

# Load and preprocess image
image = Image.open("face_image.jpg").convert('RGB')
image = image.resize((224, 224))
image_array = np.array(image, dtype=np.float32)

# Apply preprocessing
image_array = image_array / 255.0
image_array = (image_array - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]
image_array = np.transpose(image_array, (2, 0, 1))
image_array = np.expand_dims(image_array, axis=0)

# Run inference
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: image_array})

# Get prediction
prediction = np.argmax(outputs[0])
confidence = np.max(outputs[0])

print(f"Prediction: {'real' if prediction == 0 else 'spoof'}")
print(f"Confidence: {confidence:.4f}")
```

## Model Metadata

The model contains embedded metadata that provides label mappings:

```python
import onnx

# Load ONNX model
model = onnx.load("liveness_vit_with_meta.onnx")

# Print all metadata (key-value pairs)
print("Model Metadata:")
for prop in model.metadata_props:
    print(f"{prop.key}: {prop.value}")
```

**Output:**
```python
Model Metadata:
id2label: {"0": "real", "1": "spoof"}
label2id: {"real": 0, "spoof": 1}
```


### Label Mapping

| ID | Label | Description |
|----|-------|-------------|
| 0 | real | Genuine/live face detected |
| 1 | spoof | Fake/spoofed face detected |

## Preprocessing Configuration

The model requires specific preprocessing parameters:

```python
{
  "image_size": 224,
  "rescale": 0.00392156862745098,
  "mean": [0.5, 0.5, 0.5],
  "std": [0.5, 0.5, 0.5],
  "color_format": "RGB"
}
```

### Preprocessing Steps

1. **Resize**: Scale image to 224x224 pixels
2. **Rescale**: Divide pixel values by 255.0
3. **Normalize**: Apply mean subtraction and standard deviation normalization
4. **Format**: Convert from HWC to CHW format
5. **Batch**: Add batch dimension [1, 3, 224, 224]

## Multi-Language Examples

For complete implementation examples in multiple programming languages, please refer to our comprehensive examples repository:

### 📂 [View Examples on GitHub](https://github.com/Adedev-W/LivenessModels-ONNX/tree/main/Example)

The examples repository contains fully working implementations for:

- **[Python](https://github.com/Adedev-W/LivenessModels-ONNX/blob/main/Example/ONNX-python.py)** - Complete inference pipeline with OpenCV integration
- **[C#](https://github.com/Adedev-W/LivenessModels-ONNX/blob/main/Example/ONNX-charp.cs)** - .NET application integration
- **[Java](https://github.com/Adedev-W/LivenessModels-ONNX/blob/main/Example/ONNX-java.java)** - Cross-platform Java applications
- **[C++](https://github.com/Adedev-W/LivenessModels-ONNX/blob/main/Example/ONNX-cpp.cpp)** - High-performance native applications

Each example includes:
- Complete source code with proper preprocessing
- Build/run instructions
- Error handling and validation
- Performance optimizations
- Unit tests

## Citation

If you use this model in your research or commercial applications:

```bibtex
@misc{liveness-models-onnx-2024,
  title={LivenessModels-ONNX: Production-Ready Face Liveness Detection},
  author={Adedev-W and Contributors},
  year={2024},
  publisher={GitHub},
  journal={GitHub Repository},
  howpublished={\url{https://github.com/Adedev-W/LivenessModels-ONNX}},
  note={Vision Transformer-based ONNX model for cross-platform deployment}
}
```


## Acknowledgments

- **ONNX Community**: For the excellent cross-platform inference framework
- **Vision Transformer Authors**: For the transformer architecture innovation
- **PyTorch Team**: For the original model training framework  
- **OpenCV Contributors**: For comprehensive image processing tools
- **Community Contributors**: For testing, feedback, and improvements

---

**Built for production deployment across multiple platforms and programming languages**
