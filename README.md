# LivenessModels-ONNX
This repository contains a pretrained ONNX model for easy integration into your projects. ONNX (Open Neural Network Exchange) provides interoperability between different deep learning frameworks, making this model portable and production-ready.

Model source:
https://drive.google.com/file/d/1V71oWHVlj3_umCgOpAsnrEbVHdVW0rZP/view?usp=sharing

## Load Metatag
```python
import onnx
#Load ONNX model
model = onnx.load("liveness_vit_with_meta.onnx")
#Print all metadata (key-value)
print("Model Metadata:")
for prop in model.metadata_props:
    print(f"{prop.key}: {prop.value}")

```

`python
:Output`
```cpp
Model Metadata:
id2label: {"0": "real", "1": "spoof"}
label2id: {"real": 0, "spoof": 1}
```

# How to use on 

