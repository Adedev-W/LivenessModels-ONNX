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
- **Multi-language support**: Python, JavaScript, C#, Java, and C++
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
