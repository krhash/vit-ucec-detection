# vit-ucec-detection
Compare the performance of vision transformer with ResNet for detection of Uterine corpus endometrial carcinoma


## Usage:

### Process SVS Slides and Train Vision Transformer
Note: UCEC directory with SVS files must be present

``python vit_pipeline.py``


### Train ResNet CNN model
### Run VIT Pipeline to process svs slides and save pngs before training resnet
``python resnet_pipeline.py``

### Preprocess : Slide UID to Slide name mapping using selenium
slide_uids.sjon is generated using colab: https://colab.research.google.com/drive/1hTa1f8V6SRzCaHQcT5BXbVoGHLFh8nMD#scrollTo=bvlrYBeiTdFC

``python ./label_preprocessing/map_slide_name_to_type.py``


### Model Output directory: 
out_model

# Pixels to Prognosis: Deep Learning for Uterine Cancer Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project explores the use of deep learning for automated detection of uterine (endometrial) cancer from histopathology whole slide images (WSIs). We benchmark multiple neural network architectures—including DenseNet, ResNet, EfficientNet, and Vision Transformers (ViT)—and present a robust pipeline for patch extraction, preprocessing, class balancing, and model evaluation.

## Key Features

- **Comprehensive Data Pipeline:**  
  - Patch extraction from WSIs  
  - Quality filtering and tissue detection  
  - Data augmentation and class balancing

- **Model Benchmarking:**  
  - DenseNet-121  
  - ResNet-34  
  - EfficientNet-B0  
  - Vision Transformer (ViT)

- **Performance Highlights:**  
  - DenseNet and ResNet achieve >99% accuracy  
  - CNNs outperform ViT and EfficientNet on limited medical data

- **Reproducible Training and Evaluation:**  
  - Weighted loss for class imbalance  
  - Mixed-precision training  
  - ROC, PR curves, and confusion matrix analysis

## Getting Started

1. **Clone the repository:**
```bash
  git clone https://github.com/krhash/vit-ucec-detection.git
  cd vit-ucec-detection
```

2. **Install dependencies:**
```bash
  pip install -r requirements.txt
```


3. **Prepare data:**  
- Download WSIs and labels as described in the `data/` folder README.
- Run patch extraction and preprocessing scripts.

4. **Train a model:**
```bash
  python models/train_densenet.py
```


5. **Evaluate results:**
- Use scripts in `notebooks/` for analysis and visualization.

## Results

- **DenseNet-121:** 99.15% accuracy, AUROC = 1.00
- **ResNet-34:** 99.10% accuracy, AUROC = 1.00
- **EfficientNet-B0, ViT:** Underperformed due to limited data

See the [Project Report](./research_report/Project_Report.pdf) for full details.

## Project Poster

[View Poster PDF](./research_report/Poster.pdf)

## Citation

If you use this code or data, please cite:

```text
Sharma, K.S., Bommireddy, N., Gu, Z. (2023).
Pixels to Prognosis: Deep Learning for Uterine Cancer Detection.
https://github.com/krhash/vit-ucec-detection
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Contact:**  
For questions or collaboration, please open an issue or contact [Krushna Sharma](mailto:krushnasharma24@gmail.com).
