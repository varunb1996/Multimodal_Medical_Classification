# Multimodal Medical Document Classification

This project fuses text and image modalities to classify scanned medical documents.

## Technologies
- BERT (Hugging Face Transformers)
- ResNet (Torchvision)
- PyTorch

## Features
- Uses BERT to encode clinical text
- ResNet to encode image regions
- Fully-connected fusion layer for classification

## Example Input
- Text: "Patient shows signs of pulmonary infection and elevated CRP levels."
- Image: Radiology report (grayscale scan)

## Example Output
```json
{
  "class_names": ["Radiology", "Prescription", "Pathology"],
  "prediction": "Radiology",
  "confidence": 0.89
}
```

## How to Run
```bash
python multimodal_model.py
```