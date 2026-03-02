# Multimodal Fake Content Detection

This project detects fake content using:
- a **text classifier** (TF-IDF + Logistic Regression)
- an **image classifier** (MobileNetV2 transfer learning)
- a **multimodal fusion** rule (average probability)

## 1) Environment Setup

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r fake-content-detection/requirements.txt
```

## 2) Dataset Structure

Create the following structure before training:

```text
fake-content-detection/
├── data/
│   ├── raw/
│   │   └── news.csv
│   └── images/
│       ├── real/
│       │   ├── img1.jpg
│       │   └── ...
│       └── fake/
│           ├── img1.jpg
│           └── ...
```

### `news.csv` format
- Required columns:
  - `text`
  - `label` (`0` = real, `1` = fake)

## 3) Training Commands

### Train Text Model

```bash
python fake-content-detection/src/train_text_model.py --data fake-content-detection/data/raw/news.csv --model-dir fake-content-detection/models
```

### Train Image Model

```bash
python fake-content-detection/src/train_image_model.py --data-dir fake-content-detection/data/images --output-model fake-content-detection/models/image_model.h5 --epochs 5
```

## 4) Inference Command (End-to-End Multimodal)

```bash
python fake-content-detection/src/predict_multimodal.py --model-dir fake-content-detection/models
```

The script prompts for:
- news text input
- image path

## 5) Expected Output Files After Training

```text
fake-content-detection/
├── models/
│   ├── text_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── image_model.h5
├── reports/
│   ├── text_metrics.txt
│   └── image_metrics.txt
└── outputs/
    ├── text_confusion_matrix.png
    └── image_training_plot.png
```

Reports include the fixed random seed value (`Seed: 42`) for reproducibility.

## 6) Example CLI Usage

```text
$ python fake-content-detection/src/predict_multimodal.py
Enter news text: Government confirms new climate policy implementation.
Enter image path: fake-content-detection/data/images/fake/sample1.jpg
Text probability (fake): 0.2314
Image probability (fake): 0.8129
Final fused decision: Fake (score: 0.5222)
```

## 7) Notes

- Both training scripts validate required dataset paths.
- Inference validates required model artifacts before loading.
- If required files are missing, the scripts raise clear error messages.
