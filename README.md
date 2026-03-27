# Multimodal Fake Content Detection using Text and Image Fusion

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#tech-stack)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](#features)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

Production-oriented AI system for detecting potentially fake content by combining text and image intelligence. The project delivers classification, confidence scoring, explainability, and analytics in a unified Streamlit interface.

## Problem Statement
Misinformation now spreads through both language and visual manipulation. Single-modality systems often miss cross-signal inconsistencies, resulting in false negatives or low-confidence predictions.

This project addresses that gap with a multimodal pipeline that:
1. Detects fake vs real signals in text.
2. Detects fake vs real signals in images.
3. Fuses model outputs into a final decision.
4. Provides interpretable scoring and analysis for trust-aware decision making.

## Features
1. Multimodal AI fusion:
Text and image models are combined into one final prediction using confidence-aware fusion logic.
2. Truth Score:
Computes a weighted score from text, image, and fusion confidence to produce a 0-100 reliability signal.
3. Credibility Score:
Scores content based on sentiment polarity, exaggerated language, and model confidence.
4. Explainability engine:
Generates reasons behind classification (clickbait terms, emotional tone, pattern mismatch cues).
5. Analytics dashboard:
Includes distribution, model comparison, and confusion matrix visualizations.
6. Production-style UI:
Multi-tab Streamlit app for detection, analytics, and model insights.

## Tech Stack
1. Language: Python
2. ML/NLP: scikit-learn, NumPy, pandas
3. Deep Learning/CV: TensorFlow, PIL
4. Visualization: Matplotlib
5. App Layer: Streamlit
6. Experimentation: Jupyter Notebook

## Architecture Diagram (Text-Based)
```text
                    +-------------------------+
                    |   User Input Layer      |
                    |  - Text Content         |
                    |  - Uploaded Image       |
                    +-----------+-------------+
                                |
             +------------------+------------------+
             |                                     |
   +---------v---------+                 +---------v---------+
   | Text Pipeline     |                 | Image Pipeline    |
   | - preprocessing   |                 | - preprocessing   |
   | - text model      |                 | - CNN model       |
   +---------+---------+                 +---------+---------+
             |                                     |
             +------------------+------------------+
                                |
                     +----------v----------+
                     |  Fusion Engine      |
                     |  - agreement logic  |
                     |  - confidence logic |
                     +----------+----------+
                                |
       +------------------------+------------------------+
       |                                                 |
+------v----------------+                 +--------------v--------------+
| Scoring & Explanation |                 | Analytics & Visualization   |
| - Truth Score (0-100) |                 | - Fake vs Real pie chart    |
| - Credibility Score   |                 | - Accuracy comparison bar   |
| - Risk Level          |                 | - Confusion matrix          |
+------+----------------+                 +--------------+--------------+
       |                                                  |
       +------------------------+-------------------------+
                                |
                     +----------v----------+
                     | Streamlit UI        |
                     | Tabs:               |
                     | 1) Detect Content   |
                     | 2) Analytics        |
                     | 3) Model Insights   |
                     +---------------------+
```

## Screenshots
Add real screenshots after UI deployment:

1. Detect Content tab
![Detect Content](docs/screenshots/detect-content.png)

2. Analytics tab
![Analytics](docs/screenshots/analytics-dashboard.png)

3. Model Insights tab
![Model Insights](docs/screenshots/model-insights.png)

4. Scoring cards (Truth + Credibility + Risk)
![Scoring Cards](docs/screenshots/scoring-cards.png)

## Project Structure
```text
S62-0126-powerhouse-truthmatrix/
├── README.md
└── truthmatrix/
    ├── requirements.txt
    ├── cli.py
    ├── app/
    │   └── streamlit_app.py
    ├── notebooks/
    │   └── eda.ipynb
    └── src/
        ├── train.py
        ├── predict.py
        ├── explain.py
        ├── evaluate.py
        ├── preprocess.py
        ├── data_loader.py
        ├── fusion/
        │   ├── fusion.py
        │   └── pipeline.py
        └── image/
            ├── model.py
            ├── train.py
            ├── predict.py
            ├── preprocess.py
            └── data_loader.py
```

## How to Run Locally
### 1. Clone the repository
```bash
git clone https://github.com/your-org/your-repo.git
cd S62-0126-powerhouse-truthmatrix
```

### 2. Create and activate virtual environment
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
cd truthmatrix
pip install -r requirements.txt
```

### 4. Launch Streamlit app
```bash
streamlit run app/streamlit_app.py
```

### 5. Optional: run CLI
```bash
python cli.py
```

## Evaluation and Metrics
1. Accuracy, precision, recall, F1 score
2. Confusion matrix analysis
3. Class-wise confidence tracking
4. Aggregate credibility and truth score trends

## Future Improvements
1. Calibrated uncertainty estimation across modalities.
2. Transformer-based text encoder and vision transformer backbone.
3. Active learning loop for human-in-the-loop moderation.
4. Adversarial robustness testing for prompt/image perturbations.
5. Drift monitoring and automatic re-training pipeline.
6. REST API and containerized deployment with CI/CD.
7. Role-based dashboards for analysts and moderators.

## License
MIT License (recommended). Update this section based on your repository license.

## Acknowledgments
Built for practical misinformation detection research, rapid experimentation, and explainable AI workflows.
