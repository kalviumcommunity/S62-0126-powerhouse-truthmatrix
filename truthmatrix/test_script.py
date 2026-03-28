import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent / "truthmatrix"
sys.path.append(str(PROJECT_ROOT))

from src.train import load_model
from src.predict import predict as predict_text
model_path = PROJECT_ROOT / "models" / "model.pkl"
text_model = load_model(model_path)
label, confidence = predict_text("Test simple text", text_model)
print(f"Raw Label: {label}")
mapped_label = "Real" if str(label) == "1" else "Fake" if str(label) == "0" else label.title()
print(f"Mapped Label: {mapped_label}")
