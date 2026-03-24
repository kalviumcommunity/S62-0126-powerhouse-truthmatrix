from __future__ import annotations

import argparse
from pathlib import Path

from src.fusion.pipeline import unified_predict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal Fake News Detector CLI",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="News text to analyze.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file to analyze.",
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.5,
        help="Weight for text prediction when fusing results.",
    )
    parser.add_argument(
        "--image-weight",
        type=float,
        default=0.5,
        help="Weight for image prediction when fusing results.",
    )
    return parser.parse_args()


def _prompt_if_missing(value: str | None, prompt: str) -> str | None:
    if value is not None and value.strip():
        return value.strip()

    user_value = input(prompt).strip()
    return user_value if user_value else None


def main() -> None:
    args = _parse_args()

    print("\n=== Multimodal Fake News Detection CLI ===")
    print("You can provide text, image path, or both.")

    text = _prompt_if_missing(args.text, "Enter news text (or press Enter to skip): ")
    image_input = _prompt_if_missing(args.image, "Enter image path (or press Enter to skip): ")
    image_path = Path(image_input) if image_input else None

    if text is None and image_path is None:
        print("\nNo input provided. Please provide text and/or an image path.")
        return

    result = unified_predict(
        text=text,
        image_path=image_path,
        text_weight=args.text_weight,
        image_weight=args.image_weight,
    )

    final = result.get("final") or {}
    label = final.get("label")
    confidence = final.get("confidence")
    source = final.get("source")

    if label is None or confidence is None:
        print("\nPrediction could not be generated.")
    else:
        print("\n--- Final Prediction ---")
        print(f"Label      : {label}")
        print(f"Confidence : {confidence:.4f}")
        print(f"Source     : {source}")

    warnings = result.get("warnings", [])
    if warnings:
        print("\n--- Notes ---")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
