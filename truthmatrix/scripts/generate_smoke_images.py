from pathlib import Path
from PIL import Image, ImageDraw
import random


def count_files(path: Path) -> int:
    return sum(1 for f in path.glob("*") if f.is_file())


def create_samples(folder: Path, label: str, n: int) -> None:
    for i in range(n):
        if label == "fake":
            base = (random.randint(120, 255), random.randint(10, 120), random.randint(10, 120))
            outline = (255, 255, 0)
            text = f"FAKE {i}"
        else:
            base = (random.randint(10, 120), random.randint(120, 255), random.randint(10, 120))
            outline = (0, 255, 255)
            text = f"REAL {i}"

        img = Image.new("RGB", (128, 128), base)
        draw = ImageDraw.Draw(img)
        draw.text((10, 54), text, fill=(255, 255, 255))
        draw.rectangle((4, 4, 124, 124), outline=outline, width=2)
        img.save(folder / f"{label}_{i}.png")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "images"
    fake_dir = root / "fake"
    real_dir = root / "real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    fake_count = count_files(fake_dir)
    real_count = count_files(real_dir)

    if fake_count == 0:
        create_samples(fake_dir, "fake", 24)
    if real_count == 0:
        create_samples(real_dir, "real", 24)

    print(f"fake_count={count_files(fake_dir)}")
    print(f"real_count={count_files(real_dir)}")


if __name__ == "__main__":
    main()
