# -*- coding: utf-8 -*-
"""Часть 1 SmartParking: скачать датасеты и оставить готовые выборки.

Запуск из корня репозитория:

    python src/prepare_data.py

Повторный запуск ничего не качает, если отчёт уже собран.
Полная пересборка: python src/prepare_data.py --force
Проверка на одном test.zip, без 7 ГБ: python src/prepare_data.py --quick
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_RANK = {"train": 0, "val": 1, "test": 2}
PLATE_PATTERN = re.compile(r"^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}(\d{2,3})$")
CYR_TO_LAT = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
    }
)

DETECTION_REPO = "AY000554/Car_plate_detecting_dataset"
OCR_REPO = "AY000554/Car_plate_OCR_dataset"
ALL_ZIPS = ("train.zip", "val.zip", "test.zip")
QUICK_ZIPS = ("test.zip",)
QUALITY_SAMPLE_FULL = 1500
QUALITY_SAMPLE_QUICK = 200
SEED = 42


@dataclass
class DetItem:
    split: str
    image: Path
    label: Path
    width: int
    height: int
    boxes: int
    plate_area: float
    digest: str = ""


@dataclass
class OcrItem:
    split: str
    original_split: str
    image: Path
    text: str
    width: int
    height: int
    valid_format: bool
    digest: str = ""


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def say(text: str) -> None:
    print(text, flush=True)


def ignored(path: Path) -> bool:
    return any(part.startswith(".") or part == "__MACOSX" for part in path.parts)


def normalize_plate(text: str) -> str:
    cleaned = "".join(text.upper().split())
    cleaned = re.sub(r"\(\d+\)$", "", cleaned)
    return cleaned.translate(CYR_TO_LAT)


def valid_plate(text: str) -> bool:
    return bool(PLATE_PATTERN.fullmatch(normalize_plate(text)))


def region_code(text: str) -> str | None:
    match = PLATE_PATTERN.fullmatch(normalize_plate(text))
    if match is None:
        return None
    return match.group(1)


def choose_split(splits: list[str]) -> str:
    counts = Counter(splits)
    return min(counts, key=lambda name: (-counts[name], SPLIT_RANK[name]))


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_size(path: Path) -> tuple[int, int] | None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
            return int(image.size[0]), int(image.size[1])
    except Exception:
        return None


def parse_yolo(path: Path) -> tuple[int, float] | None:
    boxes = 0
    area = 0.0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="latin-1").splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            x_center, y_center, width, height = (float(part) for part in parts[1:5])
        except ValueError:
            continue
        if not (0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
            continue
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            continue
        boxes += 1
        area += width * height
    if boxes == 0:
        return None
    return boxes, area


def brightness_and_blur(path: Path) -> tuple[float, float] | None:
    import cv2
    import numpy as np

    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    brightness = float(image.mean())
    blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return brightness, blur


def download_archives(root: Path, quick: bool) -> None:
    from huggingface_hub import hf_hub_download

    names = QUICK_ZIPS if quick else ALL_ZIPS
    jobs = (
        (DETECTION_REPO, root / "data" / "raw" / "zips" / "detection"),
        (OCR_REPO, root / "data" / "raw" / "zips" / "ocr"),
    )
    for repo_id, dest in jobs:
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("license.txt", "README.md", *names):
            say(f"Проверяю {repo_id} / {name}")
            hf_hub_download(
                repo_id=repo_id,
                filename=name,
                repo_type="dataset",
                local_dir=dest,
            )


def safe_extract(zip_path: Path, dest: Path) -> None:
    marker = dest / ".extracted"
    if marker.exists() and any(dest.rglob("*")):
        say(f"Уже распакован: {dest}")
        return
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    say(f"Распаковываю {zip_path.name} → {dest}")
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            parts = Path(name).parts
            if name.startswith("/") or ".." in parts:
                raise RuntimeError(f"Небезопасный путь в архиве: {name}")
        archive.extractall(dest)
    marker.write_text(zip_path.name, encoding="utf-8")


def extract_archives(root: Path, quick: bool) -> None:
    names = QUICK_ZIPS if quick else ALL_ZIPS
    for kind in ("detection", "ocr"):
        zip_dir = root / "data" / "raw" / "zips" / kind
        for name in names:
            split = name.removesuffix(".zip")
            safe_extract(zip_dir / name, root / "data" / "raw" / kind / split)


def iter_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    found = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not ignored(path)
    ]
    return sorted(found)


def label_index(folder: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    if not folder.exists():
        return found
    for path in folder.rglob("*.txt"):
        if path.is_file() and not ignored(path):
            found[path.stem].append(path)
    return found


def match_label(image: Path, labels: dict[str, list[Path]]) -> Path | None:
    sibling = image.with_suffix(".txt")
    if sibling.exists():
        return sibling
    options = labels.get(image.stem, [])
    if not options:
        return None
    same_dir = [path for path in options if path.parent == image.parent]
    if len(same_dir) == 1:
        return same_dir[0]
    if len(options) == 1:
        return options[0]
    labels_dir = [path for path in options if path.parent.name.lower() == "labels"]
    if len(labels_dir) == 1:
        return labels_dir[0]
    return options[0]


def collect_detection(root: Path, quick: bool) -> tuple[list[DetItem], dict[str, int]]:
    from tqdm import tqdm

    names = QUICK_ZIPS if quick else ALL_ZIPS
    items: list[DetItem] = []
    stats = Counter()
    for name in names:
        split = name.removesuffix(".zip")
        folder = root / "data" / "raw" / "detection" / split
        labels = label_index(folder)
        images = iter_images(folder)
        stats["files"] += len(images)
        for image in tqdm(images, desc=f"детекция {split}", unit="кадр"):
            size = read_image_size(image)
            if size is None:
                stats["broken"] += 1
                continue
            label = match_label(image, labels)
            if label is None:
                stats["no_label"] += 1
                continue
            parsed = parse_yolo(label)
            if parsed is None:
                stats["bad_label"] += 1
                continue
            boxes, area = parsed
            items.append(
                DetItem(
                    split=split,
                    image=image,
                    label=label,
                    width=size[0],
                    height=size[1],
                    boxes=boxes,
                    plate_area=area,
                )
            )
    return items, dict(stats)


def collect_ocr(root: Path, quick: bool) -> tuple[list[OcrItem], dict[str, int]]:
    from tqdm import tqdm

    names = QUICK_ZIPS if quick else ALL_ZIPS
    items: list[OcrItem] = []
    stats = Counter()
    for name in names:
        split = name.removesuffix(".zip")
        folder = root / "data" / "raw" / "ocr" / split
        images = iter_images(folder)
        stats["files"] += len(images)
        for image in tqdm(images, desc=f"ocr {split}", unit="кадр"):
            size = read_image_size(image)
            if size is None:
                stats["broken"] += 1
                continue
            text = normalize_plate(image.stem)
            items.append(
                OcrItem(
                    split=split,
                    original_split=split,
                    image=image,
                    text=text,
                    width=size[0],
                    height=size[1],
                    valid_format=valid_plate(text),
                )
            )
            if not valid_plate(text):
                stats["invalid_name"] += 1
    return items, dict(stats)


def attach_hashes(items: list, label: str) -> None:
    from tqdm import tqdm

    for item in tqdm(items, desc=f"хеши {label}", unit="файл"):
        item.digest = file_md5(item.image)


def dedupe_exact(items: list, label: str) -> tuple[list, int]:
    kept = []
    seen: set[str] = set()
    removed = 0
    ordered = sorted(items, key=lambda item: (SPLIT_RANK[item.split], str(item.image)))
    for item in ordered:
        if item.digest in seen:
            removed += 1
            continue
        seen.add(item.digest)
        kept.append(item)
    if removed:
        say(f"{label}: убрано точных копий файлов: {removed}")
    return kept, removed


def regroup_ocr(items: list[OcrItem]) -> tuple[list[OcrItem], int]:
    groups: dict[str, list[OcrItem]] = defaultdict(list)
    for item in items:
        groups[item.text].append(item)
    moved = 0
    for group in groups.values():
        target = choose_split([item.original_split for item in group])
        for item in group:
            if item.split != target:
                moved += 1
            item.split = target
    if moved:
        say(f"OCR: один и тот же номер был в разных выборках, перенесено файлов: {moved}")
    return items, moved


def sample_quality(items: list, limit: int) -> dict[str, float | int]:
    if not items:
        return {"n": 0}
    rng = random.Random(SEED)
    chosen = items if len(items) <= limit else rng.sample(items, limit)
    brightness: list[float] = []
    blur: list[float] = []
    for item in chosen:
        measured = brightness_and_blur(item.image)
        if measured is None:
            continue
        brightness.append(measured[0])
        blur.append(measured[1])
    return {
        "n": len(brightness),
        "requested": len(chosen),
        "brightness_median": _median(brightness),
        "blur_median": _median(blur),
        "brightness": brightness,
        "blur": blur,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    import numpy as np

    return float(np.median(np.asarray(values, dtype=float)))


def _font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Arial", "Calibri", "DejaVu Sans"):
        if name in available:
            plt.rcParams["font.family"] = name
            return


def _hist(values: list[float], title: str, xlabel: str, path: Path, reading: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(values, bins=40, color="#3c6e9f")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Количество")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return {"title": title, "path": str(path), "reading": reading}


def _bars(labels: list[str], values: list[int], title: str, xlabel: str, path: Path, reading: str) -> dict[str, str]:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color="#3c6e9f")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Количество")
    axis.tick_params(axis="x", labelrotation=60)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return {"title": title, "path": str(path), "reading": reading}


def build_plots(root: Path, detection: list[DetItem], ocr: list[OcrItem], det_quality: dict, ocr_quality: dict) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    _font()
    report = root / "reports"
    plots: list[dict[str, str]] = []

    widths = [item.width for item in detection]
    heights = [item.height for item in detection]
    if widths:
        plots.append(
            _hist(
                widths,
                "Ширина кадра, детекция",
                "Пиксели",
                report / "eda_detection_width.png",
                f"Медианная ширина кадра { _fmt(_median(widths)) } px, высота { _fmt(_median(heights)) } px.",
            )
        )
        plots.append(
            _hist(
                heights,
                "Высота кадра, детекция",
                "Пиксели",
                report / "eda_detection_height.png",
                "Высота кадров детекции. Модель YOLO получит кадр целиком, не кроп номера.",
            )
        )
        areas = [item.plate_area * 100 for item in detection]
        plots.append(
            _hist(
                areas,
                "Доля кадра, которую занимает номер",
                "Проценты площади кадра",
                report / "eda_plate_area.png",
                f"Медиана { _fmt(_median(areas), 2) }% кадра. Номер мелкий, для детекции стоит сравнить вход 640 и 960.",
            )
        )

    if ocr:
        plots.append(
            _hist(
                [float(item.width) for item in ocr],
                "Ширина кропа номера",
                "Пиксели",
                report / "eda_ocr_width.png",
                (
                    f"Медиана кропа { _fmt(_median([item.width for item in ocr])) }"
                    f"×{ _fmt(_median([item.height for item in ocr])) } px. "
                    "Кропы уже горизонтальные, поворачивать их не нужно."
                ),
            )
        )
        lengths = [len(item.text) for item in ocr]
        plots.append(
            _hist(
                [float(value) for value in lengths],
                "Длина строки номера",
                "Символы",
                report / "eda_ocr_length.png",
                "У обычного легкового номера 8 символов, у трёхзначного региона — 9.",
            )
        )
        symbols: Counter[str] = Counter()
        regions: Counter[str] = Counter()
        for item in ocr:
            if not item.valid_format:
                continue
            symbols.update(item.text)
            code = region_code(item.text)
            if code:
                regions[code] += 1
        if symbols:
            labels, counts = zip(*sorted(symbols.items()))
            plots.append(
                _bars(
                    list(labels),
                    list(counts),
                    "Частоты символов на номерах",
                    "Символ",
                    report / "eda_symbols.png",
                    "Алфавит датасета: цифры и латинские буквы ABEKMHOPCTYX.",
                )
            )
        if regions:
            top = regions.most_common(15)
            plots.append(
                _bars(
                    [code for code, _ in top],
                    [count for _, count in top],
                    "15 самых частых кодов региона",
                    "Регион",
                    report / "eda_regions.png",
                    "Код региона — последние 2 или 3 цифры номера. На графике только 15 лидеров.",
                )
            )

    if det_quality.get("brightness"):
        plots.append(
            _hist(
                det_quality["brightness"],
                "Яркость кадров детекции",
                "Средняя яркость, 0–255",
                report / "eda_detection_brightness.png",
                f"Оценка по случайным {det_quality['n']} кадрам. Медиана яркости { _fmt(det_quality['brightness_median'], 1) }.",
            )
        )
        plots.append(
            _hist(
                det_quality["blur"],
                "Резкость кадров детекции",
                "Дисперсия Лапласиана",
                report / "eda_detection_blur.png",
                "Меньшее значение — более размытый кадр. Это оценка на выборке, не на каждом файле.",
            )
        )
    if ocr_quality.get("brightness"):
        plots.append(
            _hist(
                ocr_quality["brightness"],
                "Яркость кропов OCR",
                "Средняя яркость, 0–255",
                report / "eda_ocr_brightness.png",
                f"Оценка по случайным {ocr_quality['n']} кропам. Медиана яркости { _fmt(ocr_quality['brightness_median'], 1) }.",
            )
        )
    return plots


def _fmt(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "—"
    if digits == 0:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def place_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_detection(root: Path, items: list[DetItem], quick: bool) -> Path:
    dest_root = root / "data" / "processed" / "detection"
    if dest_root.exists():
        shutil.rmtree(dest_root)
    used: dict[str, set[str]] = defaultdict(set)
    for item in items:
        images_dir = dest_root / item.split / "images"
        labels_dir = dest_root / item.split / "labels"
        stem = item.image.stem
        suffix = item.image.suffix.lower()
        name = stem
        index = 2
        while name in used[item.split]:
            name = f"{stem}_{index}"
            index += 1
        used[item.split].add(name)
        place_file(item.image, images_dir / f"{name}{suffix}")
        place_file(item.label, labels_dir / f"{name}.txt")

    yaml_path = root / "configs" / "plates.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Создано src/prepare_data.py. path абсолютный, обучение можно запускать из любого каталога.",
    ]
    if quick:
        lines.append("# Собрана только выборка test. Для обучения запустите prepare_data.py без --quick.")
    lines.extend(
        [
            f"path: {dest_root.resolve().as_posix()}",
            "train: train/images",
            "val: val/images",
            "test: test/images",
            "nc: 1",
            "names:",
            "  0: license_plate",
            "",
        ]
    )
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def export_ocr(root: Path, items: list[OcrItem]) -> Path:
    dest = root / "data" / "processed" / "ocr"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "image", "text", "width", "height", "valid_format"]

    def write(path: Path, rows: list[OcrItem]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for item in rows:
                writer.writerow(
                    {
                        "split": item.split,
                        "image": str(item.image.resolve()),
                        "text": item.text,
                        "width": item.width,
                        "height": item.height,
                        "valid_format": int(item.valid_format),
                    }
                )

    write(dest / "all.csv", items)
    for split in ("train", "val", "test"):
        write(dest / f"{split}.csv", [item for item in items if item.split == split])
    return dest / "all.csv"


def split_table(items: list, quick: bool) -> list[dict[str, int | str]]:
    counts = Counter(item.split for item in items)
    names = ["test"] if quick else ["train", "val", "test"]
    return [{"split": name, "files": counts.get(name, 0)} for name in names]


def build_conclusions(
    detection: list[DetItem],
    ocr: list[OcrItem],
    det_stats: dict[str, int],
    ocr_stats: dict[str, int],
    det_dupes: int,
    ocr_dupes: int,
    ocr_moved: int,
    quick: bool,
) -> list[str]:
    lines = []
    if quick:
        lines.append(
            "Это быстрый прогон только по test.zip. Для обучения частей 2 и 3 запустите без флага --quick."
        )
    lines.append(
        f"Детекция: в выборку попало {len(detection)} кадров. "
        f"Битых файлов: {det_stats.get('broken', 0)}, без метки: {det_stats.get('no_label', 0)}, "
        f"с пустой меткой: {det_stats.get('bad_label', 0)}, точных копий убрано: {det_dupes}."
    )
    if detection:
        lines.append(
            f"Медианный кадр детекции { _fmt(_median([item.width for item in detection])) }"
            f"×{ _fmt(_median([item.height for item in detection])) } px. "
            f"Номер занимает около { _fmt(_median([item.plate_area * 100 for item in detection]), 2) }% площади."
        )
        one_box = sum(item.boxes == 1 for item in detection)
        lines.append(f"Один номер на кадре у {one_box} из {len(detection)} размеченных изображений.")
    valid = sum(item.valid_format for item in ocr)
    lines.append(
        f"OCR: в выборку попало {len(ocr)} кропов, формат номера совпал у {valid}. "
        f"Битых файлов: {ocr_stats.get('broken', 0)}, точных копий убрано: {ocr_dupes}."
    )
    if ocr_moved == 0 and quick:
        lines.append("Быстрый прогон содержит только test, пересечение train/val/test на нём не проверяется.")
    elif ocr_moved == 0:
        lines.append("Одинаковые номера не пересекают train, val и test. Официальное разбиение датасета сохранено.")
    else:
        lines.append(
            f"Одинаковый номер встречался в разных выборках. Все его кадры собраны в одну выборку, перенесено файлов: {ocr_moved}."
        )
    lines.append(
        "Следующему за детекцию: открыть configs/plates.yaml и учить YOLO. Горизонтально номера не отражать."
    )
    lines.append(
        "Следующему за OCR: открыть data/processed/ocr/train.csv. Колонка text — номер, колонка image — путь к кропу."
    )
    return lines


def summary_path(root: Path) -> Path:
    return root / "reports" / "part1_summary.json"


def load_ready(root: Path, quick: bool, force: bool) -> dict | None:
    path = summary_path(root)
    if force or not path.exists():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if bool(saved.get("quick")) != quick:
        return None
    outputs = saved.get("outputs", {})
    needed = [outputs.get("detection_yaml"), outputs.get("ocr_csv")]
    plots = [item.get("path") for item in saved.get("plots", [])]
    if not all(needed) or not all(Path(item).exists() for item in [*needed, *plots]):
        return None
    say("Часть 1 уже выполнена, беру готовый отчёт. Пересобрать: python src/prepare_data.py --force")
    return saved


def prepare(root: Path | None = None, quick: bool = False, force: bool = False) -> dict:
    root = (root or repo_root_from_here()).resolve()
    ready = load_ready(root, quick, force)
    if ready is not None:
        return ready

    say("Нужно около 15 ГБ свободного места на полном прогоне. Архивы скачиваются один раз.")
    download_archives(root, quick)
    extract_archives(root, quick)

    detection, det_stats = collect_detection(root, quick)
    ocr, ocr_stats = collect_ocr(root, quick)
    if not detection or not ocr:
        raise RuntimeError(
            "После распаковки не нашлись и кадры детекции, и кропы OCR. "
            f"Детекция: {len(detection)}, OCR: {len(ocr)}."
        )

    attach_hashes(detection, "детекция")
    attach_hashes(ocr, "ocr")
    detection, det_dupes = dedupe_exact(detection, "Детекция")
    ocr, ocr_dupes = dedupe_exact(ocr, "OCR")
    ocr, ocr_moved = regroup_ocr(ocr)

    quality_limit = QUALITY_SAMPLE_QUICK if quick else QUALITY_SAMPLE_FULL
    say("Считаю яркость и размытие на случайной выборке.")
    det_quality = sample_quality(detection, quality_limit)
    ocr_quality = sample_quality(ocr, quality_limit)
    plots = build_plots(root, detection, ocr, det_quality, ocr_quality)

    yaml_path = export_detection(root, detection, quick)
    ocr_csv = export_ocr(root, ocr)
    conclusions = build_conclusions(
        detection, ocr, det_stats, ocr_stats, det_dupes, ocr_dupes, ocr_moved, quick
    )

    def quality_public(source: dict) -> dict:
        return {
            "n": source.get("n", 0),
            "brightness_median": source.get("brightness_median"),
            "blur_median": source.get("blur_median"),
        }

    summary = {
        "quick": quick,
        "detection": {
            "exported": len(detection),
            "broken": det_stats.get("broken", 0),
            "no_label": det_stats.get("no_label", 0),
            "bad_label": det_stats.get("bad_label", 0),
            "duplicate_files_removed": det_dupes,
            "median_width": _median([item.width for item in detection]),
            "median_height": _median([item.height for item in detection]),
            "median_plate_area_percent": _median([item.plate_area * 100 for item in detection]),
            "quality_sample": quality_public(det_quality),
        },
        "ocr": {
            "exported": len(ocr),
            "broken": ocr_stats.get("broken", 0),
            "invalid_name": ocr_stats.get("invalid_name", 0),
            "duplicate_files_removed": ocr_dupes,
            "files_moved_to_keep_plate_together": ocr_moved,
            "valid_format": sum(item.valid_format for item in ocr),
            "median_width": _median([item.width for item in ocr]),
            "median_height": _median([item.height for item in ocr]),
            "quality_sample": quality_public(ocr_quality),
        },
        "tables": {
            "detection_splits": split_table(detection, quick),
            "ocr_splits": split_table(ocr, quick),
        },
        "plots": plots,
        "conclusions": conclusions,
        "outputs": {
            "detection_yaml": str(yaml_path.resolve()),
            "detection_root": str((root / "data" / "processed" / "detection").resolve()),
            "ocr_csv": str(ocr_csv.resolve()),
            "ocr_train_csv": str((root / "data" / "processed" / "ocr" / "train.csv").resolve()),
            "ocr_val_csv": str((root / "data" / "processed" / "ocr" / "val.csv").resolve()),
            "ocr_test_csv": str((root / "data" / "processed" / "ocr" / "test.csv").resolve()),
        },
    }
    path = summary_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    say("")
    for line in conclusions:
        say(f"— {line}")
    say("")
    say(f"Отчёт: {path}")
    return summary


def self_check() -> None:
    assert valid_plate("X753AT174")
    assert region_code("X753AT174") == "174"
    assert valid_plate("Y004KA52")
    assert region_code("а123вс77") == "77"
    assert not valid_plate("123ABC77")
    assert choose_split(["train", "test", "test"]) == "test"
    assert choose_split(["train", "val"]) == "train"

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as folder:
        label = Path(folder) / "plate.txt"
        label.write_text("0 0.437500 0.671731 0.104452 0.032228\n", encoding="utf-8")
        parsed = parse_yolo(label)
        assert parsed is not None
        boxes, area = parsed
        assert boxes == 1
        assert 0.003 < area < 0.004

    items = [
        OcrItem("train", "train", Path("a.png"), "A123BC77", 10, 10, True),
        OcrItem("test", "test", Path("b.png"), "A123BC77", 10, 10, True),
        OcrItem("test", "test", Path("c.png"), "A123BC77", 10, 10, True),
    ]
    regroup_ocr(items)
    assert all(item.split == "test" for item in items)
    say("self-check: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Часть 1 SmartParking: данные и EDA")
    parser.add_argument("--quick", action="store_true", help="Только test.zip, чтобы проверить запуск")
    parser.add_argument("--force", action="store_true", help="Пересобрать выборки и графики")
    parser.add_argument("--self-check", action="store_true", help="Проверить функции без скачивания")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    prepare(args.root, quick=args.quick, force=args.force)


if __name__ == "__main__":
    main()
