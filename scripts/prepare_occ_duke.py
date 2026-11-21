"""Prepare Occluded-Duke from image/mask archives and the official split lists."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

import numpy as np
from PIL import Image


SPLITS = {"train": "bounding_box_train", "query": "query", "gallery": "bounding_box_test"}
MASK_KIND = "pifpaf_maskrcnn_filtering"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def copy_member(archive, member, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    with archive.open(member) as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--masks-zip", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/Occluded_Duke"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    lists = {}
    for split in SPLITS:
        names = (args.split_dir / f"{split}.list").read_text().splitlines()
        names = [name.strip() for name in names if name.strip()]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Empty or duplicate entries in {split}.list")
        if any(PurePosixPath(name).name != name or not name.endswith(".jpg") for name in names):
            raise ValueError(f"Unexpected image filename in {split}.list")
        lists[split] = names

    report = {"dataset": "Occluded-Duke", "splits": {}, "archives": {},
              "split_source": "https://github.com/lightas/Occluded-DukeMTMC-Dataset",
              "mask_source": "https://github.com/VlSomers/bpbreid"}
    with ZipFile(args.images_zip) as images, ZipFile(args.masks_zip) as masks:
        image_index = {}
        for entry in images.infolist():
            parts = PurePosixPath(entry.filename).parts
            if len(parts) >= 2 and "__MACOSX" not in parts and parts[-1].endswith(".jpg"):
                key = (parts[-2], parts[-1])
                if key in image_index:
                    raise ValueError(f"Ambiguous image member: {key}")
                image_index[key] = entry
        mask_index = {}
        for entry in masks.infolist():
            parts = PurePosixPath(entry.filename).parts
            if len(parts) >= 3 and MASK_KIND in parts and "__MACOSX" not in parts:
                key = (parts[-2], parts[-1])
                if not entry.is_dir():
                    if key in mask_index:
                        raise ValueError(f"Ambiguous mask member: {key}")
                    mask_index[key] = entry

        plan = []
        missing = []
        for split, folder in SPLITS.items():
            for name in lists[split]:
                image = image_index.get((folder, name))
                if image is None and split == "query":
                    image = image_index.get(("bounding_box_test", name))
                mask_name = name + ".confidence_fields.npy"
                mask = mask_index.get((folder, mask_name))
                if image is None or mask is None:
                    missing.append({"split": split, "image": name,
                                    "missing_image": image is None, "missing_mask": mask is None})
                else:
                    plan.append((split, folder, name, image, mask))
        if missing:
            raise ValueError(f"Archive coverage failed: {len(missing)} missing entries; first: {missing[:5]}")

        print(f"Archive coverage verified: {len(plan)} image/mask pairs", flush=True)
        for index, (split, folder, name, image, mask) in enumerate(plan, 1):
            copy_member(images, image, args.output / folder / name)
            copy_member(masks, mask, args.output / "masks" / MASK_KIND / folder / (name + ".confidence_fields.npy"))
            if index % 2000 == 0:
                print(f"Extracted {index}/{len(plan)} pairs", flush=True)

    for split, folder in SPLITS.items():
        sizes = Counter()
        mask_shapes = Counter()
        for name in lists[split][:32]:
            with Image.open(args.output / folder / name) as image:
                sizes[str(image.size)] += 1
                image.verify()
            mask = np.load(args.output / "masks" / MASK_KIND / folder / (name + ".confidence_fields.npy"), allow_pickle=False)
            if mask.ndim != 3 or not np.isfinite(mask).all():
                raise ValueError(f"Invalid mask: {split}/{name}")
            mask_shapes[str(mask.shape)] += 1
        report["splits"][split] = {"images": len(lists[split]), "masks": len(lists[split]),
                                   "sample_image_sizes": dict(sizes), "sample_mask_shapes": dict(mask_shapes),
                                   "split_list_sha256": digest(args.split_dir / f"{split}.list")}
    for name, path in (("images", args.images_zip), ("masks", args.masks_zip)):
        report["archives"][name] = {"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
    report["status"] = "prepared"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
