#!/usr/bin/env python3
"""Download and extract VKITTI 1.3.1 from the official Naver Labs URLs.

GemDepth's current dataset loader expects the target directory to contain:

    <target-dir>/vkitti_1.3.1_rgb/
    <target-dir>/vkitti_1.3.1_depthgt/
    <target-dir>/vkitti_1.3.1_extrinsicsgt/

Official dataset page:
https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_TARGET_DIR = "/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1"

# Use the official Naver Labs links.  Only these three components are needed by
# dataset/dataset_mix.py for GemDepth training.
COMPONENTS = {
    "extrinsicsgt": {
        "url": "https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti%5F1.3.1%5Fextrinsicsgt.tar.gz",
        "archive": "vkitti_1.3.1_extrinsicsgt.tar.gz",
        "directory": "vkitti_1.3.1_extrinsicsgt",
        "description": "camera extrinsics / poses",
    },
    "depthgt": {
        "url": "https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti%5F1.3.1%5Fdepthgt.tar",
        "archive": "vkitti_1.3.1_depthgt.tar",
        "directory": "vkitti_1.3.1_depthgt",
        "description": "depth ground truth",
    },
    "rgb": {
        "url": "https://download.europe.naverlabs.com/virtual-kitti-1.3.1/vkitti%5F1.3.1%5Frgb.tar",
        "archive": "vkitti_1.3.1_rgb.tar",
        "directory": "vkitti_1.3.1_rgb",
        "description": "RGB frames",
    },
}


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download(url: str, output_path: Path) -> None:
    """Download with resume support using wget or curl."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("wget"):
        run([
            "wget",
            "-c",
            "--tries=10",
            "--timeout=60",
            "--progress=dot:giga",
            "-O",
            str(output_path),
            url,
        ])
        return

    if shutil.which("curl"):
        run([
            "curl",
            "-L",
            "--retry",
            "10",
            "--retry-delay",
            "10",
            "-C",
            "-",
            "-o",
            str(output_path),
            url,
        ])
        return

    raise RuntimeError("Neither wget nor curl is available on this node.")


def extract(archive_path: Path, target_dir: Path) -> None:
    if archive_path.suffixes[-2:] == [".tar", ".gz"]:
        run(["tar", "-xzf", str(archive_path), "-C", str(target_dir)])
    elif archive_path.suffix == ".tar":
        run(["tar", "-xf", str(archive_path), "-C", str(target_dir)])
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")


def verify(target_dir: Path) -> bool:
    print("\n" + "=" * 80)
    print(f"Verifying VKITTI 1.3.1 structure under: {target_dir}")
    print("=" * 80)

    ok = True
    for name, meta in COMPONENTS.items():
        data_dir = target_dir / meta["directory"]
        if not data_dir.exists():
            print(f"✗ missing {name}: {data_dir}")
            ok = False
            continue

        scene_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
        print(f"✓ {meta['directory']}: {len(scene_dirs)} top-level scene directories")

    if ok:
        print("\n✓ VKITTI 1.3.1 is ready for GemDepth training.")
    else:
        print("\n✗ VKITTI 1.3.1 is incomplete.")

    return ok


def download_and_extract_vkitti(
    target_dir: str,
    selected_components: list[str],
    keep_archives: bool,
    verify_only: bool,
) -> bool:
    target = Path(target_dir).expanduser().resolve()
    archive_dir = target / "archives"
    target.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target directory: {target}")
    print(f"Archive directory: {archive_dir}")
    print("Official VKITTI 1.3.1 source:")
    print("  https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1")

    if verify_only:
        return verify(target)

    for component in selected_components:
        if component not in COMPONENTS:
            raise ValueError(f"Unknown component '{component}'. Valid choices: {list(COMPONENTS)}")

        meta = COMPONENTS[component]
        extracted_dir = target / meta["directory"]
        archive_path = archive_dir / meta["archive"]

        print("\n" + "=" * 80)
        print(f"Component: {component} ({meta['description']})")
        print(f"URL:       {meta['url']}")
        print(f"Archive:   {archive_path}")
        print(f"Output:    {extracted_dir}")
        print("=" * 80)

        if extracted_dir.exists():
            print(f"✓ already extracted, skipping: {extracted_dir}")
            continue

        if not archive_path.exists():
            download(meta["url"], archive_path)
        else:
            print(f"✓ archive already exists, reusing: {archive_path}")

        print(f"Extracting {archive_path.name} ...")
        extract(archive_path, target)

        if not extracted_dir.exists():
            raise RuntimeError(f"Extraction finished but expected directory is missing: {extracted_dir}")

        print(f"✓ extracted: {extracted_dir}")
        if not keep_archives:
            archive_path.unlink(missing_ok=True)
            print(f"✓ removed archive to save space: {archive_path}")

    return verify(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download VKITTI 1.3.1 from official Naver Labs URLs")
    parser.add_argument(
        "--target-dir",
        default=DEFAULT_TARGET_DIR,
        help="Directory that will contain vkitti_1.3.1_rgb/depthgt/extrinsicsgt",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=list(COMPONENTS.keys()),
        choices=list(COMPONENTS.keys()),
        help="Subset of components to download. Default: all components required by GemDepth.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded .tar/.tar.gz files after successful extraction.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the extracted directory structure; do not download or extract.",
    )
    args = parser.parse_args()

    try:
        success = download_and_extract_vkitti(
            target_dir=args.target_dir,
            selected_components=args.components,
            keep_archives=args.keep_archives,
            verify_only=args.verify_only,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if success else 1)
