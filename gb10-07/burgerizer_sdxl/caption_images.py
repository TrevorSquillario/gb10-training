#!/usr/bin/env python3
"""Rename images in each subdirectory to <subdir>_01, <subdir>_02, ... and
create new_filename.txt containing the subdirectory name.

Usage: python rename_images_by_subdir.py --input_dir /path/to/folder
"""
import argparse
import os
from pathlib import Path


def is_image(path: Path, exts):
    return path.suffix.lower() in exts and path.is_file()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing subdirectories")
    parser.add_argument("--dry_run", action="store_true", help="Print actions without renaming")
    parser.add_argument("--exts", default=".jpg,.jpeg,.png,.gif,.bmp,.tiff,.webp",
                        help="Comma-separated image extensions to include (default common types)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_dir}")

    exts = {e.strip().lower() if e.strip().startswith('.') else '.' + e.strip().lower()
            for e in args.exts.split(',') if e.strip()}

    # Iterate only immediate subdirectories
    for sub in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        subname = sub.name
        files = sorted([p for p in sub.iterdir() if is_image(p, exts)])
        if not files:
            print(f"Skipping '{subname}': no image files found")
            continue

        print(f"Processing '{subname}' - {len(files)} image(s)")

        # For each image we'll create a matching .txt file (basename .txt)

        # Generate new names and perform renames
        counter = 1
        for orig in files:
            ext = orig.suffix.lower()
            target_name = f"{subname}_{counter:02d}{ext}"
            target = sub / target_name

            # Avoid clobbering existing different files: if target exists, append a small suffix
            if target.exists() and not target.samefile(orig):
                suffix_idx = 1
                while True:
                    alt_target = sub / f"{subname}_{counter:02d}_v{suffix_idx}{ext}"
                    if not alt_target.exists():
                        target = alt_target
                        break
                    suffix_idx += 1

            if args.dry_run:
                print(f"  Would rename: {orig} -> {target}")
                print(f"  Would write file: {target.with_suffix('.txt')} (contents: {subname})")
            else:
                os.rename(orig, target)
                print(f"  Renamed: {orig} -> {target}")

                # create a .txt file with the same base name as the new image file
                txt_path = target.with_suffix('.txt')
                txt_path.write_text(subname + "\n", encoding="utf-8")
                print(f"  Wrote: {txt_path}")

            counter += 1


if __name__ == '__main__':
    main()
