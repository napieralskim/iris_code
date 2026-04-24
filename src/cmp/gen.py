#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import matplotlib.image as mpimg
from pathlib import Path
from config import Config # config.py is at ../config.py

from eye.dto import EyeEncoded, ImgMode
from eye.init import eye_main


def process_dataset(config_path: str):
    with open(config_path, "r") as f:
        config_raw = yaml.safe_load(f)
    config = Config.model_validate(config_raw)

    img_paths = [Path(p) for p in sys.argv[1:]]
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    print(f"Znaleziono {len(img_paths)} obrazów. Rozpoczynam przetwarzanie...")

    processed_count = 0
    error_count = 0

    for img_path in img_paths:
        try:
            img_raw = mpimg.imread(str(img_path))
            res = eye_main(img_raw, config.eye, ImgMode.ENCODED)
            if not isinstance(res, EyeEncoded):
                print(f"Błąd przy pliku {img_path}: not `EyeEncoded`")
                error_count += 1
                continue

            output_file = output_dir / f"{img_path.stem}.txt"
            
            str_mat = res.iris_code.astype(int).astype(str)
            str_rows = ["".join(row) for row in str_mat]
            str_whole = "\n".join(str_rows) + "\n"

            with open(output_file, "w") as f:
                f.write(str_whole)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Przetworzono: {processed_count}/{len(img_paths)}")

        except Exception as e:
            print(f"Błąd przy pliku {img_path}: {e}")
            error_count += 1

    print(f"\nZakończono!")
    print(f"Sukces: {processed_count}")
    print(f"Błędy: {error_count}")

if __name__ == "__main__":
    process_dataset("config.yaml")
