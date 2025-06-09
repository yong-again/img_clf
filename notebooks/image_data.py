import pandas as pd
import cv2
import os
from PIL import Image
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import sys

sys.path.append('../')
from utils.util import get_parent_path


def cv2_imread_unicode(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return image
    except Exception:
        return None

def check_image_validity(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception:
        return False


def get_image_stats(image_path):
    image = cv2_imread_unicode(str(image_path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Brightness and exposure
    mean_brightness = np.mean(gray)
    exposure = np.std(gray)

    # Resolution
    height, width = gray.shape
    resolution = f"{width}x{height}"
    

    # Aspect ratio
    aspect_ratio = round(width / height, 3)

    # Blur score (variance of Laplacian)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return mean_brightness, exposure, height, width, aspect_ratio, blur_score


def process_single_folder(folder_path):
    folder_path = Path(folder_path)
    valid_images = []
    invalid_images = []
    stats_data = []

    for file in folder_path.rglob('*'):
        if file.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'):
            if check_image_validity(file):
                valid_images.append(file)
                stats = get_image_stats(file)
                if stats is not None:
                    mean_brightness, exposure, height, width, aspect_ratio, blur_score = stats
                    stats_data.append((
                        file.name,
                        mean_brightness,
                        exposure,
                        height,
                        width,
                        aspect_ratio,
                        blur_score
                    ))
            else:
                invalid_images.append(file)

    return valid_images, invalid_images, stats_data


def process_images_in_directory_parallel(directory, num_workers=4):
    subdirs = [d for d in Path(directory).iterdir() if d.is_dir()]
    with Pool(num_workers) as pool:
        results = pool.map(process_single_folder, subdirs)

    valid_images, invalid_images, stats_data = [], [], []
    for valid, invalid, stats in results:
        valid_images.extend(valid)
        invalid_images.extend(invalid)
        stats_data.extend(stats)

    return valid_images, invalid_images, stats_data


if __name__ == '__main__':
    data_dir = get_parent_path() / "data"
    image_dir = data_dir / "train"

    if not image_dir.exists():
        print(f"Image directory {image_dir} does not exist.")
        exit(1)

    valid_images, invalid_images, stats_data = process_images_in_directory_parallel(image_dir, num_workers=4)

    print(f"Valid images: {len(valid_images)}")
    print(f"Invalid images: {len(invalid_images)}")

    df = pd.DataFrame(stats_data, columns=[
        'image_name',
        'mean_brightness',
        'exposure',
        'height',
        'width',
        'aspect_ratio',
        'blur_score'
    ])
    df.sort_values(by='image_name', inplace=True)
    df.to_csv(data_dir / 'image_stats.csv', index=False, encoding='utf-8-sig')
    print("Image statistics saved to image_stats.csv")
