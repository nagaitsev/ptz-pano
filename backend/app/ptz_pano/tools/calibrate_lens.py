import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def find_chessboard_corners(gray, pattern_size):
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            return True, corners, "findChessboardCornersSB"

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners2, "findChessboardCorners"


def calibrate_directory(dir_path: Path, pattern_size=(9, 6)):
    """
    Калибровка по фотографиям в одной папке (для одного уровня зума).
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3d точки в реальном мире
    imgpoints = []  # 2d точки на изображении

    images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
    if not images:
        print(f"No images found in {dir_path}")
        return None

    img_size = None
    count = 0

    for fname in images:
        img = cv2.imread(str(fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        ret, corners, detector = find_chessboard_corners(gray, pattern_size)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            count += 1
            print(f"[{dir_path.name}] Found corners in {fname.name} with {detector}")
        else:
            print(f"[{dir_path.name}] Corners NOT found in {fname.name}")

    if count < 5:
        print(f"[{dir_path.name}] Not enough images with corners found ({count})")
        return None

    # Калибровка
    rms_error, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_size,
        None,
        None,
    )

    if rms_error:
        return {
            "matrix": mtx.tolist(),
            "dist": dist.tolist()[0],
            "resolution": img_size,
            "images_total": len(images),
            "samples_used": count,
            "rms_error": float(rms_error),
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Calibrate PTZ lens distortion at different zoom levels")
    parser.add_argument("--data", type=str, required=True, help="Path to calibration images (subfolders by zoom)")
    parser.add_argument("--out", type=str, default="config/lens_calibration.json", help="Output JSON path")
    parser.add_argument("--pattern", type=str, default="9x6", help="Chessboard size (e.g. 9x6)")
    args = parser.parse_args()

    data_root = Path(args.data)
    pattern_size = tuple(map(int, args.pattern.split("x")))
    
    results = {}
    
    # Ищем подпапки (названия должны быть числами - значениями зума)
    for subfolder in data_root.iterdir():
        if not subfolder.is_dir():
            continue
        
        try:
            zoom_val = int(subfolder.name)
        except ValueError:
            print(f"Skipping non-numeric folder: {subfolder.name}")
            continue
            
        print(f"Processing zoom {zoom_val}...")
        res = calibrate_directory(subfolder, pattern_size)
        if res:
            results[zoom_val] = res

    if not results:
        print("Calibration failed: no valid data found.")
        return

    # Сохраняем результат
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_data = {
        "pattern_size": pattern_size,
        "zooms": results
    }
    
    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Successfully saved lens calibration to {output_path}")


if __name__ == "__main__":
    main()
