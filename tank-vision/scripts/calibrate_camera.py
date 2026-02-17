"""Kamera kalibrasyon araci.

OpenCV checkerboard deseni ile kamera ic parametrelerini
(odak uzakligi, bozulma katsayilari) hesaplar.

Kullanim:
    python scripts/calibrate_camera.py --source 0 --output config/camera/calibrated.yaml
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def calibrate_camera(
    source,
    checkerboard_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    n_samples: int = 20,
    output_path: str = "config/camera/calibrated.yaml",
):
    """Kamera kalibrasyonu yap.

    Args:
        source: Video kaynagi (kamera indeksi veya dosya yolu).
        checkerboard_size: Checkerboard ic kose sayisi (satir, sutun).
        square_size_mm: Kare boyutu (mm).
        n_samples: Toplanacak ornek sayisi.
        output_path: Cikti YAML dosyasi.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D dunya noktalari
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0 : checkerboard_size[0], 0 : checkerboard_size[1]
    ].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points = []  # 3D noktalar
    img_points = []  # 2D noktalar

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Video kaynagi acilamadi: {source}")
        return

    img_size = None
    collected = 0

    print(f"[*] Checkerboard kalibrasyonu baslatildi ({checkerboard_size})")
    print(f"    {n_samples} ornek toplanacak. 'c' ile yakala, 'q' ile cik.")

    while collected < n_samples:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, checkerboard_size, corners, found)
            cv2.putText(
                display,
                f"Checkerboard bulundu! 'c' ile yakala ({collected}/{n_samples})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                display,
                f"Checkerboard bulunamadi ({collected}/{n_samples})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Kalibrasyon", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") and found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            obj_points.append(objp)
            img_points.append(corners_refined)
            collected += 1
            print(f"    Ornek {collected}/{n_samples} yakalandi")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected < 4:
        print("[!] Yeterli ornek toplanamadi (minimum 4)")
        return

    # Kalibrasyon hesapla
    print("[*] Kalibrasyon hesaplaniyor...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    if not ret:
        print("[!] Kalibrasyon basarisiz")
        return

    # Sonuclari cikar
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Yeniden projeksiyon hatasi
    total_error = 0
    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(
            img_points_proj
        )
        total_error += error
    mean_error = total_error / len(obj_points)

    # Sonuclari kaydet
    result = {
        "focal_length_px": float((fx + fy) / 2),
        "fx": float(fx),
        "fy": float(fy),
        "principal_point": {"x": float(cx), "y": float(cy)},
        "distortion_coeffs": dist_coeffs.flatten().tolist(),
        "resolution": {"width": img_size[0], "height": img_size[1]},
        "mean_reprojection_error": float(mean_error),
        "num_samples": collected,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False)

    print("\n" + "=" * 50)
    print("KALIBRASYON SONUCLARI")
    print("=" * 50)
    print(f"  Odak uzakligi: {result['focal_length_px']:.1f} px")
    print(f"  fx={fx:.1f}, fy={fy:.1f}")
    print(f"  Ana nokta: ({cx:.1f}, {cy:.1f})")
    print(f"  Ortalama hata: {mean_error:.4f} px")
    print(f"  Kaydedildi: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kamera kalibrasyonu")
    parser.add_argument("--source", default=0, help="Video kaynagi")
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=25.0, help="Kare boyutu (mm)")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", default="config/camera/calibrated.yaml")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    calibrate_camera(
        source=source,
        checkerboard_size=(args.rows, args.cols),
        square_size_mm=args.square_size,
        n_samples=args.samples,
        output_path=args.output,
    )
