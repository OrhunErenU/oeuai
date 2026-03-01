#!/usr/bin/env python3
"""Tank Vision AI — v3 YOLOv11m Training (r5 - from r4 best, workers=4)"""

if __name__ == "__main__":
    import cv2
    _native_imshow = cv2.imshow
    from ultralytics import YOLO
    import torch
    cv2.imshow = _native_imshow

    print("=" * 60)
    print("  TANK VISION AI — v3m R5 (from r4 best, workers=4)")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu} ({vram:.1f} GB VRAM)")

    # Start from r4 best weights (mAP50: 0.753, epoch 30)
    model = YOLO("C:/tv_data/v3/runs/tank_vision_v3m_r4/weights/best.pt")

    results = model.train(
        data="C:/tv_data/v3/data.yaml",
        epochs=80,
        batch=16,
        imgsz=640,
        device=0,
        workers=4,
        optimizer="SGD",
        lr0=0.0015,       # Slightly lower since already well-tuned
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,
        patience=30,
        amp=True,
        project="C:/tv_data/v3/runs",
        name="tank_vision_v3m_r5",
        exist_ok=True,
        save_period=10,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        scale=0.5,
        erasing=0.3,
        close_mosaic=15,
    )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
