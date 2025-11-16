from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8s.pt")   
    
    results = model.train(
    data=r"yolo_data_split_v3\yolo_data_split\dataset.yaml",
    epochs=180,
    imgsz=640,
    batch=8,
    device=0,

    # ===== OPTIMIZED AUGMENTATION =====
    fliplr=0.0,
    flipud=0.0,
    degrees=4.0,
    translate=0.08,
    scale=0.08,

    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.35,

    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.25,
    erasing=0.03,

    # ===== TRAINING =====
    optimizer="AdamW",
    lr0=0.01,
    lrf=0.0005,
    warmup_epochs=3,

    patience=25,
    close_mosaic=10,

    project="traffic_signs_vietnam",
    name="yolov8s_optimal_v1",
    exist_ok=True,

    val=True,
    verbose=True,
)
    
    print(f"\n✅ Training completed!")
    print(f"📁 Results: {results.save_dir}")
    
    best_model = YOLO(f"{results.save_dir}/weights/best.pt")
    metrics = best_model.val()
    print(f"\n📊 Best mAP50: {metrics.box.map50:.4f}")
    print(f"📊 Best mAP50-95: {metrics.box.map:.4f}")
