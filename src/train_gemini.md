```Python
import os
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from datetime import datetime

# --- 1. CẤU HÌNH (QUAN TRỌNG CHO AIRFLOW) ---
# Airflow sẽ dùng các biến này để chạy pipeline
MODEL_NAME = "yolov11n.pt"
DATA_PATH = "data/data.yaml"
EPOCHS = 100
IMGSZ = 640
BATCH_SIZE = 8
PROJECT_NAME = "YOLOv11_PEPSI_DETECTION" # Dùng cho cả Experiment và Model Registry

# --- 2. THIẾT LẬP MLFLOW ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(PROJECT_NAME)

print(f"Đã kết nối MLflow: {MLFLOW_TRACKING_URI}")
print(f"Sử dụng Experiment: {PROJECT_NAME}")

# --- 3. BẮT ĐẦU RUN ---
# Tạo một tên Run duy nhất để dễ dàng tracking
run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id
    print(f"Bắt đầu Run: {run_name} (ID: {run_id})")

    # --- 4. LOG THAM SỐ (QUAN TRỌNG CHO TÁI TẠO) ---
    # Mentor: Dù autologging có thể log, việc log thủ công
    # các tham số chính giúp code rõ ràng và đảm bảo chúng được log.
    print("Đang log các tham số (Parameters)...")
    mlflow.log_params({
        "model_architecture": MODEL_NAME,
        "epochs": EPOCHS,
        "image_size": IMGSZ,
        "batch_size": BATCH_SIZE,
        "data_path": DATA_PATH,
        "device": "auto"
    })

    # --- 5. HUẤN LUYỆN (AUTOLOGGING DIỄN RA Ở ĐÂY) ---
    print("Bắt đầu quá trình huấn luyện...")
    model = YOLO(MODEL_NAME)

    # Mentor: Đây là nơi phép màu xảy ra!
    # Chỉ cần 'model.train()', Ultralytics sẽ TỰ ĐỘNG phát hiện
    # MLflow run đang chạy (run_id) và log TẤT CẢ:
    # 1. Các tham số (Parameters) khác (như optimizer, lr,...)
    # 2. Metrics CỦA TỪNG EPOCH (mAP, loss, precision, recall...)
    # 3. Artifacts (best.pt, last.pt, results.csv, confusion_matrix.png,...)
    #
    # Em không cần làm gì cả. Toàn bộ file 'results_dir'
    # trong code của giảng viên đã được log tự động.
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device="auto",
        name=run_name # Đặt tên thư mục output local trùng tên run
    )
    print("Huấn luyện hoàn tất.")

    # --- 6. LOG METRIC QUYẾT ĐỊNH (QUAN TRỌNG CHO AIRFLOW) ---
    # Mentor: Autologging đã log TẤT CẢ metrics.
    # Nhưng chúng ta nên log lại metric quan trọng nhất (ví dụ: mAP_50_95)
    # một cách tường minh.
    # Airflow DAG sẽ truy vấn metric 'final_mAP_50_95' này
    # để quyết định có 'promote' mô hình lên Production hay không.
    final_map_50_95 = results.metrics.mAP_50_95
    print(f"Metric quyết định (final_mAP_50_95): {final_map_50_95}")
    mlflow.log_metric("final_mAP_50_95", final_map_50_95)

    # --- 7. ĐĂNG KÝ MÔ HÌNH (QUAN TRỌNG CHO AIRFLOW & FASTAPI) ---
    # Mentor: Đây là bước quan trọng nhất em phải làm thủ công.
    # Autologging chỉ 'log' artifact, nó không 'register' mô hình.
    #
    # Chúng ta đăng ký mô hình 'best.pt' (không phải 'last.pt' hay model
    # trong bộ nhớ như code mẫu) vào Model Registry.
    #
    # - Airflow sẽ chuyển mô hình này từ 'None' -> 'Staging'.
    # - FastAPI sẽ load mô hình từ 'Production'.

    print(f"Đang đăng ký mô hình '{PROJECT_NAME}'...")

    # Lấy đường dẫn chính xác đến file 'best.pt'
    best_model_path = os.path.join(results.save_dir, 'weights/best.pt')

    if os.path.exists(best_model_path):
        mlflow.pytorch.log_model(
            pytorch_model=best_model_path,       # Log từ file TỐT NHẤT
            artifact_path="model",             # Tên thư mục trong artifacts
            registered_model_name=PROJECT_NAME # Tên mô hình trong Registry
        )
        print(f"Đăng ký mô hình thành công. Phiên bản mới đã được tạo.")
    else:
        print(f"LỖI: Không tìm thấy tệp 'best.pt' tại {best_model_path}")
        mlflow.end_run(status="FAILED")

print(f"\n✅ Hoàn tất run {run_name}. Kiểm tra trên MLflow UI.")
```
