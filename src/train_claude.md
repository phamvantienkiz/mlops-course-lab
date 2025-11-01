```Python
"""
MLOps Training Script with MLflow Integration
==============================================
Script để huấn luyện YOLO11n model với MLflow tracking
Chuẩn bị cho integration với Airflow, FastAPI, Grafana

Author: MLOps Student
Project: Pepsi Drink Detection
"""

import os
import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings
from pathlib import Path
import yaml
import shutil
from datetime import datetime


# ============================================================================
# CONFIGURATION - ✅ QUAN TRỌNG CHO TOÀN BỘ PROJECT
# ============================================================================

class Config:
    """
    Centralized configuration class
    ✅ QUAN TRỌNG: Airflow DAG sẽ sử dụng các config này để orchestrate
    """
    # MLflow Settings - ✅ CỰC KỲ QUAN TRỌNG
    MLFLOW_TRACKING_URI = "file:///app/mlflow/mlruns"  # URI cho Docker container
    MLFLOW_EXPERIMENT_NAME = "pepsi-detection-yolo11n"  # Tên experiment

    # ✅ QUAN TRỌNG: Local development URI (khi chưa dùng Docker)
    MLFLOW_LOCAL_URI = "./mlflow/mlruns"

    # Model Settings
    MODEL_NAME = "yolov11n.pt"  # Base model
    DATASET_CONFIG = "./data/data.yaml"  # ✅ QUAN TRỌNG: Airflow sẽ validate file này

    # Training Hyperparameters - ✅ QUAN TRỌNG: Airflow sẽ log/modify các params này
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 640
    DEVICE = 0  # GPU ID, hoặc 'cpu'

    # Model Registry - ✅ CỰC KỲ QUAN TRỌNG CHO FASTAPI
    REGISTERED_MODEL_NAME = "pepsi-detector-yolo11n"  # FastAPI sẽ load model này

    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MLFLOW_DIR = PROJECT_ROOT / "mlflow"


# ============================================================================
# MLFLOW SETUP - ✅ QUAN TRỌNG
# ============================================================================

def setup_mlflow():
    """
    Cấu hình MLflow tracking URI và experiment
    ✅ QUAN TRỌNG: Airflow DAG sẽ gọi hàm này trước mỗi task
    """
    # Tạo thư mục mlflow nếu chưa có
    Config.MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    # Set tracking URI - ✅ CỰC KỲ QUAN TRỌNG
    # Trong Docker, URI sẽ là file:///app/mlflow/mlruns
    # Local development: ./mlflow/mlruns
    mlflow.set_tracking_uri(Config.MLFLOW_LOCAL_URI)

    # Set experiment - ✅ QUAN TRỌNG: Airflow sẽ track tất cả runs trong experiment này
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

    print(f"✅ MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"✅ MLflow Experiment: {Config.MLFLOW_EXPERIMENT_NAME}")


def configure_ultralytics_mlflow():
    """
    Enable MLflow integration trong Ultralytics
    ✅ QUAN TRỌNG: Ultralytics sẽ tự động log metrics, params, artifacts
    """
    # Enable MLflow trong Ultralytics settings
    settings.update({'mlflow': True})
    print("✅ Ultralytics MLflow integration enabled")


# ============================================================================
# TRAINING FUNCTION - ✅ QUAN TRỌNG
# ============================================================================

def train_model():
    """
    Main training function với MLflow tracking
    ✅ QUAN TRỌNG: Airflow sẽ gọi hàm này trong DAG task

    Returns:
        str: MLflow run_id (để Airflow track và promote model)
    """

    # Setup MLflow
    setup_mlflow()
    configure_ultralytics_mlflow()

    # ✅ QUAN TRỌNG: Start MLflow run với descriptive name
    # Airflow sẽ sử dụng run_id này để query và promote model
    run_name = f"yolo11n-pepsi-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n{'='*80}")
        print(f"🚀 Starting Training Run: {run_name}")
        print(f"📊 Run ID: {run.info.run_id}")
        print(f"{'='*80}\n")

        # ================================================================
        # LOG PARAMETERS - ✅ CỰC KỲ QUAN TRỌNG CHO AIRFLOW & GRAFANA
        # ================================================================
        # Airflow sẽ track các params này để:
        # 1. Hiển thị trong DAG logs
        # 2. So sánh giữa các runs
        # 3. Grafana sẽ visualize training configs

        training_params = {
            "model_architecture": "YOLOv11n",
            "base_model": Config.MODEL_NAME,
            "epochs": Config.EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "img_size": Config.IMG_SIZE,
            "device": Config.DEVICE,
            "optimizer": "AdamW",  # Default YOLO optimizer
            "dataset": "pepsi-detection",
        }

        mlflow.log_params(training_params)
        print("✅ Logged training parameters")

        # ================================================================
        # LOG TAGS - ✅ QUAN TRỌNG CHO AIRFLOW FILTERING
        # ================================================================
        # Tags giúp Airflow query và filter runs
        mlflow.set_tags({
            "project": "pepsi-detection",
            "model_type": "object-detection",
            "framework": "ultralytics",
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "stage": "training",  # Airflow sẽ update thành "production" sau promotion
            # ✅ BONUS: DVC dataset version (nếu implement DVC challenge)
            # "dvc_version": "v1.0.0"
        })
        print("✅ Logged tags")

        # ================================================================
        # LOAD & TRAIN MODEL - ULTRALYTICS AUTO-LOGGING
        # ================================================================
        print("\n📦 Loading YOLO11n model...")
        model = YOLO(Config.MODEL_NAME)

        print("\n🏋️ Starting training...")
        print("⏰ Ultralytics sẽ tự động log:")
        print("   - Training metrics (mAP, precision, recall, loss)")
        print("   - Validation metrics")
        print("   - Learning rate curves")
        print("   - Sample predictions")

        # Train with auto-logging
        # ✅ QUAN TRỌNG: Ultralytics sẽ tự động log vào MLflow run hiện tại
        results = model.train(
            data=Config.DATASET_CONFIG,
            epochs=Config.EPOCHS,
            batch=Config.BATCH_SIZE,
            imgsz=Config.IMG_SIZE,
            device=Config.DEVICE,
            project=str(Config.MLFLOW_DIR / "training_outputs"),  # Temporary output
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        print("\n✅ Training completed!")

        # ================================================================
        # LOG FINAL METRICS - ✅ CỰC KỲ QUAN TRỌNG CHO AIRFLOW EVALUATION
        # ================================================================
        # Airflow DAG sẽ đọc metrics này để quyết định promote model hay không
        # Tiêu chí: nếu mAP50 > threshold → promote to Production

        try:
            # Lấy metrics từ results
            final_metrics = {
                # ✅ CỰC KỲ QUAN TRỌNG: Airflow sẽ dùng mAP50 để auto-promote
                "final_mAP50": float(results.results_dict.get('metrics/mAP50(B)', 0)),
                "final_mAP50-95": float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                "final_precision": float(results.results_dict.get('metrics/precision(B)', 0)),
                "final_recall": float(results.results_dict.get('metrics/recall(B)', 0)),
                "final_train_loss": float(results.results_dict.get('train/box_loss', 0)),
            }

            mlflow.log_metrics(final_metrics)
            print(f"\n📊 Final Metrics:")
            for key, value in final_metrics.items():
                print(f"   {key}: {value:.4f}")

        except Exception as e:
            print(f"⚠️ Warning: Could not extract all metrics: {e}")

        # ================================================================
        # LOG ARTIFACTS - ✅ CỰC KỲ QUAN TRỌNG CHO FASTAPI & MONITORING
        # ================================================================
        # FastAPI cần: best.pt (model weights)
        # Grafana/Monitoring cần: confusion matrix, results plots

        training_output_dir = Config.MLFLOW_DIR / "training_outputs" / run_name

        print("\n📦 Logging artifacts...")

        # 1. ✅ CỰC KỲ QUAN TRỌNG: Best model weights (cho FastAPI)
        best_weights = training_output_dir / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")
            print("   ✅ Logged: best.pt (FastAPI sẽ load file này)")

        # 2. ⚠️ QUAN TRỌNG: Last weights (backup)
        last_weights = training_output_dir / "weights" / "last.pt"
        if last_weights.exists():
            mlflow.log_artifact(str(last_weights), artifact_path="weights")
            print("   ✅ Logged: last.pt")

        # 3. ✅ QUAN TRỌNG: Confusion matrix (cho Grafana monitoring)
        confusion_matrix = training_output_dir / "confusion_matrix.png"
        if confusion_matrix.exists():
            mlflow.log_artifact(str(confusion_matrix), artifact_path="evaluation")
            print("   ✅ Logged: confusion_matrix.png")

        # 4. ✅ QUAN TRỌNG: Results plots (cho monitoring dashboard)
        results_png = training_output_dir / "results.png"
        if results_png.exists():
            mlflow.log_artifact(str(results_png), artifact_path="evaluation")
            print("   ✅ Logged: results.png")

        # 5. ⚠️ QUAN TRỌNG: Training curves (để debug và analyze)
        for plot_name in ["F1_curve.png", "P_curve.png", "R_curve.png", "PR_curve.png"]:
            plot_path = training_output_dir / plot_name
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), artifact_path="curves")
        print("   ✅ Logged: training curves")

        # 6. ⚠️ QUAN TRỌNG: Training config (data.yaml) để reproduce
        if Path(Config.DATASET_CONFIG).exists():
            mlflow.log_artifact(Config.DATASET_CONFIG, artifact_path="config")
            print("   ✅ Logged: data.yaml")

        # ================================================================
        # MODEL REGISTRY - ✅ CỰC KỲ QUAN TRỌNG CHO FASTAPI
        # ================================================================
        # FastAPI sẽ load model từ Model Registry với stage="Production"
        # Airflow sẽ tự động promote model tốt nhất lên Production stage

        print("\n📝 Registering model to MLflow Model Registry...")

        # Log model với PyTorch format
        # ✅ QUAN TRỌNG: Cần log model để FastAPI có thể load
        try:
            # Tạo model info dict
            model_info = {
                "model_path": str(best_weights),
                "model_architecture": "YOLOv11n",
                "input_size": Config.IMG_SIZE,
                "num_classes": None,  # Sẽ được load từ data.yaml
            }

            # Log model artifact
            # ✅ FASTAPI SẼ LOAD TỪ ĐÂY
            mlflow.log_dict(model_info, "model_info.json")

            # Register model vào Model Registry
            # ✅ CỰC KỲ QUAN TRỌNG: Name này phải match với Config.REGISTERED_MODEL_NAME
            model_uri = f"runs:/{run.info.run_id}/weights/best.pt"

            model_details = mlflow.register_model(
                model_uri=model_uri,
                name=Config.REGISTERED_MODEL_NAME,
                tags={
                    "framework": "ultralytics-yolo",
                    "task": "object-detection",
                    "mAP50": final_metrics.get("final_mAP50", 0),
                }
            )

            print(f"   ✅ Model registered: {Config.REGISTERED_MODEL_NAME}")
            print(f"   📦 Version: {model_details.version}")
            print(f"   🔗 Model URI: {model_uri}")

            # ⚠️ LÀM THỦ CÔNG Ở ĐÂY - AIRFLOW SẼ TỰ ĐỘNG HÓA SAU
            # Trong production: Airflow DAG sẽ tự động promote nếu mAP > threshold
            print("\n⚠️ NOTE: Model đã được registered nhưng chưa promote to Production")
            print("   Để promote manually, chạy:")
            print(f"   python src/promote_model.py --version {model_details.version}")

        except Exception as e:
            print(f"⚠️ Warning: Model registration failed: {e}")
            print("   Có thể cần cài đặt: pip install mlflow[gateway]")

        # ================================================================
        # CLEANUP - ⚠️ TÙY CHỌN
        # ================================================================
        # Xóa temporary training outputs (đã log vào MLflow rồi)
        # ⚠️ Cân nhắc: Giữ lại để debug hoặc xóa để tiết kiệm disk

        # if training_output_dir.exists():
        #     shutil.rmtree(training_output_dir)
        #     print("\n🗑️ Cleaned up temporary training outputs")

        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Run ID: {run.info.run_id}")
        print(f"🔗 View in MLflow UI: {mlflow.get_tracking_uri()}")
        print(f"{'='*80}\n")

        return run.info.run_id


# ============================================================================
# UTILITY FUNCTIONS - ⚠️ QUAN TRỌNG CHO AIRFLOW
# ============================================================================

def get_best_model_version():
    """
    Lấy version của model tốt nhất từ Model Registry
    ✅ QUAN TRỌNG: Airflow sẽ dùng hàm này để auto-promote

    Returns:
        int: Model version với mAP cao nhất
    """
    client = mlflow.tracking.MlflowClient()

    try:
        # Get all versions của registered model
        versions = client.search_model_versions(
            f"name='{Config.REGISTERED_MODEL_NAME}'"
        )

        if not versions:
            print("⚠️ No model versions found")
            return None

        # Tìm version với mAP cao nhất
        best_version = None
        best_map = 0

        for version in versions:
            # Get run metrics
            run = client.get_run(version.run_id)
            map_score = run.data.metrics.get("final_mAP50", 0)

            if map_score > best_map:
                best_map = map_score
                best_version = version.version

        print(f"✅ Best model version: {best_version} (mAP50: {best_map:.4f})")
        return best_version

    except Exception as e:
        print(f"⚠️ Error finding best model: {e}")
        return None


def promote_model_to_production(version: int):
    """
    Promote model version lên Production stage
    ✅ CỰC KỲ QUAN TRỌNG: FastAPI chỉ load models từ Production stage
    ✅ QUAN TRỌNG: Airflow DAG sẽ gọi hàm này tự động

    Args:
        version: Model version number
    """
    client = mlflow.tracking.MlflowClient()

    try:
        # Transition model to Production
        client.transition_model_version_stage(
            name=Config.REGISTERED_MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True  # Archive old Production models
        )

        print(f"✅ Model version {version} promoted to Production!")
        print(f"   FastAPI sẽ tự động load version này")

    except Exception as e:
        print(f"❌ Error promoting model: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point
    ✅ QUAN TRỌNG: Airflow DAG sẽ call các functions này
    """

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         PEPSI DETECTION - YOLO11N TRAINING WITH MLFLOW       ║
    ║                    MLOps Final Project                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Validate dataset config exists
    if not Path(Config.DATASET_CONFIG).exists():
        print(f"❌ ERROR: Dataset config not found: {Config.DATASET_CONFIG}")
        print("   Please ensure data.yaml exists in data/ directory")
        exit(1)

    try:
        # Train model
        run_id = train_model()

        print("\n" + "="*80)
        print("🎯 NEXT STEPS:")
        print("="*80)
        print("\n1️⃣ View training results in MLflow UI:")
        print(f"   cd mlflow && mlflow ui --backend-store-uri file:///$(pwd)/mlruns")
        print(f"   Open: http://localhost:5000")

        print("\n2️⃣ (Optional) Manually promote model to Production:")
        print(f"   python src/promote_model.py --run-id {run_id}")

        print("\n3️⃣ Upcoming: Integrate with Airflow DAG")
        print("   ✅ Airflow sẽ tự động chạy training")
        print("   ✅ Airflow sẽ evaluate và auto-promote models")
        print("   ✅ FastAPI sẽ load model từ Production stage")

        print("\n4️⃣ Upcoming: Setup FastAPI model serving")
        print("   ✅ FastAPI sẽ load model từ MLflow Registry")
        print("   ✅ Endpoints: /predict, /health")

        print("\n5️⃣ Upcoming: Monitor với Grafana")
        print("   ✅ Visualize training metrics")
        print("   ✅ Monitor API performance")
        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
```
