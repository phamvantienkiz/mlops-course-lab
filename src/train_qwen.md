```Python
"""
Pepsi Drink Detection - YOLOv11n Training with MLflow Integration
This script focuses on proper MLflow integration for YOLOv11n model training.
The implementation is streamlined for the current learning stage while maintaining
best practices for MLOps with object detection models.

Author: [Your Name]
"""

import os
import sys
from pathlib import Path
import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings
from datetime import datetime

def setup_mlflow():
    """
    Configure MLflow to use the correct tracking URI based on project structure.
    This matches your directory structure: mlflow/mlruns/
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent

    # Set up MLflow tracking in the mlflow/mlruns directory
    mlflow_dir = project_root / "mlflow"
    mlruns_dir = mlflow_dir / "mlruns"

    # Create directories if they don't exist
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    print(f"✅ MLflow tracking URI set to: file://{mlruns_dir}")

    # Set experiment name - use a consistent name for all runs
    experiment_name = "pepsi-drink-detection"
    mlflow.set_experiment(experiment_name)
    print(f"✅ Using MLflow experiment: {experiment_name}")

    return mlruns_dir

def validate_dataset():
    """
    Check if the dataset configuration file exists before training.
    """
    data_path = Path("data/data.yaml")
    if not data_path.exists():
        print(f"❌ ERROR: Dataset configuration not found at {data_path}")
        print("Please ensure your data.yaml file is in the data/ directory")
        sys.exit(1)
    print(f"✅ Dataset configuration found: {data_path}")
    return str(data_path)

def main():
    print("\n" + "="*60)
    print("Pepsi Drink Detection - YOLOv11n Training with MLflow")
    print("="*60)

    # 1. Setup MLflow with correct directory structure
    mlruns_dir = setup_mlflow()

    # 2. Enable MLflow integration in Ultralytics (CRITICAL STEP)
    settings.update({"mlflow": True})
    print("✅ Ultralytics MLflow integration enabled")

    # 3. Validate dataset exists
    data_path = validate_dataset()

    # 4. Define training configuration
    MODEL_NAME = "yolov11n.pt"  # Note: No trailing space!
    EPOCHS = 100
    IMGSZ = 640
    BATCH_SIZE = 8
    DEVICE = "auto"  # Use "auto" for auto-detection of GPU/CPU

    # 5. Create a unique run name for better tracking
    run_name = f"yolo11n-pepsi-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 6. Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n🚀 Starting training run: {run_name} (ID: {run_id})")

        # 7. Log training parameters (CRITICAL for reproducibility)
        print("\n📝 Logging training parameters...")
        training_params = {
            "model_architecture": "YOLOv11n",
            "base_model": MODEL_NAME,
            "epochs": EPOCHS,
            "image_size": IMGSZ,
            "batch_size": BATCH_SIZE,
            "device": DEVICE,
            "optimizer": "SGD",  # Default YOLO optimizer
            "data_path": data_path
        }
        mlflow.log_params(training_params)

        # 8. Train the model with auto-logging enabled
        print("\n🏋️ Starting model training...")
        print("⏳ Ultralytics will automatically log: metrics, artifacts, and parameters")

        model = YOLO(MODEL_NAME)
        results = model.train(
            data=data_path,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device=DEVICE,
            name=run_name,  # Use run_name for output directory
            exist_ok=True,
            verbose=True
        )

        # 9. Log final metrics explicitly (for Airflow to evaluate later)
        print("\n📊 Logging final metrics...")
        final_metrics = {
            "final_mAP50": float(results.metrics.map50),
            "final_mAP50-95": float(results.metrics.map50_95),
            "final_precision": float(results.metrics.precision),
            "final_recall": float(results.metrics.recall),
            "final_box_loss": float(results.metrics.box_loss),
            "final_cls_loss": float(results.metrics.cls_loss)
        }
        mlflow.log_metrics(final_metrics)

        # 10. Log important artifacts (CRITICAL for model evaluation)
        print("\n📦 Logging important artifacts...")

        # Best model weights - for model serving
        if hasattr(results, 'save_dir') and results.save_dir:
            save_dir = Path(results.save_dir)
            best_model_path = save_dir / "weights" / "best.pt"

            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), "model_weights")
                print(f"   ✅ Logged best model weights: {best_model_path}")

            # Confusion matrix - for evaluation
            confusion_matrix = save_dir / "confusion_matrix.png"
            if confusion_matrix.exists():
                mlflow.log_artifact(str(confusion_matrix), "evaluation")
                print("   ✅ Logged confusion matrix")

            # Results plot - for visualizing training progress
            results_plot = save_dir / "results.png"
            if results_plot.exists():
                mlflow.log_artifact(str(results_plot), "evaluation")
                print("   ✅ Logged results plot")

        # 11. Register the best model to MLflow Model Registry
        print("\n📝 Registering model to MLflow Model Registry...")

        # First check if we have the best model
        if best_model_path.exists():
            # Log model info for reference
            model_info = {
                "model_architecture": "YOLOv11n",
                "input_size": IMGSZ,
                "epochs": EPOCHS,
                "mAP50": final_metrics["final_mAP50"],
                "mAP50-95": final_metrics["final_mAP50-95"]
            }
            mlflow.log_dict(model_info, "model_info.json")

            # Register the best model
            model_uri = f"runs:/{run_id}/model_weights/best.pt"
            registered_model_name = "PepsiDrinkDetector-YOLOv11n"

            try:
                model_details = mlflow.register_model(
                    model_uri=model_uri,
                    name=registered_model_name,
                    tags={
                        "task": "object-detection",
                        "domain": "beverage",
                        "mAP50": f"{final_metrics['final_mAP50']:.4f}"
                    }
                )
                print(f"   ✅ Model registered as '{registered_model_name}'")
                print(f"   📦 Version: {model_details.version}")
            except Exception as e:
                print(f"   ⚠️ Model registration failed: {str(e)}")
                print("   (This might be because the model is already registered)")
        else:
            print("   ❌ Could not find best model for registration")

        # 12. Final summary
        print("\n" + "="*60)
        print(f"✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"📊 Run ID: {run_id}")
        print(f"📈 View in MLflow UI: mlflow ui --backend-store-uri {mlruns_dir}")
        print(f"🔗 Model Registry: {registered_model_name}")
        print("="*60)

if __name__ == "__main__":
    main()
```
