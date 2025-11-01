```
PDD-MLOps/
│
├── data/ # Dữ liệu huấn luyện, có thể dùng DVC sau
│
├── models/
│ └── best.pt
│
├── src/
│ ├── train_with_mlflow.py # Training + log MLflow
│ ├── inference.py # Local test inference
│ ├── mlflow_log_from_yolo.py # Import model có sẵn
│ ├── utils/ # Hàm helper
│ │ └── data_utils.py
│ └── api/
│ └── main.py # FastAPI app
│
├── airflow/
│ ├── dags/
│ │ └── yolo_pipeline_dag.py
│ └── Dockerfile
│
├── mlflow/
│ ├── Dockerfile
│ ├── mlflow.db # (hoặc mount volume)
│ └── mlruns/
│
├── grafana/
│ ├── dashboards/
│ │ └── system_metrics.json
│ └── Dockerfile
│
├── prometheus/
│ └── prometheus.yml
│
├── tests/
│ ├── test_api.py
│ ├── test_train.py
│ └── conftest.py
│
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .github/
└── workflows/
└── ci.yml # Pytest + build check
```
