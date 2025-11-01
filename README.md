# 🧠 MLOps Course Lab

> A fully containerized MLOps system developed as the **Final Project** of the MLOps Bootcamp by [Cole](https://cole.ai/)  
> Instructor: **Phạm Thanh Danh**, Senior AI/ML Engineer at [Backbase](https://www.linkedin.com/in/danh-pham-288838a6)

---

## 🚀 Project Overview

This project demonstrates a **production-grade MLOps system** for an object detection task — **Pepsi Drink Detection**, built with the following key technologies:

- **MLflow** – Experiment tracking, model versioning, and registry
- **FastAPI** – Model serving through RESTful endpoints
- **Apache Airflow** – Automated pipeline orchestration
- **Docker & Docker Compose** – Full containerization for reproducibility
- **Grafana + Prometheus** – Real-time monitoring and metrics visualization
- **Pytest + GitHub Actions** – Testing and CI/CD integration

The project follows a modular, real-world MLOps architecture and can be reproduced from scratch with a few commands once the full pipeline is complete.

---

## 🧩 System Components & Progress

| Component                                            | Description                                                                                                                                    | Status             |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **A. Experiment Tracking with MLflow**               | Integration of MLflow for logging parameters, metrics, and artifacts. Best model registered in Model Registry with version promotion.          | ✅ **Completed**   |
| **B. Model Serving with FastAPI**                    | REST API exposing `/predict` and `/health` endpoints. Validates input via Pydantic and loads the latest Production model from MLflow Registry. | 🚧 **In Progress** |
| **C. Pipeline Orchestration with Apache Airflow**    | DAG for automated data ingestion → model training → evaluation → promotion.                                                                    | ⏳ **Planned**     |
| **D. Containerization with Docker & Docker Compose** | Full system orchestration (MLflow, FastAPI, Airflow, Grafana) using Docker Compose.                                                            | ⏳ **Planned**     |
| **E. Monitoring with Grafana**                       | Dashboard to visualize metrics (CPU, memory, API latency, request count).                                                                      | ⏳ **Planned**     |
| **F. Testing with Pytest**                           | Automated tests for API and training pipeline. Integration with CI/CD workflow.                                                                | ⏳ **Planned**     |

---

## 🧾 License

This project is licensed under the **MIT License** --- see the <LICENSE> file for details.

---

## 👨‍🏫 Credits

**MLOps Bootcamp -- Final Project**\
Instructor: [Phạm Thanh Danh](https://www.linkedin.com/in/danh-pham-288838a6)\
Senior AI/ML Engineer at Backbase
