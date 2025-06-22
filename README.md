The purpose of this project is to showcase the MLOPS and deployment capabilities.

The dataset is a minimal multiclass classification dataset.

# 🐧 Penguin Species Classifier API

A production-ready **FastAPI** inference service for classifying penguin species.  
This ML model is containerized with **Docker**, pushed to **Amazon ECR**, and deployed on **AWS EC2** for public access.

---

## Project Overview

This project showcases a complete ML deployment pipeline:

- ✅ Trained an XGBoost model using `scikit-learn` preprocessing and METAFLOW 
- ✅ Built a `FastAPI` inference service
- ✅ Containerized the app with `Docker`
- ✅ Pushed the image to `Amazon ECR`
- ✅ Deployed it on a public `t2.micro EC2` instance (AWS Free Tier)
- ✅ Made live predictions using `curl` from any machine

---

## 📦 Tech Stack

| Category        | Tools / Services                     |
|----------------|---------------------------------------|
| Language        | Python 3.12   
| MLOPS Workhorse | Metaflow
| ML Libraries    | pandas, numpy, scikit-learn, xgboost |
| API Framework   | FastAPI + Uvicorn                    |
| Serialization   | joblib                               |
| Deployment      | Docker, AWS EC2, AWS ECR             |
| Tracking        | MLflow (used during training)        |


---

## 🧠 Model Details

The model predicts penguin species from the following features:

- `island`
- `culmen_length_mm`
- `culmen_depth_mm`
- `flipper_length_mm`
- `body_mass_g`
- `sex`

---


---

## 🔧 How to Deploy (Short Version)

```bash
# 1. Build Docker image for EC2 platform
docker buildx build --platform linux/amd64 -t penguin-api:slim . --load

# 2. Tag and push to Amazon ECR
docker tag penguin-api:slim <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/penguin-api:slim
docker push <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/penguin-api:slim

# 3. SSH into EC2 instance and pull + run
docker pull <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/penguin-api:slim
docker run -d -p 8001:7000 <image_url>

## 🌐 API Endpoint
The endpoint is running in AWS t2.micro instance as it's the most affordable and also enough for the project and dataset.
the API can be accessed at below address.

http://13.201.166.113:8001/

reference screenshot:

<img width="1013" alt="Screenshot 2025-06-22 at 9 00 28 AM" src="https://github.com/user-attachments/assets/21646125-eb6b-4333-8095-37b836ec5fa2" />


## 📮 Sample curl Request

curl -X POST http://13.201.166.113:8001/predict \
> -H "Content-Type: application/json" \
> -d '{
>   "island": "Torgersen",
>   "culmen_length_mm": 39.1,
>   "culmen_depth_mm": 18.7,
>   "flipper_length_mm": 181.0,
>   "body_mass_g": 3750.0,
>   "sex": "Male"
> }'

output: 
{"prediction":"Adelie"}

