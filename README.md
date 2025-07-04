The purpose of this project is to showcase the MLOPS and cloud deployment capabilities.

The dataset is a minimal multiclass classification dataset.

# Data and process overview 

The data contains certain features of a penguine like culmen_length, culmen_width, island etc. and our model 
predict the penguine type out of Chinstrip, Bescoe, Gentoo.

During Experimentation we decided on XGBoost Classifier after comapring the result on XGBoost, LighGBM 
and a small level neural network(which was an overkill anyway.) 

# MLflow process

1. Start mlflow server at local system and set mlflow_tracking_uri.
2. Start and experiment and give it's name to mlflow.set_experiment('penguins')
3. Then we do experimentation on seevral models. Xgboost example is given below.

with mlflow.start_run(run_name = 'xgboost') as parent_run:

    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    for learning_rate in [0.1,0.5,1]:
        for max_depth in [1,2,3]:

            with mlflow.start_run(run_name =f'xgboost-learning_rate_{learning_rate}_depth{max_depth}', nested = True):

                model = XGBClassifier(learning_rate = learning_rate, max_depth = max_depth)
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)

                accuracy = accuracy_score(y_test, prediction)

                mlflow.log_metric('accuracy', accuracy)
                mlflow.log_param('learning_rate', learning_rate)
                mlflow.log_param('max_depth', max_depth)
  4. We then decide on the best model and use it in the training.

# 🐧 Penguin Species Classifier API

A production-ready **FastAPI** inference service for classifying penguin species.  
This ML model is containerized with **Docker**, pushed to **Amazon ECR**, and deployed on **AWS EC2** for public access.

---

## Project Overview

This project showcases a complete ML deployment pipeline:
- ✅ Experimentation being done and atrtifacts and metrics being loaded in MLflow.
- ✅ Trained an XGBoost model using `scikit-learn` preprocessing and METAFLOW 
- ✅ Built a `FastAPI` inference service
- ✅ Containerized the app with `Docker`
- ✅ Pushed the image to `Amazon ECR`
- ✅ Deployed it on a public `t2.micro EC2` instance which is enough for the same.
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
The endpoint is running in AWS t2.micro instance as its the most affordable and also enough for the project and dataset.
the API can be accessed at below address.

http://13.201.166.113:8001/


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

# Further Steps:

Next in line is a monitoring pipeline to monitor any sort of drift in the database. As it's a constant

database we can add a fabricated drift to check the authenticity of the the pipeline.
