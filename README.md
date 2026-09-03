# Heart Disease Prediction API

A FastAPI-based machine learning API that predicts the presence of heart disease using a Logistic Regression classifier. The application is Dockerized and deployed on Render.

## Project Overview

This project demonstrates how to:

- Train a machine learning classifier using the Heart Disease Dataset
- Save the trained model using Joblib
- Build a REST API using FastAPI
- Dockerize the FastAPI application
- Test the API locally using Swagger UI
- Deploy the application to Render

## Machine Learning Model

The project uses:

- Model: Logistic Regression
- Dataset: Heart Disease Dataset
- Target: Heart disease presence/absence
- Model format: Joblib

The trained model is stored in:

`model/heart_model.joblib`

## Features

The prediction API accepts the following features:

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

## FastAPI Endpoints

### GET /health

Checks whether the API is running.

Example response:

```json
{
  "status": "healthy"
}