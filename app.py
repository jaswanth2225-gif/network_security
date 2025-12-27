import sys
import os
import certifi
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train", tags=["Pipeline"])
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict", tags=["Prediction"])
async def predict_route(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)

        df["predicted_column"] = y_pred

        df.to_csv("prediction_output/output.csv")

        # Return HTML table format
        table_html = df.to_html(classes='table table-striped', border=0)
        html_response = f"""
        <html>
        <head>
            <title>Prediction Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>✅ Prediction Results</h2>
            {table_html}
            <p><a href="/docs">← Back to API</a></p>
        </body>
        </html>
        """
        return Response(content=html_response, media_type="text/html")

    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.get("/run-pipeline", tags=["Pipeline"])
async def run_training_pipeline():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training pipeline completed successfully!")
    except Exception as e:
        raise NetworkSecurityException(e, sys)    
    


if __name__=="__main__":
    app_run(app, host="0.0.0.0",port=8080)




