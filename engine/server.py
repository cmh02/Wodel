"""Wodle - API Server

Author: Chris Hinkson (@cmh02)

FastAPI backend server that handles file uploads, runs the data engineering pipeline,
trains machine learning models, and serves predictions.
"""

# Library Imports
import logging
import os
import tempfile
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Internal Modules
from engine.data.Pipeline import Pipeline as DataPipeline
from engine.model.ModelFinder import ModelFinder
from engine.utils.logger import get_logger

# Configure logging
logger = get_logger(name="Server", log_file="logs/wodle.log", level=logging.DEBUG)

app = FastAPI(title="Wodle ML API Server")

# Allow requests from React frontend (Vite defaults to port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model state
trainedPipeline: Any = None
bestModelName: str = ""
bestModelScore: float = -float("inf")
modelMetrics: dict[str, Any] = {}
processedDataset: pd.DataFrame | None = None


class PredictionRequest(BaseModel):
    """Wodel PredictionRequest

    Data model representing the inputs required to predict e1RM in Advanced Mode.
    """

    Name: str
    Set_Order: str  # Note: Set Order is expected as string in preprocessor
    Distance: float
    e1RMLag1: float
    e1RMLag2: float
    e1RMLag3: float
    timeSinceLastWorkout: float
    timeSinceLastSameExercise: float
    exerciseOrderInWorkout: int
    Age: float = 24.0
    Body_Weight: float
    BMI: float
    Body_Fat: float
    Fat_Free_Mass: float
    Subcutaneous_Fat: float
    Visceral_Fat: float
    Body_Water: float
    Skeletal_Muscle: float
    Muscle_Mass: float
    Bone_Mass: float
    Protein: float
    BMR: float
    Metabolic_Age: float


class SimplePredictionRequest(BaseModel):
    """Wodel SimplePredictionRequest

    Data model for user-friendly simple predictions.
    """

    Name: str
    Set_Order: str
    Reps: int = 8
    Weight: float = 185.0
    exerciseOrderInWorkout: int = 1
    birthday: str | None = None
    current_age: float | None = None


@app.post("/api/train")
async def trainModel(
    workoutFile: UploadFile = File(...),  # noqa: B008
    biometricsFile: UploadFile = File(...),  # noqa: B008
    birthday: str | None = Form(None),  # noqa: B008
    currentAge: float | None = Form(None),  # noqa: B008
) -> dict[str, Any]:
    """Train Model - Upload Files and Run Pipeline

    Receives the raw workout and biometrics files, saves them temporarily, executes the
    data processing pipeline, trains multiple regression models, and stores the best one.

    Args:
        workoutFile: The uploaded Strong workout logs CSV.
        biometricsFile: The uploaded Renpho biometrics CSV.
        birthday: Optional birthdate string (YYYY-MM-DD).
        currentAge: Optional current age in years.

    Returns:
        dict: A summary of training metrics and the best model name.
    """
    global trainedPipeline, bestModelName, bestModelScore, modelMetrics, processedDataset

    logger.info("Received request to train models.")

    # Create temporary files to save the uploads
    with (
        tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as workoutTemp,
        tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as biometricsTemp,
    ):
        workoutTempPath = workoutTemp.name
        biometricsTempPath = biometricsTemp.name

        try:
            # Write contents to temporary files
            workoutContent = await workoutFile.read()
            biometricsContent = await biometricsFile.read()

            workoutTemp.write(workoutContent)
            biometricsTemp.write(biometricsContent)
            workoutTemp.flush()
            biometricsTemp.flush()
            workoutTemp.close()
            biometricsTemp.close()

            # Execute pipeline
            logger.info("Running data preprocessing pipeline...")
            data = DataPipeline.run(
                workoutTempPath,
                biometricsTempPath,
                birthday=birthday,
                current_age=currentAge,
            )

            # Perform model training and analysis
            logger.info("Training models...")
            finder = ModelFinder(target_column="e1RM")

            candidateModels = {
                "LinearRegression": LinearRegression(),
                "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            }

            bestScore = -float("inf")
            bestPipe = None
            bestName = ""
            metricsSummary = {}

            for name, model in candidateModels.items():
                result = finder.performModelAnalysis(data, name, model)
                if result is None:
                    continue

                metricsSummary[name] = result["metrics"]
                r2 = result["metrics"]["r2"]
                if r2 > bestScore:
                    bestScore = r2
                    bestPipe = result["pipeline"]
                    bestName = name

            if bestPipe is None:
                raise HTTPException(status_code=500, detail="All models failed to train successfully.")

            # Update global state
            trainedPipeline = bestPipe
            bestModelName = bestName
            bestModelScore = bestScore
            modelMetrics = metricsSummary
            processedDataset = data

            logger.info(f"Model training complete. Best model: {bestName} with R2: {bestScore:.4f}")
            return {
                "status": "success",
                "bestModel": bestModelName,
                "bestR2": bestModelScore,
                "metrics": modelMetrics,
            }

        except Exception as e:
            logger.exception("Error occurred during model training.")
            raise HTTPException(status_code=500, detail=str(e)) from e

        finally:
            # Clean up temp files
            if os.path.exists(workoutTempPath):
                os.remove(workoutTempPath)
            if os.path.exists(biometricsTempPath):
                os.remove(biometricsTempPath)


@app.post("/api/predict")
async def predictTarget(request: PredictionRequest) -> dict[str, Any]:
    """Predict Target - Predict e1RM Using Trained Model

    Prepares the prediction input row, feeds it to the best trained model pipeline,
    and returns the predicted value.

    Args:
        request: The PredictionRequest input parameters.

    Returns:
        dict: The prediction result.
    """
    global trainedPipeline

    if trainedPipeline is None:
        raise HTTPException(status_code=400, detail="No model has been trained yet. Please train the model first.")

    try:
        # Convert Request object into DataFrame mapping columns matching features
        inputData = {
            "Name": [request.Name],
            "Set Order": [request.Set_Order],
            "Distance": [request.Distance],
            "e1RMLag1": [request.e1RMLag1],
            "e1RMLag2": [request.e1RMLag2],
            "e1RMLag3": [request.e1RMLag3],
            "timeSinceLastWorkout": [request.timeSinceLastWorkout],
            "timeSinceLastSameExercise": [request.timeSinceLastSameExercise],
            "exerciseOrderInWorkout": [request.exerciseOrderInWorkout],
            "Age": [request.Age],
            "Body Weight": [request.Body_Weight],
            "BMI": [request.BMI],
            "Body Fat": [request.Body_Fat],
            "Fat-Free Mass": [request.Fat_Free_Mass],
            "Subcutaneous Fat": [request.Subcutaneous_Fat],
            "Visceral Fat": [request.Visceral_Fat],
            "Body Water": [request.Body_Water],
            "Skeletal Muscle": [request.Skeletal_Muscle],
            "Muscle Mass": [request.Muscle_Mass],
            "Bone Mass": [request.Bone_Mass],
            "Protein": [request.Protein],
            "BMR": [request.BMR],
            "Metabolic Age": [request.Metabolic_Age],
        }

        testRow = pd.DataFrame(inputData)
        predictedVal = float(trainedPipeline.predict(testRow)[0])

        return {
            "prediction": predictedVal,
            "target": "e1RM",
            "modelUsed": bestModelName,
        }

    except Exception as e:
        logger.exception("Error occurred during prediction.")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/predict/simple")
async def predictSimple(request: SimplePredictionRequest) -> dict[str, Any]:
    """Predict Simple - Automated Lookup Prediction

    Calculates current set e1RM, looks up lag features and biometrics from the
    trained dataset, and performs model prediction.
    """
    global trainedPipeline, processedDataset, bestModelName

    if trainedPipeline is None:
        raise HTTPException(status_code=400, detail="No model has been trained yet. Please train the model first.")

    try:
        # Calculate current set e1RM (Epley formula: weight * (1 + reps / 30))
        current_set_e1RM = request.Weight * (1.0 + request.Reps / 30.0)

        # Default fallbacks
        lag1, lag2, lag3 = current_set_e1RM, current_set_e1RM, current_set_e1RM
        timeSinceLastWorkout = 2.0
        timeSinceLastSameExercise = 7.0
        calculated_age = request.current_age if request.current_age is not None else 24.0

        biometrics = {
            "Body Weight": 140.0,
            "BMI": 23.0,
            "Body Fat": 15.0,
            "Fat-Free Mass": 119.0,
            "Subcutaneous Fat": 13.0,
            "Visceral Fat": 6.0,
            "Body Water": 61.0,
            "Skeletal Muscle": 55.0,
            "Muscle Mass": 113.0,
            "Bone Mass": 6.0,
            "Protein": 19.0,
            "BMR": 1540.0,
            "Metabolic Age": 18.0,
        }

        if processedDataset is not None and not processedDataset.empty:
            # Filter history for the selected exercise
            exercise_rows = processedDataset[processedDataset["Name"] == request.Name]
            if not exercise_rows.empty:
                recent_ex = exercise_rows.iloc[-1]
                if "e1RMLag1" in recent_ex:
                    lag1 = float(recent_ex.get("e1RMLag1", current_set_e1RM))
                    lag2 = float(recent_ex.get("e1RMLag2", current_set_e1RM))
                    lag3 = float(recent_ex.get("e1RMLag3", current_set_e1RM))
                if "timeSinceLastSameExercise" in recent_ex:
                    timeSinceLastSameExercise = float(recent_ex.get("timeSinceLastSameExercise", 7.0))

            latest_row = processedDataset.iloc[-1]
            if "timeSinceLastWorkout" in latest_row:
                timeSinceLastWorkout = float(latest_row.get("timeSinceLastWorkout", 2.0))

            for key in biometrics:
                if key in latest_row:
                    biometrics[key] = float(latest_row[key])

            if "Age" in latest_row:
                calculated_age = float(latest_row["Age"])

        # Override age if birthday or current_age passed directly in request
        if request.birthday:
            try:
                b_date = pd.to_datetime(request.birthday)
                today = pd.Timestamp.now()
                calculated_age = float((today - b_date).days / 365.2425)
            except Exception:
                pass
        elif request.current_age is not None:
            calculated_age = float(request.current_age)

        inputData = {
            "Name": [request.Name],
            "Set Order": [request.Set_Order],
            "Distance": [0.0],
            "e1RMLag1": [lag1],
            "e1RMLag2": [lag2],
            "e1RMLag3": [lag3],
            "timeSinceLastWorkout": [timeSinceLastWorkout],
            "timeSinceLastSameExercise": [timeSinceLastSameExercise],
            "exerciseOrderInWorkout": [request.exerciseOrderInWorkout],
            "Age": [calculated_age],
            "Body Weight": [biometrics["Body Weight"]],
            "BMI": [biometrics["BMI"]],
            "Body Fat": [biometrics["Body Fat"]],
            "Fat-Free Mass": [biometrics["Fat-Free Mass"]],
            "Subcutaneous Fat": [biometrics["Subcutaneous Fat"]],
            "Visceral Fat": [biometrics["Visceral Fat"]],
            "Body Water": [biometrics["Body Water"]],
            "Skeletal Muscle": [biometrics["Skeletal Muscle"]],
            "Muscle Mass": [biometrics["Muscle Mass"]],
            "Bone Mass": [biometrics["Bone Mass"]],
            "Protein": [biometrics["Protein"]],
            "BMR": [biometrics["BMR"]],
            "Metabolic Age": [biometrics["Metabolic Age"]],
        }

        testRow = pd.DataFrame(inputData)
        predictedVal = float(trainedPipeline.predict(testRow)[0])

        return {
            "prediction": predictedVal,
            "target": "e1RM",
            "modelUsed": bestModelName,
            "derivedFeatures": {
                "currentSete1RM": current_set_e1RM,
                "calculatedAge": calculated_age,
                "e1RMLag1": lag1,
                "e1RMLag2": lag2,
                "e1RMLag3": lag3,
                "timeSinceLastWorkout": timeSinceLastWorkout,
                "timeSinceLastSameExercise": timeSinceLastSameExercise,
                "bodyWeight": biometrics["Body Weight"],
                "bmi": biometrics["BMI"],
                "bodyFat": biometrics["Body Fat"],
            },
        }

    except Exception as e:
        logger.exception("Error occurred during simple prediction.")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/status")
async def getStatus() -> dict[str, Any]:
    """Get Status - Model State Check

    Checks if a model is currently trained and returns status metadata.

    Returns:
        dict: Model state details.
    """
    return {
        "modelTrained": trainedPipeline is not None,
        "bestModel": bestModelName,
        "bestR2": bestModelScore if trainedPipeline is not None else None,
    }
