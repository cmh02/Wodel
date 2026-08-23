"""Wodle - Model Finder

Author: Chris Hinkson (@cmh02)

The Model Finder module prepares, trains, and evaluates multiple regression models
(Linear Regression, XGBoost, and Random Forest) to predict workout metrics (such as weight)
based on categorical and numerical features.
"""

# Library Imports
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Internal Modules
from engine.utils.logger import get_logger


class ModelFinder:
    """Wodel ModelFinder

    Prepares, trains, and evaluates multiple machine learning models on a workout dataset,
    identifying and returning the best model based on prediction accuracy (R-squared score).
    """

    def __init__(self, target_column: str = "e1RM") -> None:
        """Initialize ModelFinder

        Sets up the default target column and configures the logger.

        Args:
            target_column: The name of the column to predict.
        """
        self.target_column = target_column
        self.logger = get_logger(name="ModelFinder", log_file="logs/wodle.log", level=logging.DEBUG)

    def find_best_model(self, df: pd.DataFrame) -> Pipeline:
        """Find Best Model - Train & Evaluate

        Processes the input dataframe, splits it into training and testing sets,
        builds preprocessing and model pipelines, trains Linear Regression, XGBoost,
        and Random Forest regressors, evaluates their prediction accuracy, and
        returns the best overall pipeline.

        Args:
            df: The pandas DataFrame containing the clean workout data.

        Returns:
            Pipeline: The scikit-learn Pipeline representing the best performing model.
        """
        self.logger.info(f"Starting model search to predict target column: {self.target_column}")

        # Input validation
        if df is None or df.empty:
            self.logger.error("Input DataFrame is empty or None.")
            raise ValueError("Input DataFrame is empty or None.")

        if self.target_column not in df.columns:
            self.logger.error(f"Target column '{self.target_column}' not found in the DataFrame.")
            raise KeyError(f"Target column '{self.target_column}' not found in the DataFrame.")

        # Identify features (exclude target and datetime columns)
        candidate_features = [
            "Name",
            "Set Order",
            "Distance",
            "e1RMLag1",
            "e1RMLag2",
            "e1RMLag3",
            "timeSinceLastWorkout",
            "timeSinceLastSameExercise",
            "exerciseOrderInWorkout",
            "Body Weight",
            "BMI",
            "Body Fat",
            "Fat-Free Mass",
            "Subcutaneous Fat",
            "Visceral Fat",
            "Body Water",
            "Skeletal Muscle",
            "Muscle Mass",
            "Bone Mass",
            "Protein",
            "BMR",
            "Metabolic Age",
        ]
        features = [col for col in candidate_features if col in df.columns and col != self.target_column]

        self.logger.info(f"Selected features for model training: {features}")

        # Segregate categorical and numerical features based on data type
        categorical_features = []
        numerical_features = []
        for col in features:
            if pd.api.types.is_numeric_dtype(df[col]):
                numerical_features.append(col)
            else:
                categorical_features.append(col)

        # Build the dynamic ColumnTransformer
        transformers = []
        if categorical_features:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("to_str", FunctionTransformer(lambda x: pd.DataFrame(x).astype(str), check_inverse=False)),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                )
            )
        if numerical_features:
            transformers.append(
                (
                    "num",
                    Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]),
                    numerical_features,
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        # Prepare feature (X) and target (y) matrices
        X = df[features]
        y = df[self.target_column]

        # Split into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define candidate model architectures
        models: dict[str, Any] = {
            "LinearRegression": LinearRegression(),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        }

        best_score = -float("inf")
        best_pipeline = None
        best_model_name = ""

        # Train and evaluate each model
        for name, model in models.items():
            self.logger.info(f"Training and evaluating: {name}")

            # Package preprocessing and estimator inside a single pipeline
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # Metrics
                r2 = float(r2_score(y_test, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae = float(mean_absolute_error(y_test, y_pred))

                # Calculate MAPE only on non-zero target weights to avoid division by zero (bodyweight exercises)
                nonZeroMask = y_test > 0
                if nonZeroMask.any():
                    mape = float(np.mean(np.abs((y_test[nonZeroMask] - y_pred[nonZeroMask]) / y_test[nonZeroMask])))
                else:
                    mape = 0.0

                self.logger.info(
                    f"{name} performance - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.2f}, MAPE: {mape * 100.0:.2f}%"
                )

                # Calculate error per exercise with average weight to check weight-error trends
                evalDf = pd.DataFrame(
                    {"Name": X_test["Name"], "True_e1RM": y_test, "Abs_Error": np.abs(y_test - y_pred)}
                )
                exerciseStats = (
                    evalDf.groupby("Name").agg(avgWeight=("True_e1RM", "mean"), mae=("Abs_Error", "mean")).reset_index()
                )

                self.logger.info(f"{name} error breakdown by exercise (top 5 heaviest exercises):")
                nonZeroStats = exerciseStats[exerciseStats["avgWeight"] > 0]
                sortedStats = nonZeroStats.sort_values(by="avgWeight", ascending=False)
                for _, row in sortedStats.head(5).iterrows():
                    self.logger.info(
                        f"""  - {row["Name"]}:
                            Avg Weight = {row["avgWeight"]:.1f},
                            MAE = {row["mae"]:.2f}"""
                    )
                self.logger.info(f"""{name} error breakdown by exercise (bottom 5 lightest exercises with weight):""")
                for _, row in sortedStats.tail(5).iterrows():
                    self.logger.info(
                        f"""  - {row["Name"]}:
                            Avg Weight = {row["avgWeight"]:.1f},
                            MAE = {row["mae"]:.2f}"""
                    )

                if r2 > best_score:
                    best_score = r2
                    best_pipeline = pipeline
                    best_model_name = name

            except Exception:
                self.logger.exception(f"Error training {name}")

        if best_pipeline is None:
            raise RuntimeError("All models failed to train successfully.")

        self.logger.info(
            f"""Successfully determined best model: {best_model_name}
                with R2 Score of {best_score:.4f}"""
        )
        return best_pipeline


if __name__ == "__main__":
    # Demonstration of the ModelFinder
    from engine.data.Pipeline import Pipeline as DataPipeline

    # Load, clean, and engineer features using the integrated Pipeline
    data = DataPipeline.run("data/strong_workouts.csv", "data/renpho.csv")

    finder = ModelFinder(target_column="e1RM")
    best_model = finder.find_best_model(data)

    # Perform a test prediction
    test_row = pd.DataFrame(
        [
            {
                "Name": "Romanian Deadlift (Barbell)",
                "Set Order": "3",
                "Distance": 0.0,
                "e1RMLag1": 225.0,
                "e1RMLag2": 220.0,
                "e1RMLag3": 215.0,
                "timeSinceLastWorkout": 2.0,
                "timeSinceLastSameExercise": 7.0,
                "exerciseOrderInWorkout": 2,
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
        ]
    )
    predicted_e1RM = best_model.predict(test_row)[0]
    print("\n--- Demonstration Success ---")
    print(f"""Predicted e1RM for Romanian Deadlift (Barbell) (Set 3):
            {predicted_e1RM:.2f} lbs""")
    print("-----------------------------\n")
