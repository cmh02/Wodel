"""
Wodle - Feature Builder
Author: Chris Hinkson (@cmh02)

The Feature Builder module takes raw workout datasets and constructs advanced engineered
features (e1RM, lags, elapsed times, and session sequence orders) to improve prediction models.
"""

import logging
import pandas as pd

from engine.utils.logger import get_logger

class FeatureBuilder:
    """
    Wodel FeatureBuilder

    Initializes the feature builder and orchestrates the transformation pipeline on workout data.
    """

    def __init__(self) -> None:
        """
        Initialize FeatureBuilder

        Sets up the logger for tracing feature creation.
        """
        self.logger = get_logger(
            name="FeatureBuilder",
            log_file="logs/wodle.log",
            level=logging.DEBUG
        )

    def buildFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build Features - Pipeline Execution

        Orchestrates the creation of advanced features including Epley 1RM calculation,
        session-level lag features (Lag1, Lag2, Lag3), elapsed time indicators, and
        the order of exercises within a single workout session.

        Args:
            df: The raw pandas DataFrame loaded from DataLoader.

        Returns:
            pd.DataFrame: The enriched DataFrame with new engineered features.
        """
        self.logger.info("Starting feature engineering pipeline...")
        
        if df is None or df.empty:
            self.logger.error("Input DataFrame is empty or None.")
            raise ValueError("Input DataFrame is empty or None.")

        # 1. Create a copy to prevent in-place modification of the original DataFrame
        engineeredDf = df.copy()

        # 2. Calculate e1RM using the Epley formula: e1RM = Weight * (1 + Reps / 30)
        self.logger.info("Calculating Estimated 1-Rep Max (e1RM) using Epley formula...")
        engineeredDf["e1RM"] = engineeredDf["Weight"] * (1.0 + engineeredDf["Reps"] / 30.0)
        
        # Remove old Weight and Reps columns
        engineeredDf = engineeredDf.drop(columns=["Weight", "Reps"])

        # Create a temporary Date column (date-only) for grouping sessions
        engineeredDf["Date"] = engineeredDf["Time"].dt.date

        # 3. Calculate Session Lag Features (Lag1, Lag2, Lag3 max e1RM)
        self.logger.info("Computing session-level lag features (Lag1, Lag2, Lag3)...")
        
        # Find the max e1RM achieved for each exercise on each day
        sessionMax = (
            engineeredDf.groupby(["Name", "Date"])["e1RM"]
            .max()
            .reset_index()
        )
        sessionMax = sessionMax.sort_values(by=["Name", "Date"])

        # Shift the max e1RM values to find historical performance
        sessionMax["e1RMLag1"] = sessionMax.groupby("Name")["e1RM"].shift(1)
        sessionMax["e1RMLag2"] = sessionMax.groupby("Name")["e1RM"].shift(2)
        sessionMax["e1RMLag3"] = sessionMax.groupby("Name")["e1RM"].shift(3)

        # Store the previous session date to calculate elapsed time between same exercises
        sessionMax["DateLag1"] = sessionMax.groupby("Name")["Date"].shift(1)

        # Merge the computed lags back into the main DataFrame
        engineeredDf = pd.merge(
            engineeredDf,
            sessionMax[["Name", "Date", "e1RMLag1", "e1RMLag2", "e1RMLag3", "DateLag1"]],
            on=["Name", "Date"],
            how="left"
        )

        # 4. Calculate Time Indicators
        self.logger.info("Generating time indicators...")

        # Time since last workout (of any kind) in days
        uniqueTimes = pd.Series(engineeredDf["Time"].unique()).sort_values()
        timeDiffs = uniqueTimes.diff()
        timeDiffsDays = timeDiffs.dt.total_seconds() / (24.0 * 3600.0)
        
        timeMapping = pd.DataFrame({
            "Time": uniqueTimes,
            "timeSinceLastWorkout": timeDiffsDays
        })
        engineeredDf = pd.merge(engineeredDf, timeMapping, on="Time", how="left")

        # Time since last same exercise in days
        engineeredDf["Date"] = pd.to_datetime(engineeredDf["Date"])
        engineeredDf["DateLag1"] = pd.to_datetime(engineeredDf["DateLag1"])
        engineeredDf["timeSinceLastSameExercise"] = (
            (engineeredDf["Date"] - engineeredDf["DateLag1"]).dt.days
        )

        # 5. Calculate Exercise Sequence Order in Workout
        self.logger.info("Determining order of exercises within each workout session...")

        def getExerciseOrder(group: pd.DataFrame) -> pd.Series:
            """Helper to calculate exercise ordering inside a session."""
            uniqueNames = []
            for name in group["Name"]:
                if name not in uniqueNames:
                    uniqueNames.append(name)
            nameToOrder = {name: i for i, name in enumerate(uniqueNames)}
            return group["Name"].map(nameToOrder)

        engineeredDf["exerciseOrderInWorkout"] = (
            engineeredDf.groupby("Time", group_keys=False)
            .apply(getExerciseOrder)
        )

        # 6. Drop temporary columns
        engineeredDf = engineeredDf.drop(columns=["Date", "DateLag1"])
        
        self.logger.info("Successfully completed feature engineering pipeline.")
        return engineeredDf

if __name__ == "__main__":
    # Demonstration of the FeatureBuilder
    from engine.data.DataLoader import DataLoader
    
    loader = DataLoader()
    if loader.loadFromStrongCSV("data/strong_workouts.csv"):
        rawData = loader.getData()
        if rawData is not None:

            pd.set_option("display.max_rows", None)

            # Optional: also show all columns if they are getting truncated
            pd.set_option("display.max_columns", None)

            # Optional: prevent long text inside columns from being truncated
            pd.set_option("display.max_colwidth", None)

            builder = FeatureBuilder()
            features = builder.buildFeatures(rawData)
            
            print("\n--- Demonstration Success ---")
            print("First 10 rows of engineered dataset:")
            print(features.tail(10))
            print("\nColumns in new dataset:")
            print(list(features.columns))
            print("-----------------------------\n")
