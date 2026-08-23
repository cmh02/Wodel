"""Wodle - Feature Builder

Author: Chris Hinkson (@cmh02)

The Feature Builder module takes raw workout datasets and constructs advanced engineered
features (e1RM, lags, elapsed times, and session sequence orders) to improve prediction models.
"""

# Library Imports
import logging

import pandas as pd

# Internal Modules
from engine.utils.logger import get_logger

# Configure a module-level logger since this is a static utility class
logger = get_logger(name="FeatureBuilder", log_file="logs/wodle.log", level=logging.DEBUG)


class FeatureBuilder:
    """Wodel FeatureBuilder

    Provides static methods to orchestrate the feature transformation pipeline on workout data.
    """

    @staticmethod
    def buildFeatures(df: pd.DataFrame) -> pd.DataFrame:
        """Build Features - Pipeline Execution

        Orchestrates the creation of advanced features including Epley 1RM calculation,
        session-level lag features (Lag1, Lag2, Lag3), elapsed time indicators, and
        the order of exercises within a single workout session.

        Args:
            df: The raw pandas DataFrame loaded from DataLoader.

        Returns:
            pd.DataFrame: The enriched DataFrame with new engineered features.
        """
        logger.info("Starting feature engineering pipeline...")

        if df is None or df.empty:
            logger.error("Input DataFrame is empty or None.")
            raise ValueError("Input DataFrame is empty or None.")

        # Create a copy to prevent in-place modification of the original DataFrame
        engineeredDf = df.copy()

        # Calculate e1RM using the Epley formula: e1RM = Weight * (1 + Reps / 30)
        engineeredDf = FeatureBuilder._calculateE1RM(engineeredDf)

        # Calculate Session Lag Features (Lag1, Lag2, Lag3 max e1RM)
        engineeredDf = FeatureBuilder._calculateLagFeatures(engineeredDf)

        # Calculate Time Indicators
        engineeredDf = FeatureBuilder._calculateTimeIndicators(engineeredDf)

        # Calculate Exercise Sequence Order in Workout
        engineeredDf = FeatureBuilder._calculateExerciseOrder(engineeredDf)

        # Drop temporary columns
        engineeredDf = engineeredDf.drop(columns=["Date", "DateLag1"])

        logger.info("Successfully completed feature engineering pipeline.")
        return engineeredDf

    @staticmethod
    def _calculateE1RM(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate e1RM using the Epley formula and set up the temporary Date column."""
        logger.info("Calculating Estimated 1-Rep Max (e1RM) using Epley formula...")
        df["e1RM"] = df["Weight"] * (1.0 + df["Reps"] / 30.0)

        # Remove old Weight and Reps columns
        df = df.drop(columns=["Weight", "Reps"])

        # Create a temporary Date column (date-only) for grouping sessions
        df["Date"] = df["Time"].dt.date
        return df

    @staticmethod
    def _calculateLagFeatures(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Session Lag Features (Lag1, Lag2, Lag3 max e1RM)."""
        logger.info("Computing session-level lag features (Lag1, Lag2, Lag3)...")

        # Find the max e1RM achieved for each exercise on each day
        sessionMax = df.groupby(["Name", "Date"])["e1RM"].max().reset_index()
        sessionMax = sessionMax.sort_values(by=["Name", "Date"])

        # Shift the max e1RM values to find historical performance
        sessionMax["e1RMLag1"] = sessionMax.groupby("Name")["e1RM"].shift(1)
        sessionMax["e1RMLag2"] = sessionMax.groupby("Name")["e1RM"].shift(2)
        sessionMax["e1RMLag3"] = sessionMax.groupby("Name")["e1RM"].shift(3)

        # Store the previous session date to calculate elapsed time between same exercises
        sessionMax["DateLag1"] = sessionMax.groupby("Name")["Date"].shift(1)

        # Merge the computed lags back into the main DataFrame
        df = pd.merge(
            df,
            sessionMax[["Name", "Date", "e1RMLag1", "e1RMLag2", "e1RMLag3", "DateLag1"]],
            on=["Name", "Date"],
            how="left",
        )
        return df

    @staticmethod
    def _calculateTimeIndicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time elapsed since last workout and last same exercise."""
        logger.info("Generating time indicators...")

        # Time since last workout (of any kind) in days
        uniqueTimes = pd.Series(df["Time"].unique()).sort_values()
        timeDiffs = uniqueTimes.diff()
        timeDiffsDays = timeDiffs.dt.total_seconds() / (24.0 * 3600.0)

        timeMapping = pd.DataFrame({"Time": uniqueTimes, "timeSinceLastWorkout": timeDiffsDays})
        df = pd.merge(df, timeMapping, on="Time", how="left")

        # Time since last same exercise in days
        df["Date"] = pd.to_datetime(df["Date"])
        df["DateLag1"] = pd.to_datetime(df["DateLag1"])
        df["timeSinceLastSameExercise"] = (df["Date"] - df["DateLag1"]).dt.days
        return df

    @staticmethod
    def _calculateExerciseOrder(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exercise Sequence Order in Workout."""
        logger.info("Determining order of exercises within each workout session...")

        def getExerciseOrder(group: pd.DataFrame) -> pd.Series:
            """Helper to calculate exercise ordering inside a session."""
            uniqueNames = []
            for name in group["Name"]:
                if name not in uniqueNames:
                    uniqueNames.append(name)
            nameToOrder = {name: i for i, name in enumerate(uniqueNames)}
            return group["Name"].map(nameToOrder)

        df["exerciseOrderInWorkout"] = df.groupby("Time", group_keys=False).apply(getExerciseOrder)
        return df
