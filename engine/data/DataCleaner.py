"""
Wodle - Data Cleaner
Author: Chris Hinkson (@cmh02)

The Data Cleaner module provides utility static methods for cleaning workout datasets,
such as filtering out rows containing NaN values or removing distance-based records.
"""

import logging
import pandas as pd

from engine.utils.logger import get_logger

# Configure a module-level logger since this is a static utility class
logger = get_logger(
    name="DataCleaner",
    log_file="logs/wodle.log",
    level=logging.DEBUG
)

class DataCleaner:
    """
    Wodel DataCleaner

    Provides static cleaning methods to prepare workout data for feature engineering.
    """

    @staticmethod
    def removeAnyNaN(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove Any NaN - Row Filtering

        Filters out any rows from the DataFrame that contain one or more NaN (missing) values.

        Args:
            df: The input pandas DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with all NaN rows removed.
        """
        logger.info("Executing removeAnyNaN filter...")
        if df is None:
            logger.error("Input DataFrame is None.")
            raise ValueError("Input DataFrame is None.")

        initialCount = len(df)
        cleanedDf = df.dropna()
        finalCount = len(cleanedDf)
        removedCount = initialCount - finalCount
        
        logger.info(f"Removed {removedCount} rows containing NaNs. Remaining rows: {finalCount}.")
        return cleanedDf

    @staticmethod
    def removeAnyDistance(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove Any Distance - Cardio Filtering

        Filters out distance-based cardio exercises by removing any rows where the
        Distance metric is greater than 0.

        Args:
            df: The input pandas DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame containing only strength-based (non-distance) records.
        """
        logger.info("Executing removeAnyDistance filter...")
        if df is None:
            logger.error("Input DataFrame is None.")
            raise ValueError("Input DataFrame is None.")

        # Keep rows where Distance is 0, NaN, or <= 0
        initialCount = len(df)
        cleanedDf = df[~(df["Distance"] > 0)]
        finalCount = len(cleanedDf)
        removedCount = initialCount - finalCount
        
        logger.info(f"Removed {removedCount} rows with distance > 0. Remaining rows: {finalCount}.")
        return cleanedDf