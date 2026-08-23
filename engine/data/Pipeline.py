"""
Wodle - Data Pipeline
Author: Chris Hinkson (@cmh02)

The Data Pipeline module coordinates the execution of loading, cleaning, and feature
engineering steps to produce a finalized dataset for machine learning models.
"""

# Library Imports
import logging

import pandas as pd

# Internal Modules
from engine.data.DataCleaner import DataCleaner
from engine.data.DataLoader import DataLoader
from engine.data.FeatureBuilder import FeatureBuilder
from engine.utils.logger import get_logger

# Configure a module-level logger since this is a static utility class
logger = get_logger(
    name="Pipeline",
    log_file="logs/wodle.log",
    level=logging.DEBUG
)

class Pipeline:
    """
    Wodel Pipeline

    Provides a static interface to orchestrate the entire data preprocessing and
    feature engineering pipeline in a single step.
    """

    @staticmethod
    def run(filePath: str) -> pd.DataFrame:
        """
        Run - Execute Data Pipeline

        Coordinates loading data from CSV, removing distance-based cardio exercises,
        building session-level features/lags, and removing any incomplete (NaN) records.

        Args:
            filePath: The path to the CSV file to load and process.

        Returns:
            pd.DataFrame: The fully cleaned and engineered DataFrame ready for model fitting.
        """
        logger.info(f"Starting data pipeline for: {filePath}")
        
        # Load Data
        rawData = DataLoader.loadFromStrongCSV(filePath)
        
        # Clean Cardio/Distance
        strengthData = DataCleaner.removeAnyDistance(rawData)
        
        # Build Features (estimated 1RMs, lags, elapsed times)
        featuredData = FeatureBuilder.buildFeatures(strengthData)
        
        # Remove any rows with NaN values (e.g. initial lag NaNs)
        finalData = DataCleaner.removeAnyNaN(featuredData)
        
        logger.info(f"Data ipeline execution completed. Final row count: {len(finalData)}.")
        return finalData
