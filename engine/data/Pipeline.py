"""Wodle - Data Pipeline

Author: Chris Hinkson (@cmh02)

The Data Pipeline module coordinates the execution of loading, cleaning, and feature
engineering steps to produce a finalized dataset for machine learning models.
"""

# Library Imports
import logging

import pandas as pd

# Internal Modules
from engine.data.DataAugmenter import DataAugmenter
from engine.data.DataCleaner import DataCleaner
from engine.data.DataLoader import DataLoader
from engine.data.FeatureBuilder import FeatureBuilder
from engine.utils.logger import get_logger

# Configure a module-level logger since this is a static utility class
logger = get_logger(name="Pipeline", log_file="logs/wodle.log", level=logging.DEBUG)


class Pipeline:
    """Wodel Pipeline

    Provides a static interface to orchestrate the entire data preprocessing and
    feature engineering pipeline in a single step.
    """

    @staticmethod
    def run(filePath: str, biometricsFilePath: str = "data/renpho.csv") -> pd.DataFrame:
        """Run - Execute Data Pipeline

        Coordinates loading data from CSV, removing distance-based cardio exercises,
        building session-level features/lags, and removing any incomplete (NaN) records.
        Additionally, loads biometrics data and augments the workout dataset.

        Args:
            filePath: The path to the CSV file to load and process.
            biometricsFilePath: The path to the biometrics CSV file to load.

        Returns:
            pd.DataFrame: The fully cleaned, engineered, and augmented DataFrame ready for model fitting.
        """
        logger.info(f"Starting data pipeline for: {filePath}")

        # Load Workout Data
        rawData = DataLoader.loadFromStrongCSV(filePath)

        # Clean Cardio/Distance
        strengthData = DataCleaner.removeAnyDistance(rawData)

        # Build Features (estimated 1RMs, lags, elapsed times)
        featuredData = FeatureBuilder.buildFeatures(strengthData)

        # Remove any rows with NaN values (e.g. initial lag NaNs)
        cleanedData = DataCleaner.removeAnyNaN(featuredData)

        # Load Biometrics Data
        biometricsData = DataLoader.loadFromRenphoCSV(biometricsFilePath)

        # Augment with Biometrics
        augmentedData = DataAugmenter.augment(cleanedData, biometricsData)

        # Remove any remaining NaN values resulting from augmentation (if any)
        finalData = DataCleaner.removeAnyNaN(augmentedData)

        logger.info(f"Data pipeline execution completed. Final row count: {len(finalData)}.")
        return finalData
