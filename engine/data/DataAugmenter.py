"""Wodle - Data Augmenter

Author: Chris Hinkson (@cmh02)

The Data Augmenter module provides functionality to combine strength training
workout data with biometric scale measurements using midpoint-based temporal matching.
"""

# Library Imports
import logging

import numpy as np
import pandas as pd

# Internal Modules
from engine.utils.logger import get_logger

# Configure a module-level logger since this is a static utility class
logger = get_logger(name="DataAugmenter", log_file="logs/wodle.log", level=logging.DEBUG)


class DataAugmenter:
    """Wodel DataAugmenter

    Provides static methods to augment and combine workout datasets with biometrics data.
    """

    @staticmethod
    def augment(workoutDf: pd.DataFrame, biometricsDf: pd.DataFrame) -> pd.DataFrame:
        """Augment - Main Augmentation Entry Point

        Orchestrates the data augmentation pipeline by combining workout data with biometrics data.

        Args:
            workoutDf: The workout log DataFrame.
            biometricsDf: The biometrics/scale DataFrame.

        Returns:
            pd.DataFrame: The combined DataFrame with augmented biometric features.
        """
        logger.info("Starting data augmentation pipeline...")

        if workoutDf is None or workoutDf.empty:
            logger.warning("Workout DataFrame is empty or None. Returning empty DataFrame.")
            return pd.DataFrame()

        if biometricsDf is None or biometricsDf.empty:
            logger.warning("Biometrics DataFrame is empty or None. Returning workout DataFrame unmodified.")
            return workoutDf.copy()

        combinedDf = DataAugmenter.augmentStrengthAndBiometrics(workoutDf, biometricsDf)

        logger.info("Successfully completed data augmentation pipeline.")
        return combinedDf

    @staticmethod
    def augmentStrengthAndBiometrics(workoutDf: pd.DataFrame, biometricsDf: pd.DataFrame) -> pd.DataFrame:
        """Augment Strength and Biometrics - Midpoint Matching

        Combines workout data and biometrics data by associating each workout session with
        the closest biometric measurement based on temporal midpoints.

        Args:
            workoutDf: The workout log DataFrame.
            biometricsDf: The biometrics/scale DataFrame.

        Returns:
            pd.DataFrame: The merged DataFrame.
        """
        logger.info("Combining strength and biometrics data using midpoint-based temporal matching...")

        # Sort biometrics by Time to ensure correct midpoint calculations
        sortedBiometrics = biometricsDf.sort_values(by="Time").reset_index(drop=True)

        # Extract times
        biometricsTimes = sortedBiometrics["Time"]

        # Calculate midpoints between consecutive biometric points
        if len(sortedBiometrics) > 1:
            midpoints = biometricsTimes[:-1] + (biometricsTimes[1:].values - biometricsTimes[:-1].values) / 2
        else:
            midpoints = pd.Series(dtype="datetime64[ns]")

        # Map each workout time to the index of the corresponding biometric point
        # All data < midpoint goes to the first biometric point, and >= goes to the second biometric point
        indices = np.searchsorted(midpoints, workoutDf["Time"], side="right")

        # Select the mapped biometric records
        mappedBiometrics = sortedBiometrics.iloc[indices].reset_index(drop=True)

        # Rename Time column to avoid conflict
        mappedBiometrics = mappedBiometrics.rename(columns={"Time": "BiometricTime"})

        # Concatenate columns
        combinedDf = pd.concat([workoutDf.reset_index(drop=True), mappedBiometrics], axis=1)

        return combinedDf
