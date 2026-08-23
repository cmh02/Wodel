"""Wodle - Data Loader

Author: Chris Hinkson (@cmh02)

The Data Loader module is responsible for parsing and loading data from a variety
of sources. It provides a simple API for accessing and manipulating data.

# Strength Training Workout Data

The loader converts any data format for strenght training workout logs
into a single data format that can be used within the rest of this service.
This data format is a Pandas Dataset where each row is a single set of an
exercise with the following columns:

- Time: DateTime of the measurement
- Name: The name of the strength exercise (machine, activity, etc.)
- Set Order: The order of the set in the exercise
- Weight: The weight lifted on each rep of the exercise if the exercise is weight-related
- Reps: The number of reps completed in the set
- Distance: The distance covered in the exercise if the exercise is distance-related

# Scale / Biometrics Data

The loader converts any data format for body scale or biometrics data
into a single data format that can be used within the rest of this service.
This data format is a Pandas Dataset where each row is a single measurement
with the following columns:

- Time: DateTime of the measurement
- Body Weight: The weight of the individual in lbs
- BMI: Body Mass Index
- Body Fat: Body fat percentage
- Fat-Free Mass: Fat-free mass in lbs
- Subcutaneous Fat: Subcutaneous fat percentage
- Visceral Fat: Visceral fat rating
- Body Water: Body water percentage
- Skeletal Muscle: Skeletal muscle percentage
- Muscle Mass: Muscle mass in lbs
- Bone Mass: Bone mass in lbs
- Protein: Protein percentage
- BMR: Basal Metabolic Rate in kcal
- Metabolic Age: Metabolic age in years
"""

# Library Imports
import logging

import pandas as pd

# Internal Modules
from engine.utils.logger import get_logger

# Configure a module-level logger since this is a static utility class
logger = get_logger(name="DataLoader", log_file="logs/wodle.log", level=logging.DEBUG)


class DataLoader:
    """Wodel DataLoader

    Provides static methods to load and parse workout datasets.
    """

    @staticmethod
    def loadFromStrongCSV(filePath: str) -> pd.DataFrame:
        """Load Data - Strong App Format

        This helper loads CSV data from the Strong App in the standard export format
        and parses it into the target schema.

        Args:
            filePath: The path to the CSV file to load.

        Returns:
            pd.DataFrame: The loaded and cleaned pandas DataFrame.
        """
        logger.info(f"Attempting to load Strong CSV data from: {filePath}")
        try:
            # Read CSV
            df = pd.read_csv(filePath)

            # Map required columns
            column_mapping = {
                "Date": "Time",
                "Exercise Name": "Name",
                "Set Order": "Set Order",
                "Weight": "Weight",
                "Reps": "Reps",
                "Distance": "Distance",
            }

            # Verify if expected columns exist
            for src_col in column_mapping:
                if src_col not in df.columns:
                    raise KeyError(f"Expected column '{src_col}' not found in CSV.")

            # Select and rename columns
            df_cleaned = df[list(column_mapping.keys())].rename(columns=column_mapping)

            # Convert Time to datetime
            df_cleaned["Time"] = pd.to_datetime(df_cleaned["Time"])

            logger.info(f"Successfully loaded and parsed {len(df_cleaned)} rows of data.")
            return df_cleaned

        except Exception:
            logger.exception(f"Failed to load CSV from {filePath}!")
            raise

    @staticmethod
    def loadFromRenphoCSV(filePath: str) -> pd.DataFrame:
        """Load Data - Renpho Scale Format

        This helper loads CSV data from the Renpho Scale in the standard export format
        and parses it into the target schema.

        Args:
            filePath: The path to the CSV file to load.

        Returns:
            pd.DataFrame: The loaded and cleaned pandas DataFrame.
        """
        logger.info(f"Attempting to load Renpho CSV data from: {filePath}")
        try:
            # Read CSV
            df = pd.read_csv(filePath)

            # Strip leading/trailing whitespaces from column names
            df.columns = df.columns.str.strip()

            # Map required columns
            column_mapping = {
                "Time of Measurement": "Time",
                "Weight(lb)": "Body Weight",
                "BMI": "BMI",
                "Body Fat(%)": "Body Fat",
                "Fat-Free Mass(lb)": "Fat-Free Mass",
                "Subcutaneous Fat(%)": "Subcutaneous Fat",
                "Visceral Fat": "Visceral Fat",
                "Body Water(%)": "Body Water",
                "Skeletal Muscle(%)": "Skeletal Muscle",
                "Muscle Mass(lb)": "Muscle Mass",
                "Bone Mass(lb)": "Bone Mass",
                "Protein (%)": "Protein",
                "BMR(kcal)": "BMR",
                "Metabolic Age": "Metabolic Age",
            }

            # Verify if expected columns exist
            for src_col in column_mapping:
                if src_col not in df.columns:
                    raise KeyError(f"Expected column '{src_col}' not found in CSV.")

            # Select and rename columns
            df_cleaned = df[list(column_mapping.keys())].rename(columns=column_mapping)

            # Convert Time to datetime
            df_cleaned["Time"] = pd.to_datetime(df_cleaned["Time"])

            logger.info(f"Successfully loaded and parsed {len(df_cleaned)} rows of Renpho data.")
            return df_cleaned

        except Exception:
            logger.exception(f"Failed to load Renpho CSV from {filePath}!")
            raise
