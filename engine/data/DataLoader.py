"""
Wodle - Data Loader
Author: Chris Hinkson (@cmh02)

The Data Loader module is responsible for parsing and loading data from a variety
of sources. It provides a simple API for accessing and manipulating data.

The loader coverts any data format into a single data format that can be used
within the rest of this service. This data format is a Pandas Dataset where each
row is a single set of an exercise with the following columns:

- Time: DateTime of the measurement
- Name: The name of the strength exercise (machine, activity, etc.)
- Set Order: The order of the set in the exercise
- Weight: The weight lifted on each rep of the exercise if the exercise is weight-related
- Reps: The number of reps completed in the set
- Distance: The distance covered in the exercise if the exercise is distance-related
"""

import logging
import pandas as pd

from engine.utils.logger import get_logger

class DataLoader:
    """ 
    Wodel Data Loader
    
    Initialize the data loader
    """
    def __init__(self):

        # Prepare data holder
        self.data: pd.DataFrame | None = None

        # Get logger
        self.logger = get_logger(
            name="DataLoader",
            log_file="logs/wodle.log",
            level=logging.DEBUG
        )

    def getData(self) -> pd.DataFrame | None:
        """
        Get Data

        This helper provides an easy function-based way to obtain loaded data.
        """
        return self.data
        
    def loadFromStrongCSV(self, filePath: str) -> bool:
        """
        Load Data - Strong App Format

        This helper loads CSV data from the Strong App in the standard export format.
        This format is expected to have the following column structure:
        - Date
        - Workout Name
        - Duration
        - Exercise Name
        - Set Order
        - Weight
        - Reps
        - Distance
        - Seconds
        - RPE

        Args:
            filePath: The path to the CSV file to load

        Returns:
            bool: True if the file was loaded successfully, false otherwise
        """
        self.logger.info(f"Attempting to load Strong CSV data from: {filePath}")
        try:
            # Read CSV
            df = pd.read_csv(filePath)
            
            # Map required columns
            # Source columns: Date, Exercise Name, Set Order, Weight, Reps, Distance
            # Target columns: Time, Name, Set Order, Weight, Reps, Distance
            column_mapping = {
                'Date': 'Time',
                'Exercise Name': 'Name',
                'Set Order': 'Set Order',
                'Weight': 'Weight',
                'Reps': 'Reps',
                'Distance': 'Distance'
            }
            
            # Verify if expected columns exist
            for src_col in column_mapping.keys():
                if src_col not in df.columns:
                    raise KeyError(f"Expected column '{src_col}' not found in CSV.")
            
            # Select and rename columns
            df_cleaned = df[list(column_mapping.keys())].rename(columns=column_mapping)
            
            # Convert Time to datetime
            df_cleaned['Time'] = pd.to_datetime(df_cleaned['Time'])
            
            # Store in self.data
            self.data = df_cleaned
            
            self.logger.info(f"Successfully loaded and parsed {len(df_cleaned)} rows of data.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV from {filePath}: {e}", exc_info=True)
            return False

if __name__ == "__main__":
    loader = DataLoader()
    loader.loadFromStrongCSV("data/strong_workouts.csv")
    print(loader.getData())