"""
Project configuration for the Food Safety Inspection Analytics project.

Centralizes the main dataset paths, modelling constants, target variable,
and predictor columns used throughout the project.
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = DATA_DIR / "20250108_opendata_moni_23.csv"

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / "food_safety_sample_level.csv"
)

TABLEAU_DATA_PATH = (
    PROCESSED_DATA_DIR / "tableau_food_safety_dashboard.csv"
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

CSV_SEPARATOR = ";"
CSV_ENCODING = "latin1"


# ---------------------------------------------------------------------
# Modelling configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_SPLITS = 5

TARGET_COLUMN = "Target_Adverse"

FINAL_CLASSIFICATION_THRESHOLD = 0.40


# ---------------------------------------------------------------------
# Final modelling predictors
# ---------------------------------------------------------------------

FEATURE_COLUMNS = [
    "PnMgT",
    "MatrixGrT",
    "MatrixT",
    "BetriebsartT",
    "Country_Feature",
    "Processing_Feature",
    "Packaging_Feature",
    "Sampling_Month",
]
