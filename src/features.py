"""
Feature-engineering utilities for the BVL Food Safety Inspection Analytics project.

These functions reproduce the sample-level feature transformations used in
the final data-inspection notebook. The raw BVL dataset contains repeated
analytical records, while the machine-learning dataset requires one row per
physical food sample.
"""

import pandas as pd


def create_processing_feature(sample_features_raw):
    """
    Create one processing feature per food sample.

    Multiple legitimate VerarbeitungT (Processing Description) values are
    preserved by combining the unique descriptions into one sorted category.
    Samples without processing information are represented as "Missing".
    """
    processing_feature = (
        sample_features_raw[
            ["ProbenNrPseud", "VerarbeitungT"]
        ]
        .drop_duplicates()
        .groupby("ProbenNrPseud")["VerarbeitungT"]
        .apply(
            lambda x: " | ".join(
                sorted(x.dropna().unique())
            )
        )
        .reset_index(name="Processing_Feature")
    )

    processing_feature["Processing_Feature"] = (
        processing_feature["Processing_Feature"]
        .replace("", "Missing")
    )

    return processing_feature


def create_packaging_feature(sample_features_raw, food_sample_ids):
    """
    Create one packaging feature per food sample.

    VerpackungT (Packaging Description) is represented directly at sample
    level. Genuine missing values are retained explicitly as "Missing".
    """
    packaging_feature = (
        sample_features_raw[
            ["ProbenNrPseud", "VerpackungT"]
        ]
        .drop_duplicates()
        .groupby("ProbenNrPseud")["VerpackungT"]
        .first()
        .reindex(sorted(food_sample_ids))
        .reset_index()
        .rename(
            columns={
                "VerpackungT": "Packaging_Feature"
            }
        )
    )

    packaging_feature["Packaging_Feature"] = (
        packaging_feature["Packaging_Feature"]
        .fillna("Missing")
    )

    return packaging_feature


def resolve_country(values):
    """
    Resolve multiple country-of-origin descriptions for one sample.

    If one specific country occurs together with "Ungeklärt"
    (Unresolved/Unknown), the specific country is retained.
    """
    values = list(values.dropna().unique())

    specific_values = [
        value
        for value in values
        if value != "Ungeklärt"
    ]

    if len(specific_values) == 1:
        return specific_values[0]

    if len(values) == 1:
        return values[0]

    if len(values) == 0:
        return "Missing"

    return " | ".join(sorted(values))


def create_country_feature(sample_features_raw, food_sample_ids):
    """
    Create one country-of-origin feature per food sample.
    """
    country_feature = (
        sample_features_raw[
            ["ProbenNrPseud", "HerkStaatT"]
        ]
        .drop_duplicates()
        .groupby("ProbenNrPseud")["HerkStaatT"]
        .apply(resolve_country)
        .reindex(sorted(food_sample_ids))
        .reset_index(name="Country_Feature")
    )

    return country_feature


def create_sampling_date_features(sample_dates):
    """
    Derive calendar features from Sampling_Date.

    Sampling_Month corresponds to the month in which the physical sample
    was collected.
    """
    sample_dates = sample_dates.copy()

    sample_dates["Sampling_Date"] = pd.to_datetime(
        sample_dates["Sampling_Date"]
    )

    sample_dates["Sampling_Month"] = (
        sample_dates["Sampling_Date"].dt.month
    )

    sample_dates["Sampling_Month_Name"] = (
        sample_dates["Sampling_Date"].dt.month_name()
    )

    return sample_dates


def build_sample_level_dataset(
    sample_target,
    sample_features_raw,
    country_feature,
    processing_feature,
    packaging_feature,
    sample_dates,
):
    """
    Assemble the final one-row-per-food-sample dataset.

    ProbenNrPseud is retained for traceability but should not subsequently
    be used as a predictive machine-learning feature.
    """
    base_features = (
        sample_features_raw[
            [
                "ProbenNrPseud",
                "PnMgT",
                "MatrixGrT",
                "MatrixT",
                "BetriebsartT",
            ]
        ]
        .drop_duplicates()
    )

    model_df = (
        sample_target
        .merge(
            base_features,
            on="ProbenNrPseud",
            how="left",
        )
        .merge(
            country_feature,
            on="ProbenNrPseud",
            how="left",
        )
        .merge(
            processing_feature,
            on="ProbenNrPseud",
            how="left",
        )
        .merge(
            packaging_feature,
            on="ProbenNrPseud",
            how="left",
        )
        .merge(
            sample_dates[
                [
                    "ProbenNrPseud",
                    "Sampling_Date",
                    "Sampling_Month",
                ]
            ],
            on="ProbenNrPseud",
            how="left",
        )
    )

    return model_df


def validate_sample_level_dataset(model_df):
    """
    Return basic validation statistics for the modelling dataset.
    """
    return {
        "shape": model_df.shape,
        "duplicate_sample_ids":
            int(model_df["ProbenNrPseud"].duplicated().sum()),
        "missing_values":
            int(model_df.isna().sum().sum()),
        "adverse_samples":
            int(model_df["Target_Adverse"].sum()),
        "non_adverse_samples":
            int((model_df["Target_Adverse"] == 0).sum()),
    }
