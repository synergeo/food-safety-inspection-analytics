"""
Data cleaning utilities for the BVL Food Safety Inspection Analytics project.

These functions reproduce the main scope and target-preparation operations
used in the final data-inspection notebook.

The project uses a conservative definition of an adverse food-safety outcome.
The raw BVL analytical-result data are filtered to a strict food-product
scope before constructing the sample-level binary target.
"""

import pandas as pd


# Assessment descriptions classified as clearly adverse in the project.
ADVERSE_ASSESSMENTS = {
    "Beanstandet",
    "Standard nicht erfüllt",
    "> Höchstmenge (führt zur Beanstandung)",
}


# Product groups excluded from the strict food-product scope.
NON_FOOD_GROUPS = {
    "Kosmetische Mittel und Stoffe zu deren Herstellung",
    "Spielwaren und Scherzartikel",
    "Bedarfsgegenstände mit Körperkontakt und zur Körperpflege",
    "Bedarfsgegenstände mit Lebensmittelkontakt (BgLm)",
}


def identify_adverse_samples(df):
    """
    Identify samples containing at least one clearly adverse assessment.

    Parameters
    ----------
    df : pandas.DataFrame
        Analytical-level BVL data containing ProbenNrPseud and BewertungT.

    Returns
    -------
    set
        Pseudonymised sample IDs with at least one clearly adverse result.
    """
    required = {"ProbenNrPseud", "BewertungT"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    adverse_mask = df["BewertungT"].isin(ADVERSE_ASSESSMENTS)

    return set(
        df.loc[
            adverse_mask,
            "ProbenNrPseud"
        ].dropna().unique()
    )


def create_sample_group_table(df):
    """
    Create one unique Sample + Matrix Group table.

    Parameters
    ----------
    df : pandas.DataFrame
        BVL data containing ProbenNrPseud and MatrixGrT.

    Returns
    -------
    pandas.DataFrame
        Unique sample and matrix-group combinations.
    """
    required = {"ProbenNrPseud", "MatrixGrT"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    return (
        df[
            ["ProbenNrPseud", "MatrixGrT"]
        ]
        .drop_duplicates()
        .copy()
    )


def filter_food_scope(sample_groups):
    """
    Restrict the project to the strict food-product scope.

    The same four clearly non-food matrix groups excluded in the final
    notebook are removed here.

    Parameters
    ----------
    sample_groups : pandas.DataFrame
        Unique Sample + Matrix Group table.

    Returns
    -------
    pandas.DataFrame
        Sample-group table restricted to food products.
    """
    return sample_groups[
        ~sample_groups["MatrixGrT"].isin(NON_FOOD_GROUPS)
    ].copy()


def get_food_sample_ids(food_scope):
    """
    Return the unique sample IDs belonging to the food-product scope.
    """
    return set(
        food_scope["ProbenNrPseud"]
        .dropna()
        .unique()
    )


def create_sample_target(food_sample_ids, adverse_samples):
    """
    Construct the binary sample-level adverse target.

    Target_Adverse = 1:
        The food sample contains at least one clearly adverse assessment.

    Target_Adverse = 0:
        No clearly adverse assessment was identified for the food sample.

    Parameters
    ----------
    food_sample_ids : iterable
        Sample IDs included in the strict food-product scope.

    adverse_samples : iterable
        Sample IDs containing at least one clearly adverse assessment.

    Returns
    -------
    pandas.DataFrame
        One row per food sample with Target_Adverse.
    """
    food_sample_ids = set(food_sample_ids)
    adverse_samples = set(adverse_samples)

    food_adverse_samples = (
        food_sample_ids.intersection(adverse_samples)
    )

    sample_target = pd.DataFrame({
        "ProbenNrPseud": sorted(food_sample_ids)
    })

    sample_target["Target_Adverse"] = (
        sample_target["ProbenNrPseud"]
        .isin(food_adverse_samples)
        .astype(int)
    )

    return sample_target


def validate_sample_target(sample_target):
    """
    Perform basic quality checks on the sample-level target dataset.

    Returns
    -------
    dict
        Number of rows, duplicate sample IDs, missing values,
        adverse samples and non-adverse samples.
    """
    return {
        "rows": len(sample_target),
        "duplicate_sample_ids":
            sample_target["ProbenNrPseud"].duplicated().sum(),
        "missing_values":
            int(sample_target.isna().sum().sum()),
        "adverse_samples":
            int(sample_target["Target_Adverse"].sum()),
        "non_adverse_samples":
            int((sample_target["Target_Adverse"] == 0).sum()),
    }
