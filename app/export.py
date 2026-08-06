"""Export utilities for E-Reputation AI.

This module keeps file-generation logic separate from the Streamlit UI.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd


def dataframe_to_csv_bytes(
    dataframe: pd.DataFrame,
    *,
    index: bool = False,
    encoding: str = "utf-8-sig",
) -> bytes:
    """Serialize a DataFrame to CSV bytes suitable for download."""
    if dataframe is None:
        dataframe = pd.DataFrame()

    return dataframe.to_csv(
        index=index,
        encoding=encoding,
    ).encode(encoding)


def dataframe_to_excel_bytes(
    dataframe: pd.DataFrame,
    *,
    sheet_name: str = "Report",
) -> bytes:
    """Serialize a DataFrame to an Excel workbook in memory."""
    if dataframe is None:
        dataframe = pd.DataFrame()

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name=sheet_name[:31],
        )

    buffer.seek(0)
    return buffer.getvalue()


def build_analysis_exports(
    analyzed_reviews: pd.DataFrame | None = None,
    aspect_summary: pd.DataFrame | None = None,
    recommendations: pd.DataFrame | None = None,
) -> dict[str, bytes]:
    """Build the standard CSV exports produced by the application."""
    exports: dict[str, bytes] = {}

    datasets = {
        "analyzed_reviews.csv": analyzed_reviews,
        "aspect_summary.csv": aspect_summary,
        "recommendations_by_aspect.csv": recommendations,
    }

    for filename, dataframe in datasets.items():
        if dataframe is not None:
            exports[filename] = dataframe_to_csv_bytes(dataframe)

    return exports
