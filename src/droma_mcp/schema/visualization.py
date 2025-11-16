"""Pydantic schemas for DROMA visualization operations."""

from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum

# Import shared enums
from .database_query import DataType


class PlotDrugSensitivityRankModel(BaseModel):
    """Schema for plotting drug sensitivity rankings."""
    
    dataset_name: str = Field(
        description="Dataset name (e.g., 'CCLE', 'gCSI') for DromaSet or MultiDromaSet"
    )
    select_drugs: str = Field(
        description="Drug name to plot"
    )
    data_type: DataType = Field(
        default=DataType.ALL,
        description="Filter by data type: 'all' (default), 'CellLine', 'PDC', 'PDO', 'PDX'"
    )
    tumor_type: str = Field(
        default="all",
        description="Filter by tumor type: 'all' (default) or specific tumor type"
    )
    overlap_only: bool = Field(
        default=False,
        description="For MultiDromaSet, whether to use only overlapping samples (default: False)"
    )
    highlight: Optional[Union[int, str, List[str]]] = Field(
        default=None,
        description="""Samples to highlight. Can be:
        - Numeric value (e.g., 10) to highlight top N samples by rank
        - String from data_type (e.g., 'CellLine') to highlight all samples of that type
        - String from tumor_type (e.g., 'breast cancer') to highlight all samples of that tumor type
        - List of specific sample IDs to highlight
        Note: If more than 20 samples are highlighted, only the top 20 will be labeled"""
    )
    color: Optional[str] = Field(
        default=None,
        description="Variable to use for coloring points. Options: NULL (default), 'data_type', 'tumor_type', or any column name in sample annotations"
    )
    zscore: bool = Field(
        default=False,
        description="Whether to use z-score normalized values (default: False)"
    )
    merge: bool = Field(
        default=False,
        description="""Only applicable when zscore=TRUE and using MultiDromaSet.
        If TRUE, merges data from multiple projects when drug appears in at least 2 projects.
        If FALSE, returns separate plots for each project (default: False)"""
    )
    point_size: float = Field(
        default=2.0,
        description="Size of points in the plot (default: 2)"
    )
    highlight_alpha: float = Field(
        default=0.6,
        description="Alpha transparency for non-highlighted points (default: 0.6)"
    )
    sample_annotations: Optional[str] = Field(
        default=None,
        description="Optional dataframe identifier containing sample annotations"
    )
    db_path: Optional[str] = Field(
        default=None,
        description="Optional path to SQLite database for loading sample annotations"
    )
    output_file: Optional[str] = Field(
        default=None,
        description="Optional output file path for saving the plot (e.g., 'plot.png', 'plot.pdf')"
    )
    width: float = Field(
        default=10.0,
        description="Plot width in inches (default: 10)"
    )
    height: float = Field(
        default=6.0,
        description="Plot height in inches (default: 6)"
    )


__all__ = [
    "PlotDrugSensitivityRankModel"
]

