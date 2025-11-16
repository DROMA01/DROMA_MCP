"""Pydantic schemas for DROMA analysis operations."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

# Import shared enums
from .database_query import DataType
from .data_loading import MolecularType


class AnalyzeDrugOmicPairModel(BaseModel):
    """Schema for analyzing drug-omic pair associations."""
    
    dataset_name: str = Field(
        description="Dataset name (e.g., 'CCLE', 'gCSI') for DromaSet or MultiDromaSet"
    )
    feature_type: MolecularType = Field(
        description="Type of omics data to analyze (e.g., 'mRNA', 'mutation_gene', 'cnv')"
    )
    select_features: str = Field(
        description="Name of the specific omics feature (e.g., 'ABCB1', 'TP53')"
    )
    select_drugs: str = Field(
        description="Name of the drug to analyze (e.g., 'Paclitaxel', 'Gemcitabine')"
    )
    data_type: DataType = Field(
        default=DataType.ALL,
        description="Filter by data type: 'all' (default), 'CellLine', 'PDC', 'PDO', 'PDX'"
    )
    tumor_type: str = Field(
        default="all",
        description="Filter by tumor type: 'all' (default) or specific tumor types"
    )
    overlap_only: bool = Field(
        default=False,
        description="For MultiDromaSet, whether to use only overlapping samples (default: False)"
    )
    merged_enabled: bool = Field(
        default=True,
        description="Whether to create a merged dataset from all studies (default: True)"
    )
    meta_enabled: bool = Field(
        default=True,
        description="Whether to perform meta-analysis (default: True)"
    )
    zscore: bool = Field(
        default=True,
        description="""Whether to apply z-score normalization to treatment response and molecular profiles (default: True).
        If False, merged_enabled should be set to False to avoid combining non-normalized data from different studies."""
    )
    data_type_anno: Optional[str] = Field(
        default=None,
        description="Optional annotation to add in plot titles (e.g., 'Cell lines'). Will be appended as '(annotation)'"
    )
    save_plots: bool = Field(
        default=True,
        description="Whether to save plots to files (default: True)"
    )
    plot_width: float = Field(
        default=10.0,
        description="Plot width in inches (default: 10)"
    )
    plot_height: float = Field(
        default=6.0,
        description="Plot height in inches (default: 6)"
    )


__all__ = [
    "AnalyzeDrugOmicPairModel"
]

