"""Pydantic schemas for DROMA data loading operations."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

# Import shared enums
from .database_query import DataType


class MolecularType(str, Enum):
    """Supported molecular data types."""
    MRNA = "mRNA"
    CNV = "cnv"
    METH = "meth"
    PROTEIN_RPPA = "proteinrppa"
    PROTEIN_MS = "proteinms"
    MUTATION_GENE = "mutation_gene"
    MUTATION_SITE = "mutation_site"
    FUSION = "fusion"


class LoadMolecularProfilesModel(BaseModel):
    """Schema for loading molecular profiles with z-score normalization."""
    
    dataset_name: str = Field(
        description="Dataset name (e.g., 'CCLE', 'gCSI')"
    )
    feature_type: MolecularType = Field(
        description="Type of molecular data to load"
    )
    select_features: Optional[List[str]] = Field(
        default=None,
        description="Specific features to load. If None, loads all features"
    )
    samples: Optional[List[str]] = Field(
        default=None,
        description="Specific samples to load. If None, loads all samples"
    )
    data_type: DataType = Field(
        default=DataType.ALL,
        description="Filter by data type"
    )
    tumor_type: str = Field(
        default="all",
        description="Filter by tumor type ('all' or specific tumor types)"
    )
    zscore: bool = Field(
        default=False,
        description="Whether to apply z-score normalization"
    )
    format: str = Field(
        default="long",
        description="Data format: 'long' or 'wide'"
    )


class LoadTreatmentResponseModel(BaseModel):
    """Schema for loading treatment response data with z-score normalization."""
    
    dataset_name: str = Field(
        description="Dataset name (e.g., 'CCLE', 'gCSI')"
    )
    select_drugs: Optional[List[str]] = Field(
        default=None,
        description="Specific drugs to load. If None, loads all drugs"
    )
    samples: Optional[List[str]] = Field(
        default=None,
        description="Specific samples to load. If None, loads all samples"
    )
    data_type: DataType = Field(
        default=DataType.ALL,
        description="Filter by data type"
    )
    tumor_type: str = Field(
        default="all",
        description="Filter by tumor type ('all' or specific tumor types)"
    )
    zscore: bool = Field(
        default=False,
        description="Whether to apply z-score normalization"
    )


class MultiProjectMolecularProfilesModel(BaseModel):
    """Schema for loading multi-project molecular profiles with z-score normalization."""
    
    multidromaset_id: str = Field(
        description="MultiDromaSet object identifier"
    )
    feature_type: MolecularType = Field(
        description="Type of molecular data to load"
    )
    select_features: Optional[List[str]] = Field(
        default=None,
        description="Specific features to load"
    )
    overlap_only: bool = Field(
        default=False,
        description="Whether to use only overlapping samples"
    )
    data_type: DataType = Field(
        default=DataType.ALL,
        description="Filter by data type"
    )
    tumor_type: str = Field(
        default="all",
        description="Filter by tumor type"
    )
    zscore: bool = Field(
        default=False,
        description="Whether to apply z-score normalization"
    )
    format: str = Field(
        default="long",
        description="Data format: 'long' or 'wide'"
    )


class MultiProjectTreatmentResponseModel(BaseModel):
    """Schema for loading multi-project treatment response data with z-score normalization."""
    
    multidromaset_id: str = Field(
        description="MultiDromaSet object identifier"
    )
    select_drugs: Optional[List[str]] = Field(
        default=None,
        description="Specific drugs to load"
    )
    overlap_only: bool = Field(
        default=False,
        description="Whether to use only overlapping samples"
    )
    data_type: DataType = Field(
        default=DataType.ALL,
        description="Filter by data type"
    )
    tumor_type: str = Field(
        default="all",
        description="Filter by tumor type"
    )
    zscore: bool = Field(
        default=False,
        description="Whether to apply z-score normalization"
    )


class ZscoreNormalizationModel(BaseModel):
    """Schema for z-score normalization operations."""
    
    data_id: str = Field(
        description="Identifier for the data to normalize"
    )
    check_type: bool = Field(
        default=True,
        description="Whether to check if data appears to be continuous"
    )


class DataValidationModel(BaseModel):
    """Schema for data validation responses."""
    
    is_normalized: bool = Field(
        description="Whether data has been z-score normalized"
    )
    data_shape: tuple = Field(
        description="Shape of the data matrix (rows, columns)"
    )
    feature_count: int = Field(
        description="Number of features in the dataset"
    )
    sample_count: int = Field(
        description="Number of samples in the dataset"
    )
    data_type: str = Field(
        description="Type of data (continuous, discrete, etc.)"
    )
    normalization_applicable: bool = Field(
        description="Whether z-score normalization is applicable to this data type"
    )


class BatchLoadModel(BaseModel):
    """Schema for batch loading operations."""
    
    dromaset_ids: List[str] = Field(
        description="List of DromaSet identifiers to load"
    )
    molecular_types: List[MolecularType] = Field(
        description="Molecular data types to load"
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="Specific features to load across all datasets"
    )
    normalize_separately: bool = Field(
        default=True,
        description="Whether to normalize each dataset separately"
    )
    merge_strategy: Literal["intersect", "union", "separate"] = Field(
        default="separate",
        description="How to handle features across datasets"
    )


class CheckZScoreNormalizationModel(BaseModel):
    """Schema for checking z-score normalization status."""
    
    cache_key: str = Field(
        description="Cache key of the dataset to check"
    )


class GetCachedDataInfoModel(BaseModel):
    """Schema for getting cached data information."""
    
    cache_key: Optional[str] = Field(
        default=None,
        description="Specific cache key to get info for. If None, returns info for all cached data"
    )
    include_summary_stats: bool = Field(
        default=True,
        description="Whether to include summary statistics"
    )


class ExportCachedDataModel(BaseModel):
    """Schema for exporting cached data."""
    
    cache_key: str = Field(
        description="Cache key of the dataset to export"
    )
    format: Literal["csv", "excel", "json"] = Field(
        default="csv",
        description="Export format"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Custom filename for export (auto-generated if None)"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in the export"
    ) 