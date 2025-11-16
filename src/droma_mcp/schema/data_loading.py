"""Pydantic schemas for DROMA data loading operations."""

from pydantic import BaseModel, Field, field_validator
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
    
    @field_validator('dataset_name')
    @classmethod
    def normalize_dataset_name(cls, v: str) -> str:
        """Normalize dataset_name by removing spaces around commas."""
        return ','.join([part.strip() for part in v.split(',')])


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
    
    @field_validator('dataset_name')
    @classmethod
    def normalize_dataset_name(cls, v: str) -> str:
        """Normalize dataset_name by removing spaces around commas."""
        return ','.join([part.strip() for part in v.split(',')])


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


class ViewCachedDataModel(BaseModel):
    """Schema for viewing cached data with preview."""
    
    cache_key: str = Field(
        description="The cache key to retrieve data"
    )
    preview_size: int = Field(
        default=5,
        description="Number of rows/columns to preview (default: 5, max: 10)"
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific features/rows to view"
    )
    samples: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific samples/columns to view"
    )


class ExportCachedDataModel(BaseModel):
    """Schema for exporting cached data to file."""
    
    cache_key: str = Field(
        description="Cache key of the dataset to export"
    )
    file_format: str = Field(
        default="csv",
        description="Export file format (e.g., 'csv', 'excel', 'json')"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Custom filename for export (auto-generated if None)"
    )
    release_memory: bool = Field(
        default=True,
        description="Whether to release memory after export (recommended for large datasets)"
    )


class ViewExportedDataModel(BaseModel):
    """Schema for viewing exported data files."""
    
    export_id: str = Field(
        description="Export ID returned from export_cached_data"
    )
    full_data: bool = Field(
        default=True,
        description="Load full data (True, default) or preview only (False)"
    )
    preview_size: int = Field(
        default=10,
        description="Number of rows/columns to preview when full_data=False (default: 10, max: 50)"
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific features/rows to view"
    )
    samples: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific samples/columns to view"
    ) 