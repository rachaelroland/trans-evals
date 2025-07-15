import json
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datasets import load_dataset as hf_load_dataset

from .base import BaseDataset, DatasetExample, BiasType, EvaluationType
from .tango import TANGODataset
from .holistic_bias import HolisticBiasDataset
from .crows_pairs import CrowsPairsDataset
from .bbq import BBQDataset

logger = logging.getLogger(__name__)


def load_tango(data_path: Optional[str] = None) -> TANGODataset:
    """Load the TANGO dataset."""
    dataset = TANGODataset()
    dataset.load(data_path)
    return dataset


def load_holistic_bias(data_path: Optional[str] = None, identity_terms: Optional[List[str]] = None) -> HolisticBiasDataset:
    """Load the HolisticBias dataset, optionally filtering by identity terms."""
    dataset = HolisticBiasDataset(identity_terms=identity_terms)
    dataset.load(data_path)
    return dataset


def load_crows_pairs(data_path: Optional[str] = None) -> CrowsPairsDataset:
    """Load the CrowS-Pairs dataset."""
    dataset = CrowsPairsDataset()
    dataset.load(data_path)
    return dataset


def load_bbq(data_path: Optional[str] = None) -> BBQDataset:
    """Load the BBQ dataset."""
    dataset = BBQDataset()
    dataset.load(data_path)
    return dataset


def load_dataset(dataset_name: str, **kwargs) -> BaseDataset:
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
        **kwargs: Additional arguments to pass to the dataset loader
        
    Returns:
        Loaded dataset
    """
    dataset_loaders = {
        "tango": load_tango,
        "holistic_bias": load_holistic_bias,
        "crows_pairs": load_crows_pairs,
        "bbq": load_bbq
    }
    
    if dataset_name.lower() not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(dataset_loaders.keys())}")
    
    return dataset_loaders[dataset_name.lower()](**kwargs)


def merge_datasets(datasets: List[BaseDataset]) -> BaseDataset:
    """Merge multiple datasets into one."""
    class MergedDataset(BaseDataset):
        def load(self):
            pass
    
    merged = MergedDataset(name="merged")
    for dataset in datasets:
        merged.examples.extend(dataset.examples)
    
    return merged