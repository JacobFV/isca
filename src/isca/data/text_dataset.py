from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import List, Dict
import random


class HFDataset(Dataset):
    def __init__(self, dataset_name: str, config: str | None, split: str, model_name: str, max_len: int, text_column: str = "text"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.max_len = max_len
        self.dataset = load_dataset(
            dataset_name,
            config,
            split=split,
            trust_remote_code=True
        )
        self.text_column = text_column

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][self.text_column]
        if isinstance(text, list):  # Some datasets return list of strings
            text = " ".join(text)
            
        e = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in e.items()}
        item["labels"] = item["input_ids"].clone()
        return item


class MixedDataset:
    """Factory for creating mixed datasets"""
    DATASETS = {
        "wikitext": {
            "name": "wikitext",
            "split": "train",
            "text_column": "text",
        },
        "code": {
            "name": "codeparrot/github-code",
            "split": "train",
            "text_column": "content",
        },
        "papers": {
            "name": "scientific_papers",
            "split": "train",
            "text_column": "article",
        },
        "books": {
            "name": "bookcorpus",
            "split": "train",
            "text_column": "text",
        }
    }
    
    @classmethod
    def create(cls, model_name: str, max_len: int, datasets: List[Dict] = None) -> Dataset:
        if datasets is None:
            # Use all datasets with default configs if none specified
            datasets = [{"name": k} for k in cls.DATASETS.keys()]
            
        all_datasets = []
        for dataset_config in datasets:
            name = dataset_config["name"]
            if name not in cls.DATASETS:
                print(f"Unknown dataset {name}, skipping")
                continue
                
            base_config = cls.DATASETS[name]
            try:
                dataset = HFDataset(
                    dataset_name=base_config["name"],
                    config=dataset_config.get("subset", None),
                    split=base_config["split"],
                    model_name=model_name,
                    max_len=max_len,
                    text_column=base_config["text_column"]
                )
                all_datasets.append(dataset)
                print(f"Loaded dataset: {name} with {len(dataset)} samples")
            except Exception as e:
                print(f"Failed to load dataset {name}: {str(e)}")
                
        if not all_datasets:
            raise ValueError("No datasets were successfully loaded")
                
        return ConcatDataset(all_datasets)
