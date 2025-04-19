from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, file_path, model_name, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        # Set up padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_len = max_len
        self.lines = Path(file_path).read_text().splitlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        e = self.tokenizer(
            self.lines[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in e.items()}
        item["labels"] = item["input_ids"].clone()
        return item
