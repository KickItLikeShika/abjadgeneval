from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class CustomDataset(Dataset):
    def __init__(self, tokenizer_name, texts, labels=None, max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        input_dict = {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
        }
        # token_type_ids not used by mdeberta
        if 'token_type_ids' in encoded:
            input_dict['token_type_ids'] = torch.tensor(encoded['token_type_ids'], dtype=torch.long)

        if self.labels is not None:
            input_dict['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return input_dict


def apply_dynamic_padding(inputs):
    """Truncate the text to the max length in the batch to make training faster"""
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        if k != 'labels' and v.dim() == 2:  # skip 1D tensors like labels
            inputs[k] = inputs[k][:, :mask_len]
    return inputs


def collate_fn(batch):
    """Collate with dynamic padding"""
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    if 'token_type_ids' in batch[0]:
        inputs['token_type_ids'] = torch.stack([b['token_type_ids'] for b in batch])

    if 'labels' in batch[0]:
        inputs['labels'] = torch.stack([b['labels'] for b in batch])

    # apply dynamic padding
    inputs = apply_dynamic_padding(inputs)
    return inputs
