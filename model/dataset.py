"""
Dataset preparation and processing for training.

Supports:
- HuggingFace datasets
- Custom JSONL files
- Text preprocessing and tokenization
- Data augmentation (optional)
"""

import logging
from typing import Dict, Any, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict
import torch

logger = logging.getLogger(__name__)


def load_data(config: Dict[str, Any]) -> DatasetDict:
    """
    Load dataset from HuggingFace or local files.
    
    Args:
        config: Dataset configuration
        
    Returns:
        DatasetDict with train/eval/test splits
    """
    dataset_config = config['dataset']
    
    # Load from HuggingFace
    if 'name' in dataset_config:
        logger.info(f"Loading dataset: {dataset_config['name']}")
        
        dataset = load_dataset(
            dataset_config['name'],
            dataset_config.get('config'),
            cache_dir=config['model'].get('cache_dir')
        )
        
        # Rename splits if needed
        if 'train_split' in dataset_config:
            train_split = dataset_config['train_split']
            eval_split = dataset_config.get('eval_split', 'validation')
            test_split = dataset_config.get('test_split', 'test')
            
            dataset_dict = DatasetDict({
                'train': dataset[train_split],
                'validation': dataset[eval_split] if eval_split in dataset else None,
                'test': dataset[test_split] if test_split in dataset else None,
            })
            
            # Remove None values
            dataset_dict = DatasetDict({k: v for k, v in dataset_dict.items() if v is not None})
            
            return dataset_dict
        
        return dataset
    
    # Load from local files
    elif 'train_file' in dataset_config:
        logger.info("Loading dataset from local files")
        
        data_files = {}
        if 'train_file' in dataset_config:
            data_files['train'] = dataset_config['train_file']
        if 'eval_file' in dataset_config:
            data_files['validation'] = dataset_config['eval_file']
        if 'test_file' in dataset_config:
            data_files['test'] = dataset_config['test_file']
        
        # Determine file type
        file_extension = dataset_config['train_file'].split('.')[-1]
        
        if file_extension == 'jsonl' or file_extension == 'json':
            dataset = load_dataset('json', data_files=data_files)
        elif file_extension == 'csv':
            dataset = load_dataset('csv', data_files=data_files)
        elif file_extension == 'txt':
            dataset = load_dataset('text', data_files=data_files)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return dataset
    
    else:
        raise ValueError("Must specify either 'name' or 'train_file' in dataset config")


def preprocess_function(examples: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    """
    Preprocess examples for causal language modeling.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    # Get text field (try common field names)
    text_field = None
    for field in ['text', 'content', 'document', 'article']:
        if field in examples:
            text_field = field
            break
    
    if text_field is None:
        raise ValueError(f"Could not find text field in examples. Available fields: {examples.keys()}")
    
    # Tokenize
    tokenized = tokenizer(
        examples[text_field],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None,
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    
    return tokenized


def prepare_dataset(
    config: Dict[str, Any],
    tokenizer
) -> Tuple[Dataset, Dataset]:
    """
    Prepare train and eval datasets.
    
    Args:
        config: Full configuration
        tokenizer: Tokenizer to use
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    dataset_config = config['dataset']
    
    # Load raw dataset
    raw_datasets = load_data(config)
    
    logger.info(f"Raw dataset: {raw_datasets}")
    
    # Tokenize datasets
    max_length = dataset_config.get('max_length', 512)
    num_workers = dataset_config.get('preprocessing_num_workers', 4)
    overwrite_cache = dataset_config.get('overwrite_cache', False)
    
    logger.info("Tokenizing datasets")
    
    tokenized_datasets = raw_datasets.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        num_proc=num_workers,
        remove_columns=raw_datasets['train'].column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing dataset",
    )
    
    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets.get('validation', tokenized_datasets.get('test'))
    
    if eval_dataset is None:
        logger.warning("No validation set found, splitting train set")
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
    
    return train_dataset, eval_dataset


def create_custom_dataset(
    texts: list,
    tokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Create a dataset from a list of texts.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset
    """
    # Create dataset from texts
    dataset = Dataset.from_dict({'text': texts})
    
    # Tokenize
    tokenized = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing texts",
    )
    
    return tokenized


# Example usage for custom data format
def load_instruction_dataset(file_path: str) -> Dataset:
    """
    Load instruction-following dataset in format:
    {"instruction": "...", "input": "...", "output": "..."}
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Dataset with formatted prompts
    """
    import json
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            # Format as instruction-following prompt
            if item.get('input'):
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            
            data.append({'text': prompt})
    
    return Dataset.from_dict({'text': [d['text'] for d in data]})


# Data augmentation functions (optional)
def augment_text(text: str, augmentation_type: str = 'none') -> str:
    """
    Apply data augmentation to text.
    
    Args:
        text: Input text
        augmentation_type: Type of augmentation (none, synonym, backtranslation)
        
    Returns:
        Augmented text
    """
    if augmentation_type == 'none':
        return text
    
    # Add augmentation implementations here
    # For example: synonym replacement, back-translation, etc.
    
    return text
