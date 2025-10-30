"""
MyAI Training Pipeline

Complete training/fine-tuning pipeline with:
- LoRA/QLoRA support for efficient training
- Mixed precision training
- Gradient accumulation
- Checkpoint management
- Evaluation and metrics
- WandB integration (optional)

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --lora
    python train.py --config config.yaml --qlora
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset
import click

from dataset import prepare_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """
    Load and configure model and tokenizer.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = config['model']
    lora_config = config.get('lora', {})
    qlora_config = config.get('qlora', {})
    
    logger.info(f"Loading model: {model_config['name']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        cache_dir=model_config.get('cache_dir'),
        trust_remote_code=model_config.get('trust_remote_code', True)
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization for QLoRA
    quantization_config = None
    if qlora_config.get('enabled', False):
        logger.info("Using QLoRA (4-bit quantization)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config.get('load_in_4bit', True),
            bnb_4bit_compute_dtype=getattr(torch, qlora_config.get('bnb_4bit_compute_dtype', 'float16')),
            bnb_4bit_quant_type=qlora_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=qlora_config.get('bnb_4bit_use_double_quant', True),
        )
    
    # Load model
    model_kwargs = {
        'cache_dir': model_config.get('cache_dir'),
        'trust_remote_code': model_config.get('trust_remote_code', True),
    }
    
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
        model_kwargs['device_map'] = 'auto'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name'],
        **model_kwargs
    )
    
    # Prepare model for k-bit training if using QLoRA
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    if lora_config.get('enabled', False):
        logger.info("Applying LoRA")
        peft_config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 16),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
            bias=lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def setup_training_args(config: Dict[str, Any]) -> TrainingArguments:
    """Create training arguments from config."""
    train_config = config['training']
    
    return TrainingArguments(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config.get('num_train_epochs', 3),
        per_device_train_batch_size=train_config.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=train_config.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 4),
        learning_rate=train_config.get('learning_rate', 2e-5),
        weight_decay=train_config.get('weight_decay', 0.01),
        warmup_steps=train_config.get('warmup_steps', 500),
        max_steps=train_config.get('max_steps', -1),
        fp16=train_config.get('fp16', True),
        bf16=train_config.get('bf16', False),
        gradient_checkpointing=train_config.get('gradient_checkpointing', True),
        optim=train_config.get('optim', 'adamw_torch'),
        logging_steps=train_config.get('logging_steps', 10),
        logging_dir=train_config.get('logging_dir', './logs'),
        report_to=train_config.get('report_to', ['tensorboard']),
        evaluation_strategy=train_config.get('evaluation_strategy', 'steps'),
        eval_steps=train_config.get('eval_steps', 500),
        save_strategy=train_config.get('save_strategy', 'steps'),
        save_steps=train_config.get('save_steps', 500),
        save_total_limit=train_config.get('save_total_limit', 3),
        load_best_model_at_end=train_config.get('load_best_model_at_end', True),
        metric_for_best_model=train_config.get('metric_for_best_model', 'eval_loss'),
        seed=train_config.get('seed', 42),
        dataloader_num_workers=train_config.get('dataloader_num_workers', 4),
        remove_unused_columns=train_config.get('remove_unused_columns', False),
        push_to_hub=train_config.get('push_to_hub', False),
    )


@click.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.option('--lora', is_flag=True, help='Enable LoRA')
@click.option('--qlora', is_flag=True, help='Enable QLoRA')
@click.option('--resume', default=None, help='Resume from checkpoint')
def main(config: str, lora: bool, qlora: bool, resume: Optional[str]):
    """Main training function."""
    
    # Load configuration
    logger.info(f"Loading configuration from {config}")
    cfg = load_config(config)
    
    # Override LoRA/QLoRA settings from CLI
    if lora:
        cfg['lora']['enabled'] = True
    if qlora:
        cfg['qlora']['enabled'] = True
        cfg['lora']['enabled'] = True  # QLoRA requires LoRA
    
    # Setup WandB if enabled
    if cfg.get('wandb', {}).get('enabled', False):
        import wandb
        wandb.init(
            project=cfg['wandb'].get('project', 'myai-training'),
            entity=cfg['wandb'].get('entity'),
            name=cfg['wandb'].get('name'),
            config=cfg
        )
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(cfg)
    
    # Prepare dataset
    logger.info("Preparing dataset")
    train_dataset, eval_dataset = prepare_dataset(cfg, tokenizer)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = setup_training_args(cfg)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training")
    
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()
    
    # Save final model
    logger.info("Saving final model")
    trainer.save_model(os.path.join(cfg['training']['output_dir'], 'final'))
    tokenizer.save_pretrained(os.path.join(cfg['training']['output_dir'], 'final'))
    
    # Evaluate
    logger.info("Running final evaluation")
    metrics = trainer.evaluate()
    logger.info(f"Final metrics: {metrics}")
    
    # Save metrics
    with open(os.path.join(cfg['training']['output_dir'], 'metrics.yaml'), 'w') as f:
        yaml.dump(metrics, f)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
