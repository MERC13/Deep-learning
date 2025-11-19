"""
DPO Trainer

This module implements the Direct Preference Optimization training logic.
"""

from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from typing import Dict
import os


class DPOTrainerWrapper:
    """Wrapper for DPO training with custom configuration."""
    
    def __init__(self, model, ref_model, tokenizer, train_dataset, eval_dataset, config: Dict):
        """
        Initialize the DPO trainer.
        
        Args:
            model: Model to train
            ref_model: Reference model (frozen)
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            config: Configuration dictionary
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        
        self.training_config = config.get('training', {})
        self.output_config = config.get('output', {})
        self.wandb_config = config.get('wandb', {})
        
        self.trainer = self._create_trainer()
    
    def _create_trainer(self) -> DPOTrainer:
        """Create and configure the DPO trainer."""
        
        # Configure training arguments
        training_args = DPOConfig(
            # Output
            output_dir=self.output_config.get('output_dir', 'outputs/dpo_model'),
            logging_dir=self.output_config.get('logging_dir', 'outputs/logs'),
            
            # Training parameters
            learning_rate=self.training_config.get('learning_rate', 5e-5),
            num_train_epochs=self.training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=self.training_config.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=self.training_config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 4),
            warmup_steps=self.training_config.get('warmup_steps', 100),
            
            # Logging and evaluation
            logging_steps=self.training_config.get('logging_steps', 10),
            eval_strategy="steps",
            eval_steps=self.training_config.get('eval_steps', 500),
            save_steps=self.training_config.get('save_steps', 1000),
            save_total_limit=self.training_config.get('save_total_limit', 3),
            
            # Mixed precision
            fp16=self.training_config.get('fp16', False),
            bf16=self.training_config.get('bf16', False),
            
            # DPO specific
            beta=self.training_config.get('beta', 0.1),  # Temperature parameter
            max_length=self.config.get('dataset', {}).get('max_length', 512),
            max_prompt_length=self.config.get('dataset', {}).get('max_prompt_length', 256),
            
            # W&B
            report_to="wandb" if self.wandb_config.get('enabled', False) else "none",
            run_name=self.wandb_config.get('run_name', 'dpo-training'),
            
            # Other
            remove_unused_columns=False,
            gradient_checkpointing=self.training_config.get('gradient_checkpointing', False),
        )
        
        # Initialize W&B if enabled
        if self.wandb_config.get('enabled', False):
            import wandb
            wandb.init(
                project=self.wandb_config.get('project', 'dpo-alignment'),
                name=self.wandb_config.get('run_name', 'dpo-training'),
                config=self.config
            )
        
        # Create DPO trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )
        
        return trainer
    
    def train(self):
        """Start the training process."""
        print("\n" + "="*80)
        print("Starting DPO Training")
        print("="*80)
        
        # Train the model
        self.trainer.train()
        
        print("\n" + "="*80)
        print("Training completed!")
        print("="*80)
        
        return self.trainer
    
    def save_model(self, output_dir: str = None):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model (uses config if not provided)
        """
        if output_dir is None:
            output_dir = self.output_config.get('output_dir', 'outputs/dpo_model')
        
        print(f"\nSaving model to {output_dir}...")
        
        # Save the model and tokenizer
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("✓ Model saved successfully")
    
    def evaluate(self):
        """Evaluate the model on the evaluation dataset."""
        print("\nEvaluating model...")
        
        metrics = self.trainer.evaluate()
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return metrics
