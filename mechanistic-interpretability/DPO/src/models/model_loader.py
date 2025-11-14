"""
Model Loader for DPO Training

This module handles loading and configuring base models for DPO training.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from typing import Dict, Tuple


class ModelLoader:
    """Load and configure models for DPO training."""
    
    def __init__(self, config: Dict):
        """
        Initialize the model loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.lora_config = config.get('lora', {})
        
    def load_model_and_tokenizer(self) -> Tuple:
        """
        Load the base model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer, reference_model)
        """
        model_name = self.model_config.get('name', 'gpt2')
        print(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = self._load_tokenizer(model_name)
        
        # Configure quantization if needed
        quantization_config = self._get_quantization_config()
        
        # Load model
        model = self._load_model(model_name, quantization_config)
        
        # Apply LoRA if specified
        if self.model_config.get('use_peft', False):
            model = self._apply_lora(model)
        
        # Load reference model (frozen copy for DPO)
        ref_model = self._load_model(model_name, quantization_config)
        ref_model.eval()  # Set to evaluation mode
        
        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False
        
        print(f"✓ Model loaded successfully")
        print(f"  - Model: {model_name}")
        print(f"  - Trainable parameters: {self._count_parameters(model):,}")
        print(f"  - Using LoRA: {self.model_config.get('use_peft', False)}")
        
        return model, tokenizer, ref_model
    
    def _load_tokenizer(self, model_name: str):
        """Load tokenizer and configure special tokens."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure we have a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def _get_quantization_config(self):
        """Get quantization configuration if specified."""
        if self.model_config.get('load_in_4bit', False):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.model_config.get('load_in_8bit', False):
            return BitsAndBytesConfig(load_in_8bit=True)
        
        return None
    
    def _load_model(self, model_name: str, quantization_config):
        """Load the language model."""
        model_kwargs = {
            'pretrained_model_name_or_path': model_name,
            'torch_dtype': torch.float16 if self.config.get('training', {}).get('fp16', False) else torch.float32,
        }
        
        if quantization_config:
            model_kwargs['quantization_config'] = quantization_config
            model_kwargs['device_map'] = 'auto'
        
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Enable gradient checkpointing if specified
        if self.config.get('training', {}).get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
        
        return model
    
    def _apply_lora(self, model):
        """Apply LoRA (Low-Rank Adaptation) to the model."""
        print("Applying LoRA configuration...")
        
        # Prepare model for k-bit training if using quantization
        if self.model_config.get('load_in_4bit') or self.model_config.get('load_in_8bit'):
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_config.get('r', 16),
            lora_alpha=self.lora_config.get('lora_alpha', 32),
            lora_dropout=self.lora_config.get('lora_dropout', 0.05),
            target_modules=self.lora_config.get('target_modules', ['c_attn']),
            bias=self.lora_config.get('bias', 'none'),
            task_type=self.lora_config.get('task_type', 'CAUSAL_LM')
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _count_parameters(self, model) -> int:
        """Count trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
