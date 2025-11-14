"""
Model Evaluator for DPO

This module handles comprehensive evaluation of DPO-trained models.
"""

from typing import Dict, List
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from .metrics import DPOMetrics


class ModelEvaluator:
    """Evaluate DPO-trained models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained model to evaluate
            ref_model: Reference model
            tokenizer: Tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config
        self.eval_config = config.get('evaluation', {})
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.ref_model.to(self.device)
        
        self.metrics_calculator = DPOMetrics()
    
    def evaluate(self, eval_dataset: Dataset) -> Dict:
        """
        Run comprehensive evaluation on the dataset.
        
        Args:
            eval_dataset: Dataset to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*80)
        print("Running Evaluation")
        print("="*80)
        
        # Limit samples if specified
        max_samples = self.eval_config.get('max_samples', len(eval_dataset))
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
        
        print(f"Evaluating on {len(eval_dataset)} samples...")
        
        # Extract data
        prompts = [ex['prompt'] for ex in eval_dataset]
        chosen_responses = [ex['chosen'] for ex in eval_dataset]
        rejected_responses = [ex['rejected'] for ex in eval_dataset]
        
        # Compute preference accuracy
        print("\nComputing preference accuracy...")
        preference_accuracy = self.metrics_calculator.compute_preference_accuracy(
            self.model,
            self.ref_model,
            prompts,
            chosen_responses,
            rejected_responses,
            self.tokenizer,
            self.device
        )
        
        results = {
            'preference_accuracy': preference_accuracy,
            'num_samples': len(eval_dataset)
        }
        
        # Generate samples if specified
        if self.eval_config.get('generate_samples', False):
            print("\nGenerating sample responses...")
            samples = self._generate_samples(eval_dataset)
            results['samples'] = samples
        
        # Print results
        print("\n" + "="*80)
        print("Evaluation Results")
        print("="*80)
        print(f"Preference Accuracy: {preference_accuracy:.2%}")
        print(f"Samples Evaluated: {len(eval_dataset)}")
        
        return results
    
    def _generate_samples(self, dataset: Dataset) -> List[Dict]:
        """
        Generate sample responses for qualitative evaluation.
        
        Args:
            dataset: Dataset to sample from
            
        Returns:
            List of generated samples
        """
        num_samples = self.eval_config.get('num_samples_to_generate', 5)
        samples = []
        
        self.model.eval()
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                example = dataset[i]
                prompt = example['prompt']
                
                # Tokenize prompt
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove prompt from generated text
                response = generated_text[len(prompt):].strip()
                
                samples.append({
                    'prompt': prompt,
                    'generated_response': response,
                    'chosen_response': example['chosen'],
                    'rejected_response': example['rejected']
                })
                
                # Print sample
                print(f"\n--- Sample {i+1} ---")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Generated: {response[:150]}...")
                print(f"Preferred: {example['chosen'][:150]}...")
        
        return samples
    
    def compare_models(self, prompts: List[str]) -> Dict:
        """
        Compare trained model vs reference model on given prompts.
        
        Args:
            prompts: List of prompts to compare on
            
        Returns:
            Comparison results
        """
        results = {
            'trained_model_responses': [],
            'reference_model_responses': []
        }
        
        for prompt in prompts:
            # Get response from trained model
            trained_response = self._generate_response(self.model, prompt)
            results['trained_model_responses'].append(trained_response)
            
            # Get response from reference model
            ref_response = self._generate_response(self.ref_model, prompt)
            results['reference_model_responses'].append(ref_response)
        
        return results
    
    def _generate_response(self, model, prompt: str) -> str:
        """Generate a response from a model."""
        model.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
        return response
