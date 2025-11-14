"""
Evaluation Metrics for DPO

This module implements metrics to evaluate DPO-trained models.
"""

import torch
from typing import Dict, List
import numpy as np


class DPOMetrics:
    """Calculate metrics for DPO model evaluation."""
    
    @staticmethod
    def compute_preference_accuracy(
        model,
        ref_model,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
        tokenizer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ) -> float:
        """
        Compute the preference accuracy: how often does the model prefer the chosen response?
        
        Args:
            model: The trained model
            ref_model: The reference model
            prompts: List of prompts
            chosen_responses: List of preferred responses
            rejected_responses: List of rejected responses
            tokenizer: Tokenizer
            device: Device to run on
            
        Returns:
            Preference accuracy (0-1)
        """
        model.eval()
        ref_model.eval()
        
        correct = 0
        total = len(prompts)
        
        with torch.no_grad():
            for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses):
                # Tokenize
                chosen_text = prompt + chosen
                rejected_text = prompt + rejected
                
                chosen_tokens = tokenizer(chosen_text, return_tensors='pt').to(device)
                rejected_tokens = tokenizer(rejected_text, return_tensors='pt').to(device)
                
                # Get log probabilities
                chosen_logprobs = model(**chosen_tokens).logits.log_softmax(dim=-1)
                rejected_logprobs = model(**rejected_tokens).logits.log_softmax(dim=-1)
                
                # Sum log probabilities for each sequence
                chosen_score = chosen_logprobs.sum().item()
                rejected_score = rejected_logprobs.sum().item()
                
                # Check if model prefers chosen over rejected
                if chosen_score > rejected_score:
                    correct += 1
        
        return correct / total
    
    @staticmethod
    def compute_kl_divergence(model_logits, ref_logits) -> float:
        """
        Compute KL divergence between model and reference model.
        
        Args:
            model_logits: Logits from the trained model
            ref_logits: Logits from the reference model
            
        Returns:
            KL divergence
        """
        model_probs = torch.softmax(model_logits, dim=-1)
        ref_probs = torch.softmax(ref_logits, dim=-1)
        
        kl_div = torch.sum(model_probs * (torch.log(model_probs) - torch.log(ref_probs)), dim=-1)
        return kl_div.mean().item()
    
    @staticmethod
    def compute_reward_margin(
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> float:
        """
        Compute the average margin between chosen and rejected rewards.
        
        Args:
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses
            
        Returns:
            Average reward margin
        """
        margins = chosen_rewards - rejected_rewards
        return margins.mean().item()
    
    @staticmethod
    def compute_perplexity(loss: float) -> float:
        """
        Compute perplexity from loss.
        
        Args:
            loss: Cross-entropy loss
            
        Returns:
            Perplexity
        """
        return np.exp(loss)
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
        """
        Aggregate metrics from multiple batches.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        keys = metrics_list[0].keys()
        
        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            
        return aggregated
