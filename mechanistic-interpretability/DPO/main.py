"""
Direct Preference Optimization (DPO) Training
Main entry point for DPO training to address AI misalignment.

This script orchestrates the complete DPO training pipeline:
1. Load configuration
2. Load and prepare dataset
3. Load model and tokenizer
4. Train using DPO
5. Evaluate the trained model
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset_loader import PreferenceDatasetLoader
from src.data.data_processor import DPODataProcessor
from src.models.model_loader import ModelLoader
from src.models.dpo_trainer import DPOTrainerWrapper
from src.evaluation.evaluator import ModelEvaluator
from src.utils.helpers import load_config, set_seed, print_header, save_config
from src.utils.logger import setup_logger, log_experiment_info


def main(config_path: str):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    print_header("Loading Configuration")
    config = load_config(config_path)
    print(f"✓ Configuration loaded from {config_path}")
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Setup logger
    logger = setup_logger(
        log_file=f"{config.get('output', {}).get('logging_dir', 'outputs/logs')}/training.log"
    )
    log_experiment_info(logger, config)
    
    # Save config to output directory
    output_dir = config.get('output', {}).get('output_dir', 'outputs/dpo_model')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_config(config, f"{output_dir}/config.yaml")
    
    # Step 1: Load Dataset
    print_header("Step 1: Loading Dataset")
    dataset_loader = PreferenceDatasetLoader(config)
    datasets = dataset_loader.load_dataset()
    
    train_dataset = datasets['train']
    eval_dataset = datasets['test']
    
    dataset_loader.validate_dataset(train_dataset)
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Evaluation samples: {len(eval_dataset)}")
    
    # Show a sample
    print("\n--- Sample from Dataset ---")
    sample = dataset_loader.get_sample(train_dataset, 0)
    print(f"Prompt: {sample['prompt'][:200]}...")
    print(f"Chosen: {sample['chosen'][:200]}...")
    print(f"Rejected: {sample['rejected'][:200]}...")
    
    # Step 2: Load Model and Tokenizer
    print_header("Step 2: Loading Model and Tokenizer")
    model_loader = ModelLoader(config)
    model, tokenizer, ref_model = model_loader.load_model_and_tokenizer()
    
    # Step 3: Process Dataset
    print_header("Step 3: Processing Dataset")
    data_processor = DPODataProcessor(tokenizer, config)
    
    train_dataset = data_processor.process_dataset(train_dataset)
    eval_dataset = data_processor.process_dataset(eval_dataset)
    
    # Get dataset statistics
    stats = data_processor.get_statistics(train_dataset.select(range(min(100, len(train_dataset)))))
    print("\nDataset Statistics (first 100 samples):")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Step 4: Train Model
    print_header("Step 4: Training Model with DPO")
    
    dpo_trainer = DPOTrainerWrapper(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config
    )
    
    # Start training
    dpo_trainer.train()
    
    # Save the trained model
    dpo_trainer.save_model()
    
    # Step 5: Evaluate Model
    print_header("Step 5: Evaluating Trained Model")
    
    evaluator = ModelEvaluator(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=config
    )
    
    # Run evaluation
    eval_results = evaluator.evaluate(eval_dataset)
    
    # Save evaluation results
    import json
    with open(f"{output_dir}/eval_results.json", 'w') as f:
        # Convert non-serializable objects to strings
        serializable_results = {
            k: v for k, v in eval_results.items() 
            if k != 'samples'
        }
        json.dump(serializable_results, f, indent=2)
    
    print_header("Training Complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Preference Accuracy: {eval_results['preference_accuracy']:.2%}")
    
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a language model using Direct Preference Optimization (DPO)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/dpo_config.yaml",
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Run training
    main(args.config)
