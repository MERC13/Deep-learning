"""
Generate DPO Process Visualization

This script creates a visual diagram explaining the DPO training process.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_dpo_flowchart():
    """Create a flowchart showing the DPO training process."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Direct Preference Optimization (DPO) Training Flow', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Colors
    data_color = '#3498db'
    model_color = '#2ecc71'
    training_color = '#e74c3c'
    eval_color = '#f39c12'
    
    # Step 1: Preference Data
    box1 = FancyBboxPatch((0.5, 9.5), 2.5, 1, boxstyle="round,pad=0.1", 
                          facecolor=data_color, edgecolor='black', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.75, 10.3, 'Preference Dataset', fontsize=12, ha='center', fontweight='bold', color='white')
    ax.text(1.75, 9.9, '(prompt, chosen, rejected)', fontsize=9, ha='center', color='white')
    
    # Step 2: Base Model
    box2 = FancyBboxPatch((4, 9.5), 2, 1, boxstyle="round,pad=0.1", 
                          facecolor=model_color, edgecolor='black', linewidth=2)
    ax.add_patch(box2)
    ax.text(5, 10.2, 'Base Model', fontsize=12, ha='center', fontweight='bold', color='white')
    ax.text(5, 9.8, '(e.g., GPT-2)', fontsize=9, ha='center', color='white')
    
    # Step 3: Reference Model
    box3 = FancyBboxPatch((7.5, 9.5), 2, 1, boxstyle="round,pad=0.1", 
                          facecolor=model_color, edgecolor='black', linewidth=2, linestyle='--')
    ax.add_patch(box3)
    ax.text(8.5, 10.2, 'Reference Model', fontsize=12, ha='center', fontweight='bold', color='white')
    ax.text(8.5, 9.8, '(frozen copy)', fontsize=9, ha='center', color='white')
    
    # Arrow from base to reference
    arrow1 = FancyArrowPatch((6, 10), (7.5, 10), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='gray')
    ax.add_patch(arrow1)
    ax.text(6.75, 10.3, 'copy', fontsize=9, ha='center', style='italic')
    
    # Step 4: Training Process (central box)
    box4 = FancyBboxPatch((2, 6.5), 6, 2, boxstyle="round,pad=0.15", 
                          facecolor=training_color, edgecolor='black', linewidth=3, alpha=0.9)
    ax.add_patch(box4)
    ax.text(5, 8.1, 'DPO Training', fontsize=14, ha='center', fontweight='bold', color='white')
    
    # DPO formula inside training box
    ax.text(5, 7.5, r'Optimize: $\log \sigma(\beta \cdot [\log P(y_{chosen}) - \log P(y_{rejected})])$', 
            fontsize=11, ha='center', color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
    ax.text(5, 7.0, '• Increase P(chosen responses)', fontsize=9, ha='center', color='white')
    ax.text(5, 6.7, '• Decrease P(rejected responses)', fontsize=9, ha='center', color='white')
    
    # Arrows into training
    arrow2 = FancyArrowPatch((1.75, 9.5), (3.5, 8.5), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    arrow3 = FancyArrowPatch((5, 9.5), (5, 8.5), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((8.5, 9.5), (6.5, 8.5), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow4)
    
    # Step 5: Aligned Model
    box5 = FancyBboxPatch((3.5, 4.5), 3, 1, boxstyle="round,pad=0.1", 
                          facecolor=model_color, edgecolor='black', linewidth=2)
    ax.add_patch(box5)
    ax.text(5, 5.2, 'Aligned Model', fontsize=12, ha='center', fontweight='bold', color='white')
    ax.text(5, 4.8, '(DPO fine-tuned)', fontsize=9, ha='center', color='white')
    
    # Arrow from training to aligned model
    arrow5 = FancyArrowPatch((5, 6.5), (5, 5.5), arrowstyle='->', 
                            mutation_scale=20, linewidth=3, color='black')
    ax.add_patch(arrow5)
    
    # Step 6: Evaluation
    box6 = FancyBboxPatch((2, 2.5), 6, 1.5, boxstyle="round,pad=0.1", 
                          facecolor=eval_color, edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(box6)
    ax.text(5, 3.7, 'Evaluation', fontsize=12, ha='center', fontweight='bold')
    ax.text(5, 3.3, '• Preference Accuracy (> 70%)', fontsize=9, ha='center')
    ax.text(5, 3.0, '• Sample Generation Quality', fontsize=9, ha='center')
    ax.text(5, 2.7, '• KL Divergence from Base', fontsize=9, ha='center')
    
    # Arrow to evaluation
    arrow6 = FancyArrowPatch((5, 4.5), (5, 4), arrowstyle='->', 
                            mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow6)
    
    # Benefits box
    benefits_box = FancyBboxPatch((0.5, 0.2), 4, 1.8, boxstyle="round,pad=0.1", 
                                  facecolor='#9b59b6', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(benefits_box)
    ax.text(2.5, 1.7, 'DPO Benefits', fontsize=11, ha='center', fontweight='bold', color='white')
    ax.text(2.5, 1.4, '✓ No reward model needed', fontsize=9, ha='center', color='white')
    ax.text(2.5, 1.1, '✓ Simpler than RLHF', fontsize=9, ha='center', color='white')
    ax.text(2.5, 0.8, '✓ More stable training', fontsize=9, ha='center', color='white')
    ax.text(2.5, 0.5, '✓ Better alignment', fontsize=9, ha='center', color='white')
    
    # Key parameters box
    params_box = FancyBboxPatch((5.5, 0.2), 4, 1.8, boxstyle="round,pad=0.1", 
                               facecolor='#34495e', edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(params_box)
    ax.text(7.5, 1.7, 'Key Parameters', fontsize=11, ha='center', fontweight='bold', color='white')
    ax.text(7.5, 1.4, 'β (beta): 0.1 - 0.3', fontsize=9, ha='center', color='white')
    ax.text(7.5, 1.1, 'Learning rate: 5e-5', fontsize=9, ha='center', color='white')
    ax.text(7.5, 0.8, 'Epochs: 1-3', fontsize=9, ha='center', color='white')
    ax.text(7.5, 0.5, 'Batch size: 2-8', fontsize=9, ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig('outputs/dpo_flowchart.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ DPO flowchart saved to: outputs/dpo_flowchart.png")
    plt.show()

def create_comparison_chart():
    """Create a comparison chart: RLHF vs DPO."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # RLHF Process
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Traditional RLHF', fontsize=18, fontweight='bold', pad=20)
    
    # RLHF boxes
    rlhf_boxes = [
        (2, 10, 'Step 1: Supervised\nFine-tuning', '#3498db'),
        (2, 8, 'Step 2: Train\nReward Model', '#e74c3c'),
        (2, 6, 'Step 3: RL Training\n(PPO)', '#f39c12'),
        (2, 4, 'Aligned Model', '#2ecc71'),
    ]
    
    for i, (y, label, color) in enumerate(rlhf_boxes):
        box = FancyBboxPatch((2, y), 6, 1.5, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax1.add_patch(box)
        ax1.text(5, y + 0.75, label, fontsize=12, ha='center', fontweight='bold', color='white')
        
        if i < len(rlhf_boxes) - 1:
            arrow = FancyArrowPatch((5, y), (5, y + 1.5), arrowstyle='->', 
                                  mutation_scale=25, linewidth=3, color='black')
            ax1.add_patch(arrow)
    
    ax1.text(5, 2.5, 'Complex: 3 stages', fontsize=11, ha='center', style='italic', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax1.text(5, 1.5, 'Requires reward model', fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    ax1.text(5, 0.5, 'Can be unstable', fontsize=11, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # DPO Process
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Direct Preference Optimization (DPO)', fontsize=18, fontweight='bold', pad=20)
    
    # DPO boxes
    dpo_boxes = [
        (2, 8, 'Preference Data\n+ Base Model', '#3498db'),
        (2, 5.5, 'DPO Training\n(Single Step)', '#2ecc71'),
        (2, 3, 'Aligned Model', '#2ecc71'),
    ]
    
    for i, (y, label, color) in enumerate(dpo_boxes):
        height = 2 if i == 1 else 1.5
        box = FancyBboxPatch((2, y), 6, height, boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax2.add_patch(box)
        ax2.text(5, y + height/2, label, fontsize=12, ha='center', fontweight='bold', color='white')
        
        if i < len(dpo_boxes) - 1:
            next_y = dpo_boxes[i+1][0]
            next_height = 2 if i+1 == 1 else 1.5
            arrow = FancyArrowPatch((5, y), (5, next_y + next_height), arrowstyle='->', 
                                  mutation_scale=25, linewidth=3, color='black')
            ax2.add_patch(arrow)
    
    ax2.text(5, 1.5, '✓ Simple: 1 stage', fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.text(5, 0.7, '✓ No reward model', fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('outputs/rlhf_vs_dpo.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ RLHF vs DPO comparison saved to: outputs/rlhf_vs_dpo.png")
    plt.show()

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("Generating DPO visualizations...")
    print("-" * 80)
    
    create_dpo_flowchart()
    create_comparison_chart()
    
    print("-" * 80)
    print("✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - outputs/dpo_flowchart.png")
    print("  - outputs/rlhf_vs_dpo.png")
