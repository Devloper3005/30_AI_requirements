import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import time
import random
import shutil
import tempfile
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def find_optimal_lr(data_path, model_dir="bert_req_eval_model", use_roberta=True,
                   batch_size=8, search_epochs=1, train_all_layers=False,
                   lr_range=(1e-6, 1e-3), num_points=20, callbacks=None):
    """
    Perform learning rate search to find optimal learning rate
    
    Args:
        data_path: Path to training JSONL file
        model_dir: Directory to save model
        use_roberta: Use RoBERTa (True) or BERT (False)
        batch_size: Training batch size
        search_epochs: Number of epochs for each learning rate
        train_all_layers: Whether to train all layers
        lr_range: Tuple of (min_lr, max_lr)
        num_points: Number of learning rates to try
        callbacks: Dictionary of callback functions
    
    Returns:
        tuple: (optimal_lr, learning_curves)
    """
    if callbacks is None:
        callbacks = {
            'on_log': lambda msg: print(msg),
            'on_progress': lambda step, total: None
        }
    
    # Generate learning rates on log scale
    learning_rates = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num_points)
    callbacks['on_log'](f"Testing learning rates: {learning_rates}")
    
    # Load data
    label_map = {"agreed": 0, "partly agreed": 1, "not agreed": 2}
    
    # Initialize model and tokenizer
    if use_roberta:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model_class = lambda: RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_map))
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model_class = lambda: BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))
    
    # Load and split dataset
    dataset = RequirementsDataset(data_path, tokenizer, label_map)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    results = []
    best_lr = None
    best_val_f1 = -1
    
    # Test each learning rate
    for lr in learning_rates:
        callbacks['on_log'](f"\nTesting learning rate: {lr}")
        model = model_class()
        
        if not train_all_layers:
            prefix = "roberta" if use_roberta else "bert"
            for name, param in model.named_parameters():
                if not (
                    name.startswith(f"{prefix}.encoder.layer.8") or
                    name.startswith(f"{prefix}.encoder.layer.9") or
                    name.startswith(f"{prefix}.encoder.layer.10") or
                    name.startswith(f"{prefix}.encoder.layer.11") or
                    name.startswith("classifier")
                ):
                    param.requires_grad = False
        
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        # Training loop
        model.train()
        train_losses = []
        val_f1s = []
        
        for epoch in range(search_epochs):
            epoch_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(**{k: v for k, v in batch.items() if k != 'labels'})
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(batch['labels'].cpu().numpy())
            
            val_f1 = f1_score(val_labels, val_preds, average='weighted')
            val_f1s.append(val_f1)
            
            callbacks['on_log'](f"Epoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_lr = lr
        
        results.append({
            'lr': lr,
            'train_losses': train_losses,
            'val_f1s': val_f1s,
            'final_val_f1': val_f1s[-1]
        })
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.semilogx([r['lr'] for r in results], [r['final_val_f1'] for r in results], 'bo-')
    plt.grid(True)
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation F1 Score')
    plt.title('Learning Rate vs Validation F1')
    
    plt.subplot(1, 2, 2)
    plt.semilogx([r['lr'] for r in results], [r['train_losses'][-1] for r in results], 'ro-')
    plt.grid(True)
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Training Loss')
    plt.title('Learning Rate vs Training Loss')
    
    plt.tight_layout()
    plt.savefig('lr_search_results.png')
    plt.close()
    
    callbacks['on_log'](f"\nBest learning rate found: {best_lr} (Val F1: {best_val_f1:.4f})")
    return best_lr, results

# Example usage in train_model:
def train_model_with_lr_search(data_path, model_dir="bert_req_eval_model", use_roberta=True,
                             batch_size=8, epochs=3, train_all_layers=False,
                             callbacks=None):
    """Train model with automatic learning rate search"""
    
    # First find optimal learning rate
    optimal_lr, lr_results = find_optimal_lr(
        data_path=data_path,
        model_dir=model_dir,
        use_roberta=use_roberta,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Then train with the optimal learning rate
    return train_model(
        data_path=data_path,
        model_dir=model_dir,
        use_roberta=use_roberta,
        batch_size=batch_size,
        epochs=epochs,
        train_all_layers=train_all_layers,
        lr=optimal_lr,
        callbacks=callbacks
    )
