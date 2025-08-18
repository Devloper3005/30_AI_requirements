"""
Requirements Classification Model Training Module

This module handles the training of models for requirements classification using BERT or RoBERTa.
It includes dataset handling, model configuration, and training pipeline with progress monitoring.
"""

import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Union
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    PreTrainedTokenizer, PreTrainedModel
)
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import shutil
import tempfile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RequirementsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.samples = list(zip(texts, labels))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label)
        return inputs

def prepare_datasets(data_path, tokenizer, label_map, val_split=0.2, random_state=42):
    """
    Split data into training and validation sets with stratification.
    """
    samples = []
    labels = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            samples.append(item['text'])
            labels.append(label_map[item['supplier_status'].lower()])
    
    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        samples, labels, 
        test_size=val_split, 
        random_state=random_state,
        stratify=labels
    )
    
    return (
        RequirementsDataset(X_train, y_train, tokenizer),
        RequirementsDataset(X_val, y_val, tokenizer)
    )

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model_state):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model_state
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model_state
            self.counter = 0
        return self.should_stop

def plot_training_history(history, save_path=None):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def train_model(data_path, model_dir="bert_req_eval_model", use_roberta=True, 
                batch_size=8, epochs=3, lr=2e-5, callbacks=None, find_lr=True):
    """
    Train requirement evaluation model optimized for technical requirements classification.
    
    The model uses a partial fine-tuning approach, training the last 6 layers of the transformer
    model plus the classification head. This architecture is chosen to:
    1. Maintain basic language understanding in lower layers
    2. Adapt middle and upper layers for technical/engineering domain
    3. Balance between domain adaptation and preventing overfitting
    
    Layer Training Strategy:
    - Layers 0-5: Frozen (basic language patterns)
    - Layers 6-11: Trainable (domain adaptation)
    - Classifier: Trainable (task-specific)
    
    Args:
        data_path: Path to training JSONL file containing requirements
        model_dir: Directory to save the trained model
        use_roberta: Use RoBERTa (True) or BERT (False). RoBERTa recommended for technical text
        batch_size: Training batch size (default: 8)
        epochs: Number of training epochs (default: 3)
        lr: Learning rate (default: 2e-5). This rate is optimized for transformer fine-tuning
        callbacks: Dictionary of callback functions for progress updates
            - on_log(message): Called when logging a message
            - on_progress(step, total_steps): Called to update progress
            - on_epoch_end(epoch, avg_loss): Called at end of epoch
    
    Returns:
        tuple: (model, tokenizer, training_history)
    """
    # Default callbacks
    if callbacks is None:
        callbacks = {
            'on_log': lambda msg: print(msg),
            'on_progress': lambda step, total: None,
            'on_epoch_end': lambda epoch, loss: None
        }
    
    # Initialize callbacks that aren't provided
    for key in ['on_log', 'on_progress', 'on_epoch_end']:
        if key not in callbacks:
            callbacks[key] = lambda *args: None
    
    # Load appropriate model based on selection
    label_map = {"agreed": 0, "partly agreed": 1, "not agreed": 2}
    
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        # Continue training from last saved model
        if use_roberta:
            tokenizer = RobertaTokenizer.from_pretrained(model_dir)
            model = RobertaForSequenceClassification.from_pretrained(model_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_dir)
            model = BertForSequenceClassification.from_pretrained(model_dir)
        callbacks['on_log']("Loaded existing model for continued training.")
    else:
        # Train from scratch
        if use_roberta:
            callbacks['on_log']("Loading RoBERTa tokenizer and model...")
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_map))
        else:
            callbacks['on_log']("Loading BERT tokenizer and model...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))
        callbacks['on_log']("No existing model found. Training from scratch.")
    
    # Configure optimized layer architecture for technical requirements
    prefix = "roberta" if use_roberta else "bert"
    callbacks['on_log']("Configuring optimized layer architecture for technical requirements...")
    callbacks['on_log']("• Freezing layers 0-5 (preserve basic language understanding)")
    callbacks['on_log']("• Enabling layers 6-8 (technical domain adaptation)")
    callbacks['on_log']("• Enabling layers 9-11 (requirements classification)")
    
    # Configure which layers to train
    for name, param in model.named_parameters():
        if not (
            # Middle layers (6-8) for domain adaptation
            name.startswith(f"{prefix}.encoder.layer.6") or
            name.startswith(f"{prefix}.encoder.layer.7") or
            name.startswith(f"{prefix}.encoder.layer.8") or
            # Upper layers (9-11) for classification
            name.startswith(f"{prefix}.encoder.layer.9") or
            name.startswith(f"{prefix}.encoder.layer.10") or
            name.startswith(f"{prefix}.encoder.layer.11") or
            # Classification head always trained
            name.startswith("classifier")
        ):
            param.requires_grad = False
    
    callbacks['on_log'](f"Loading dataset from {data_path}...")
    dataset = RequirementsDataset(data_path, tokenizer, label_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    callbacks['on_log'](f"Dataset loaded with {len(dataset)} samples")
    
    # Setup optimizer with weight decay and learning rate scheduler
    if lr is None:
        lr = 2e-5  # Default learning rate for transformer fine-tuning
    
    callbacks['on_log'](f"Using learning rate: {lr}")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr,
        weight_decay=0.01
    )
    
    # Add learning rate scheduler
    total_steps = epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    model.train()
    callbacks['on_log']("Starting training...")
    
    # Track training history
    history = {'epoch_losses': []}
    
    # Training loop
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            step += 1
            callbacks['on_progress'](step, total_steps)
            
            # Log every few batches
            if batch_idx % 5 == 0 or batch_idx == len(dataloader) - 1:
                callbacks['on_log'](f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        history['epoch_losses'].append(avg_epoch_loss)
        callbacks['on_log'](f"Epoch {epoch+1} completed, Avg Loss: {avg_epoch_loss:.4f}")
        callbacks['on_epoch_end'](epoch+1, avg_epoch_loss)
    
    # Save model safely
    callbacks['on_log']("Training complete. Saving model...")

    # Save to a temporary directory first
    temp_dir = tempfile.mkdtemp()
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    # Remove the old model directory if it exists
    if os.path.exists(model_dir):
        try:
            shutil.rmtree(model_dir)
        except Exception as e:
            callbacks['on_log'](f"Warning: Could not remove old model directory: {str(e)}")

    # Move the temp directory to the model_dir
    try:
        shutil.move(temp_dir, model_dir)
        callbacks['on_log'](f"Model saved successfully to {model_dir}!")
    except Exception as e:
        callbacks['on_log'](f"Error during model save: {str(e)}")
        raise Exception(f"Failed to save model: {str(e)}")

    # Return values should include the actual directory where model was saved
    return model, tokenizer, {"epoch_losses": history['epoch_losses'], "model_dir": model_dir}

# Main function
if __name__ == "__main__":
    print("Requirements Evaluation Model Training")
    print("-" * 40)
    data_path = input("Enter path to JSONL training data: ")
    use_roberta = input("Use RoBERTa instead of BERT? (y/n): ").lower() == 'y'
    batch_size = int(input("Enter batch size (default 8): ") or "8")
    epochs = int(input("Enter number of epochs (default 3): ") or "3")
    train_all = input("Train all layers? (y/n): ").lower() == 'y'
    model_dir = input("Enter model directory (default 'bert_req_eval_model'): ") or "bert_req_eval_model"
    find_lr = input("Perform learning rate search? (y/n): ").lower() == 'y'
    
    if find_lr:
        from lr_search import find_optimal_lr
        print("\nPerforming learning rate search...")
        optimal_lr, _ = find_optimal_lr(
            data_path=data_path,
            model_dir=model_dir,
            use_roberta=use_roberta,
            batch_size=batch_size,
            callbacks={'on_log': lambda msg: print(msg)}
        )
        print(f"\nOptimal learning rate found: {optimal_lr}")
    else:
        optimal_lr = 3e-5  # default learning rate
    
    train_model(
        data_path, 
        model_dir=model_dir,
        use_roberta=use_roberta, 
        batch_size=batch_size,
        epochs=epochs,
        train_all_layers=train_all,
        lr=optimal_lr,
        find_lr=False,
        callbacks={'on_log': lambda msg: print(msg)}
    )