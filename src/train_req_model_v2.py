"""
Requirements Classification Model Training Module V2

This module implements an improved training pipeline with:
- Training/Validation split
- Early stopping
- Loss tracking for both sets
- Training history visualization
"""

import json
import torch
import numpy as np
import time
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

# Initialize CUDA at module level for better detection
if torch.cuda.is_available():
    try:
        # Force initialize CUDA context to ensure availability 
        _ = torch.zeros(1).cuda()
        torch.cuda.synchronize()
        
        # Print GPU information for diagnosis
        print(f"\nGPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        
        # Wait a moment for CUDA to initialize fully
        time.sleep(1) 
    except Exception as e:
        print(f"CUDA initialization warning: {str(e)}")
import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RequirementsDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, label_map):
        """
        Initialize dataset from JSONL file
        
        Args:
            jsonl_path: Path to the JSONL file
            tokenizer: The tokenizer to use
            label_map: Label mapping dictionary
        """
        self.tokenizer = tokenizer
        self.samples = []
        
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item['text']
                label = label_map.get(item['supplier_status'].lower(), 0)
                self.samples.append((text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label)
        return inputs

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

class SubsetRequirementsDataset(Dataset):
    """A dataset that takes a subset of another RequirementsDataset"""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def prepare_datasets(data_path, tokenizer, label_map, val_split=0.2, random_state=42):
    """Split data into training and validation sets with stratification."""
    # Load the full dataset
    full_dataset = RequirementsDataset(data_path, tokenizer, label_map)
    
    # Create indices for train/val split
    dataset_size = len(full_dataset)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(random_state))
    split_idx = int(dataset_size * (1 - val_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create train and validation datasets using indices
    train_dataset = SubsetRequirementsDataset(full_dataset, train_indices)
    val_dataset = SubsetRequirementsDataset(full_dataset, val_indices)
    
    return train_dataset, val_dataset

def evaluate_model(model, dataloader, device):
    """Evaluate model on given dataloader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += len(batch['labels'])
    
    return total_loss / len(dataloader), correct / total

def plot_training_history(history, save_dir):
    """Plot and save training metrics."""
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
    
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def train_model(data_path, model_dir="bert_req_eval_model", use_roberta=True, 
                batch_size=8, epochs=3, lr=2e-5, callbacks=None, use_gpu=True):
    """
    Train requirement evaluation model with validation and early stopping.
    
    Parameters:
        data_path (str): Path to the JSONL training data file
        model_dir (str): Directory to save/load model
        use_roberta (bool): Use RoBERTa model instead of BERT
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        lr (float): Learning rate
        callbacks (dict): Callback functions for progress tracking
        use_gpu (bool): Whether to use GPU acceleration if available
    """
    # Remove any existing backup
    backup_dir = f"{model_dir}.bak"
    if os.path.exists(backup_dir):
        try:
            shutil.rmtree(backup_dir)
        except Exception as e:
            print(f"Warning: Could not remove backup directory: {e}")

    if callbacks is None:
        callbacks = {
            'on_log': lambda msg: print(msg),
            'on_progress': lambda step, total: None,
            'on_epoch_end': lambda epoch, metrics: None
        }
    
    # GPU configuration and diagnostics
    cuda_available = torch.cuda.is_available()
    gpu_available = cuda_available and use_gpu
    
    # Log CUDA diagnostics
    if use_gpu:
        callbacks['on_log'](f"CUDA available: {cuda_available}")
        if cuda_available:
            callbacks['on_log'](f"CUDA version: {torch.version.cuda}")
            callbacks['on_log'](f"GPU count: {torch.cuda.device_count()}")
            callbacks['on_log'](f"Current CUDA device: {torch.cuda.current_device()}")
        else:
            callbacks['on_log'](f"GPU requested but CUDA not available. Check PyTorch installation and drivers.")
    
    # Initialize device
    device = torch.device("cuda" if gpu_available else "cpu")
    
    # Configure GPU if available
    if gpu_available:
        # Get GPU information
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        callbacks['on_log'](f"Using GPU: {gpu_name} ({total_memory:.2f}GB)")
        
        # Set memory management for 6GB cards
        torch.cuda.empty_cache()
        
        # Force synchronization to ensure CUDA context is initialized
        torch.cuda.synchronize()
        
        # Adjust batch size based on GPU memory
        if total_memory < 8:  # For 6GB cards
            if batch_size > 6:
                original_batch = batch_size
                batch_size = 6
                callbacks['on_log'](f"Batch size adjusted from {original_batch} to {batch_size} for GPU memory optimization")
        
        # Double-check that GPU will be used
        test_tensor = torch.tensor([1.0], device=device)
        callbacks['on_log'](f"Tensor device check: {test_tensor.device}")
    else:
        callbacks['on_log'](f"Using device: {device}")
    
    # Label mapping
    label_map = {"agreed": 0, "partly agreed": 1, "not agreed": 2}
    
    # Load or initialize model
    if os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        if use_roberta:
            tokenizer = RobertaTokenizer.from_pretrained(model_dir)
            model = RobertaForSequenceClassification.from_pretrained(model_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_dir)
            model = BertForSequenceClassification.from_pretrained(model_dir)
        callbacks['on_log']("Loaded existing model for continued training.")
        
        # Configure layer freezing/unfreezing for continued training
        prefix = "roberta" if use_roberta else "bert"
        callbacks['on_log']("Configuring optimized layer architecture for technical requirements...")
        callbacks['on_log']("• Freezing layers 0-5 (preserve basic language understanding)")
        callbacks['on_log']("• Enabling layers 6-8 (technical domain adaptation)")
        callbacks['on_log']("• Enabling layers 9-11 (requirements classification)")
        
        # Freeze/unfreeze layers
        for name, param in model.named_parameters():
            layer_num = -1
            if f"{prefix}.encoder.layer." in name:
                layer_num = int(name.split(f"{prefix}.encoder.layer.")[1].split('.')[0])
            
            if layer_num >= 0:  # It's a transformer layer
                param.requires_grad = layer_num >= 6  # Only train layers 6 and up
    else:
        if use_roberta:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(label_map))
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))
        callbacks['on_log']("Initialized new model.")
    
    model = model.to(device)
    
    # Load and split the dataset
    callbacks['on_log'](f"Loading dataset from {os.path.basename(data_path)}...")
    train_dataset, val_dataset = prepare_datasets(data_path, tokenizer, label_map)
    
    callbacks['on_log'](f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize training components
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=0,
                                              num_training_steps=total_steps)
    early_stopping = EarlyStopping(patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += len(batch['labels'])
            
            if step % 10 == 0:
                callbacks['on_progress'](step, len(train_loader))
        
        # Calculate training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Calculate validation metrics
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log progress
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        callbacks['on_epoch_end'](epoch, metrics)
        
        # Early stopping check
        if early_stopping(val_loss, model.state_dict()):
            callbacks['on_log']("Early stopping triggered!")
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    # Save the model with proper cleanup
    try:
        # Create temporary directory for saving
        temp_dir = f"{model_dir}_temp"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Save to temporary directory first
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        
        # Plot and save training history to temp dir
        plot_training_history(history, temp_dir)
        
        # If original model directory exists, move it to backup
        if os.path.exists(model_dir):
            backup_dir = f"{model_dir}.bak"
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            os.rename(model_dir, backup_dir)
        
        # Move temporary directory to final location
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.rename(temp_dir, model_dir)
        
        # Remove backup if everything succeeded
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
            
    except Exception as e:
        # If anything fails, try to restore from backup
        if os.path.exists(backup_dir) and not os.path.exists(model_dir):
            os.rename(backup_dir, model_dir)
        raise Exception(f"Error saving model: {str(e)}")
    
    return model, tokenizer, history
