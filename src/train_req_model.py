import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
<<<<<<< HEAD
from transformers import get_linear_schedule_with_warmup
import os
import time
import random
import shutil
import tempfile
=======
#test
>>>>>>> 8f3a9ced41c5225c70dc4de774ae8bea7d1f4f12

class RequirementsDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, label_map):
        self.samples = []
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item['text']
                label = label_map.get(item['supplier_status'].lower(), 0)
                self.samples.append((text, label))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label)
        return inputs

<<<<<<< HEAD
def train_model(data_path, model_dir="bert_req_eval_model", use_roberta=True, 
                batch_size=8, epochs=3, train_all_layers=False, lr=None, 
                callbacks=None, find_lr=True):
    """
    Train requirement evaluation model with improved architecture and optional learning rate search
    
    Args:
        data_path: Path to training JSONL file
        model_dir: Directory to save model
        use_roberta: Use RoBERTa (True) or BERT (False)
        batch_size: Training batch size
        epochs: Number of training epochs
        train_all_layers: Whether to train all layers
        lr: Learning rate
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
=======
# Main function
if __name__ == "__main__":
>>>>>>> 8f3a9ced41c5225c70dc4de774ae8bea7d1f4f12
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
    
    # Freeze layers based on configuration
    if not train_all_layers:
        callbacks['on_log']("Freezing layers (training only last 4 layers)...")
        # Note: RoBERTa uses slightly different naming
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
    
    callbacks['on_log'](f"Loading dataset from {data_path}...")
    dataset = RequirementsDataset(data_path, tokenizer, label_map)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    callbacks['on_log'](f"Dataset loaded with {len(dataset)} samples")
    
    # Setup optimizer with weight decay and learning rate scheduler
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