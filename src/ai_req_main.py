# %% LIBRARY IMPORTS
#numerics 
import numpy as np

# Torch library
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import transformers
from transformers import BertConfig, BertModel, BertTokenizer  # BERT model and tokenizer
from transformers import BertForSequenceClassification

# Sci-Learn library
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Custom library
from conf_model import create_bert_model
from conf_model import test_model_output
from train_req_model import RequirementsDataset


# %%
# Main function
if __name__ == "__main__":

    # %% # Initialize BERT model and tokenizer
    model, tokenizer = create_bert_model()

    # %% Load Dataset 

    raw_data = RequirementsDataset("requirements_for_llm.jsonl", tokenizer, label_map={"agreed": 0, "partly agreed": 1, "not agreed": 2})

    print(f'Length of input data set', raw_data.__len__())

    dataloader = DataLoader(raw_data, batch_size=8, shuffle=True)

    # %% Test run of the model 
    # # Example text
    text = "This is a sample requirement. and this another one"

    model.eval()
    test_label = test_model_output(text, model, tokenizer)
    print(f"Test label: {test_label}")

    #embeddings = process_text(text, model, tokenizer)
    #print(f"Embedding logits for text input: {embeddings.logits}") # Output of logits for the number of labels

    # %% Configure model for training 
    
    # Freeze all layers except encoder.layer.8, 9, 10, 11 and the classifier
    for name, param in model.named_parameters():
        # Only unfreeze last 4 layers and classifier
        if not (
            name.startswith("bert.encoder.layer.8") or
            name.startswith("bert.encoder.layer.9") or
            name.startswith("bert.encoder.layer.10") or
            name.startswith("bert.encoder.layer.11") or
            name.startswith("classifier")
        ):
            param.requires_grad = False
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    
    for epoch in range(3):
        # Perform training 
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} training complete.")
        
        # # Validation loop
        # model.eval()
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for batch in val_loader:
        #         input_ids = batch["input_ids"].to(model.device)
        #         attention_mask = batch["attention_mask"].to(model.device)
        #         labels = batch["labels"].to(model.device)
                
        #         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #         preds = torch.argmax(outputs.logits, dim=1)
        #         correct += (preds == labels).sum().item()
        #         total += labels.size(0)
        # accuracy = correct / total
        # print(f"Validation accuracy after epoch {epoch+1}: {accuracy:.4f}")
        
    # Testing required to calculate loss 
    # print(f"Epoch {epoch+1} loss: {loss.item()}")
    #model.save_pretrained("bert_req_eval_model")
    #tokenizer.save_pretrained("bert_req_eval_model")
