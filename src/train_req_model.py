import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
#test

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

# Main function
if __name__ == "__main__":
    label_map = {"agreed": 0, "partly agreed": 1, "not agreed": 2}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_map))

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

    dataset = RequirementsDataset("requirements_for_llm.jsonl", tokenizer, label_map)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    model.train()
    for epoch in range(3):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} loss: {loss.item()}")
    model.save_pretrained("bert_req_eval_model")
    tokenizer.save_pretrained("bert_req_eval_model")