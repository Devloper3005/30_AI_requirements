import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
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

class ReqEvalApp:
    def __init__(self, master):
        self.master = master
        master.title("Requirement Evaluation Tool")

        self.label_map = {"agreed": 0, "partly agreed": 1, "not agreed": 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.model = None
        self.tokenizer = None

        self.train_button = tk.Button(master, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.progress = ttk.Progressbar(master, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=5)

        self.test_label = tk.Label(master, text="Enter requirement text:")
        self.test_label.pack()
        self.test_entry = tk.Entry(master, width=80)
        self.test_entry.pack(pady=5)

        self.test_button = tk.Button(master, text="Test Requirement", command=self.test_requirement)
        self.test_button.pack(pady=10)

        self.result_label = tk.Label(master, text="", fg="blue")
        self.result_label.pack(pady=10)

    def train_model(self):
        self.result_label.config(text="Training model, please wait...")
        self.master.update()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(self.label_map))
        dataset = RequirementsDataset("requirements_for_llm.jsonl", tokenizer, self.label_map)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()
        total_steps = 3 * len(dataloader)
        self.progress["maximum"] = total_steps
        step = 0
        for epoch in range(3):
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = model(**{k: v for k, v in batch.items() if k != 'labels'}, labels=batch['labels'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                step += 1
                self.progress["value"] = step
                self.master.update_idletasks()
        model.save_pretrained("bert_req_eval_model")
        tokenizer.save_pretrained("bert_req_eval_model")
        self.model = model
        self.tokenizer = tokenizer
        self.result_label.config(text="Training complete!")
        self.progress["value"] = 0

    def test_requirement(self):
        req_text = self.test_entry.get()
        if not req_text:
            messagebox.showwarning("Input Error", "Please enter a requirement text.")
            return
        if self.model is None or self.tokenizer is None:
            try:
                self.model = BertForSequenceClassification.from_pretrained("bert_req_eval_model")
                self.tokenizer = BertTokenizer.from_pretrained("bert_req_eval_model")
                self.model.eval()
            except Exception:
                messagebox.showerror("Model Error", "Model not trained yet. Please train the model first.")
                return
        inputs = self.tokenizer(req_text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        result = self.reverse_label_map[predicted_class]
        self.result_label.config(text=f"Predicted supplier status: {result}")
# Main function
if __name__ == "__main__":
    root = tk.Tk()
    app = ReqEvalApp(root)
    root.mainloop()