import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load trained model and tokenizer#test
model = BertForSequenceClassification.from_pretrained("bert_req_eval_model")
tokenizer = BertTokenizer.from_pretrained("bert_req_eval_model")
model.eval()

reverse_label_map = {0: "agreed", 1: "partly agreed", 2: "not agreed"}

def predict_supplier_status(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return reverse_label_map[predicted_class]

# Main function
if __name__ == "__main__":
    print("Requirement Evaluation Tool")
    print("--------------------------")
    while True:
        req_text = input("Enter requirement text (or type 'exit' to quit): ")
        if req_text.lower() == 'exit':
            break
        result = predict_supplier_status(req_text)
        print(f"Predicted supplier status: {result}\n")