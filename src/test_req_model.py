import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Global variables for model and tokenizer
model = None
tokenizer = None
reverse_label_map = {0: "agreed", 1: "partly agreed", 2: "not agreed"}

<<<<<<< HEAD
def load_model():
    """Load the model and tokenizer if not already loaded"""
    global model, tokenizer
    if model is None or tokenizer is None:
        # Check if model directory exists
        model_path = "bert_req_eval_model"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory '{model_path}' not found. Please train a model first.")
        
        # Check if model files exist
        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(f"Model configuration file not found in '{model_path}'. Model may be corrupted.")
            
        # Check what type of model we're dealing with
        # First try loading with Auto classes which can handle multiple model types
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            # Fallback to specific model types
            try:
                # Check for RoBERTa config file
                if os.path.exists(os.path.join(model_path, "config.json")):
                    import json
                    with open(os.path.join(model_path, "config.json"), 'r') as f:
                        config = json.load(f)
                    
                    # If it's a RoBERTa model
                    if "roberta" in config.get("_name_or_path", "").lower() or config.get("model_type", "").lower() == "roberta":
                        tokenizer = RobertaTokenizer.from_pretrained(model_path)
                        model = RobertaForSequenceClassification.from_pretrained(model_path)
                    else:
                        tokenizer = BertTokenizer.from_pretrained(model_path)
                        model = BertForSequenceClassification.from_pretrained(model_path)
                else:
                    # Default to BERT
                    tokenizer = BertTokenizer.from_pretrained(model_path)
                    model = BertForSequenceClassification.from_pretrained(model_path)
            except Exception as inner_e:
                raise Exception(f"Failed to load model: {str(e)}. Inner error: {str(inner_e)}")
        
        model.eval()
    return model, tokenizer

def predict_supplier_status(text):
    """Predict supplier status for a requirement text"""
    global model, tokenizer
    
    # Load model if not already loaded
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
        
=======
def predict_supplier_status(text, model, tokenizer):
>>>>>>> 8f3a9ced41c5225c70dc4de774ae8bea7d1f4f12
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return reverse_label_map[predicted_class]

<<<<<<< HEAD
def predict_with_confidence(text):
    """Predict supplier status with confidence scores"""
    global model, tokenizer
    
    # Load model if not already loaded
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
        
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(logits, dim=1).item()
    
    result = reverse_label_map[predicted_class]
    confidence_scores = {reverse_label_map[i]: float(probs[i]) * 100 for i in range(len(probs))}
    
    return result, confidence_scores

=======
>>>>>>> 8f3a9ced41c5225c70dc4de774ae8bea7d1f4f12
# Main function
if __name__ == "__main__":
    # Load model when running as script
    load_model()
    
    print("Requirement Evaluation Tool")
    print("--------------------------")
    while True:
        req_text = input("Enter requirement text (or type 'exit' to quit): ")
        if req_text.lower() == 'exit':
            break
            
        result, confidence = predict_with_confidence(req_text)
        print(f"Predicted supplier status: {result}")
        for status, score in confidence.items():
            print(f"- {status}: {score:.1f}%")
        print()