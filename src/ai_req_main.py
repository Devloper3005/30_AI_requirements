# %% LIBRARY IMPORTS
# numerics 
import numpy as np

# Keras and TensorFlow for deep learning
import torch 
import transformers
from transformers import BertConfig, BertModel, BertTokenizer  # BERT model and tokenizer
from transformers import BertForSequenceClassification


# %% # Load pre-trained BERT model and tokenizer

def load_bert_model():
    # Load pre-trained model and tokenizer
    config = BertConfig.from_pretrained('bert-base-cased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', config = config)
    return model, tokenizer

def process_text(text, model, tokenizer):
    # Prepare the text input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Get the BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs

# Load model
model, tokenizer = load_bert_model()

#print(transformers.__version__)
#print(torch.__version__)




# %% Test run of the model 

# # Example text
text = "This is a sample requirement. and this another one"

# # Process the text
embeddings = process_text(text, model, tokenizer)
print(f"Embedding logits for text input: {embeddings.logits}") # Output of logits for the number of labels


# %%
# ...existing code...