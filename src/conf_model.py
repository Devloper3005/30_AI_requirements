
# %%
import torch 
import transformers
from transformers import BertConfig, BertModel, BertTokenizer  # BERT model and tokenizer
from transformers import BertForSequenceClassification

# %% Config data 

# %%  Define static variables
reverse_label_map = {0: "agreed", 1: "partly agreed", 2: "not agreed"}

# %%
def create_bert_model():
    # Load pre-trained model and tokenizer
    config = BertConfig.from_pretrained('bert-base-cased', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-cased', config = config)
    print("BERT model and tokenizer loaded successfully.")
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = create_bert_model()

# %%
def test_model_output(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return reverse_label_map[predicted_class]
