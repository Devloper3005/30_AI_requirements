# ...existing code...

def set_trainable_layers(model, train_classifier_only=True, train_last_n_encoder_layers=0):
    """
    Set which layers are trainable.
    - train_classifier_only: If True, only the classifier head is trainable.
    - train_last_n_encoder_layers: If >0, unfreezes the last n encoder layers.
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier head
    if train_classifier_only:
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Optionally unfreeze last n encoder layers
    if train_last_n_encoder_layers > 0:
        encoder_layers = model.bert.encoder.layer
        for layer in encoder_layers[-train_last_n_encoder_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

# Load model
model, tokenizer = load_bert_model()

# Example: Only classifier head is trainable
set_trainable_layers(model, train_classifier_only=True)

# Example: Unfreeze classifier head and last 2 encoder layers
# set_trainable_layers(model, train_classifier_only=True, train_last_n_encoder_layers=2)


# Install nlpaug and nltk if not already installed
# pip install nlpaug nltk

import nlpaug.augmenter.word as naw
import random

class RequirementsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_map, augment=False, aug_prob=0.3):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.augment = augment
        self.aug_prob = aug_prob
        self.aug = naw.SynonymAug(aug_src='wordnet')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Apply augmentation with a certain probability
        if self.augment and random.random() < self.aug_prob:
            text = self.aug.augment(text)
        
        # Tokenize as usual
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.label_map[label])
        return item
    
    
import nlpaug.augmenter.word as naw
import random


class RequirementsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_map, num_augments=1):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.aug = naw.SynonymAug(aug_src='wordnet')
        self.texts = []
        self.labels = []

        # Add original data
        for text, label in zip(texts, labels):
            self.texts.append(text)
            self.labels.append(label)
            # Add augmented data
            for _ in range(num_augments):
                aug_text = self.aug.augment(text)
                self.texts.append(aug_text)
                self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.label_map[label])
        return item