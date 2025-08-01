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
