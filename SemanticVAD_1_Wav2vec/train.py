import os
import time
import torch
import torch.nn as nn
import evaluate
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    TrainingArguments,
    Trainer,
    set_seed
)



from transformers.modeling_outputs import SequenceClassifierOutput

class DistilHuBERTClassifier(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.encoder = base_model
        # Use MLP
        self.classifier = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


class DistilHuBERTClassifierBinary(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.classifier = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.encoder(input_values=input_values, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            logits = logits.view(-1)
            labels = labels.float().view(-1)
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )