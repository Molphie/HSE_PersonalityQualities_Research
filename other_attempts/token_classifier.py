import torch
import pickle
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F

traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_labels(raw):
    scores = {}
    for trait, trait_dict in raw.items():
        for clip_id, value in trait_dict.items():
            scores.setdefault(clip_id, {})[trait] = value
    return scores

def load_data():
    with open("../transcription_training.pkl", "rb") as f:
        texts_train = pickle.load(f)
    with open("../annotation_training.pkl", "rb") as f:
        labels_train = load_labels(pickle.load(f, encoding="latin1"))
    with open("../transcription_validation.pkl", "rb") as f:
        texts_val = pickle.load(f)
    with open("../annotation_validation.pkl", "rb") as f:
        labels_val = load_labels(pickle.load(f, encoding="latin1"))
    return texts_train, labels_train, texts_val, labels_val

class TokenClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.samples = []
        for k, text in texts.items():
            if k not in labels:
                continue
            tokens = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            for i, trait in enumerate(traits):
                label_value = labels[k][trait]
                label = 1 if label_value > 0.5 else 0
                self.samples.append((tokens, label, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, label, trait_index = self.samples[idx]
        item = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor([label] * 128),
            "trait_index": trait_index
        }
        return item

class TraitClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.base = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2)

    def forward(self, input_ids, attention_mask):
        return self.base(input_ids=input_ids, attention_mask=attention_mask).logits
    
def train_model():
    texts_train, labels_train, texts_val, labels_val = load_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = TokenClassificationDataset(texts_train, labels_train, tokenizer)
    val_ds = TokenClassificationDataset(texts_val, labels_val, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    model = TraitClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(1, 20):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch} [Train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, 2), labels.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()

        print(f"Epoch {epoch}: Train loss = {total_train_loss / len(train_dl):.4f}")

        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, 2), labels.view(-1))
                total_val_loss += loss.item()

                preds = outputs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        val_loss = total_val_loss / len(val_dl)
        val_acc = correct / total
        print(f"Epoch {epoch}: Val loss = {val_loss:.4f}, Val acc = {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "token_classifier.pt")
            print("Model improved and saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

train_model()