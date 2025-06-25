import torch
import pickle
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from captum.attr import IntegratedGradients
import torch.nn.functional as F
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords

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

class RegressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.samples = []
        for k, text in texts.items():
            if k not in labels:
                continue
            tokens = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            target = [labels[k][t] for t in traits]
            self.samples.append((tokens, torch.tensor(target, dtype=torch.float)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, target = self.samples[idx]
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "targets": target
        }

# Модель
class BERTRegressor(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(self.bert.config.hidden_size, len(traits))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] токен
        pooled = self.dropout(pooled)
        return self.regressor(pooled)
    
def train_model():
    texts_train, labels_train, texts_val, labels_val = load_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = RegressionDataset(texts_train, labels_train, tokenizer)
    val_ds = RegressionDataset(texts_val, labels_val, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    model = BERTRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)  # weight decay
    criterion = torch.nn.HuberLoss(delta=1.0)  # Huber loss

    best_val_loss = float("inf")

    for epoch in range(1, 31):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch} [Train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            preds = model(input_ids, attention_mask)
            loss = criterion(preds, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # стабилизация
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

        print(f"Epoch {epoch}: Train loss = {total_train_loss / len(train_dl):.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)

                preds = model(input_ids, attention_mask)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_dl)
        print(f"Epoch {epoch}: Val loss = {val_loss_avg:.4f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), "bert_regressor_best.pt")
            print("Новый лучший модельный вес сохранён")

    print("Обучение завершено!")

def extract_keywords():
    with open("../transcription_validation.pkl", "rb") as f:
        transcripts_val = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = BERTRegressor().to(device)
    model.load_state_dict(torch.load("bert_regressor_best.pt", map_location=device))
    model.eval()

    nltk.download("stopwords")
    stop_words = set(stopwords.words("english") + list(string.punctuation))

    def clean_token(token):
        return token.replace("##", "").lower()

    def interpret(text, trait_index):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        embeddings = model.bert.embeddings(input_ids)
        embeddings.requires_grad_()

        def forward_func(inputs_embeds):
            outputs = model.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]
            return model.regressor(pooled)[:, trait_index]

        ig = IntegratedGradients(forward_func)
        attributions, _ = ig.attribute(inputs=embeddings, return_convergence_delta=True)

        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        scores = attributions.detach().cpu().numpy()

        return list(zip(input_tokens, scores))

    all_tokens_per_trait = {t: [] for t in traits}
    for text in tqdm(transcripts_val.values(), desc="Extracting"):
        for i, trait in enumerate(traits):
            token_scores = interpret(text, i)
            filtered = [
                clean_token(t) for t, s in token_scores
                if t not in tokenizer.all_special_tokens and
                   clean_token(t) not in stop_words and
                   clean_token(t).isalpha()
            ]
            all_tokens_per_trait[trait].extend(filtered)

    all_tokens_flat = sum(all_tokens_per_trait.values(), [])
    global_most_common = {w for w, _ in Counter(all_tokens_flat).most_common(50)}

    top_k = 10
    most_common_per_trait = {
        trait: [
            word for word, _ in Counter(words).most_common(top_k)
            if word not in global_most_common
        ]
        for trait, words in all_tokens_per_trait.items()
    }

    lines = [
        f"Words associated with high {trait}: {', '.join(words)}."
        for trait, words in most_common_per_trait.items()
    ]
    hint_text = "\n".join(lines)
    with open("trait_hints.txt", "w") as f:
        f.write(hint_text)

    print("Подсказки сохранены в 'trait_hints.txt':")
    print(hint_text)

if __name__ == '__main__':
    train_model()
    extract_keywords()