import torch
import pickle
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from captum.attr import IntegratedGradients
from nltk.corpus import stopwords
from collections import Counter

stop_words = set(stopwords.words("english"))

traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERTRegressor(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(self.bert.config.hidden_size, len(traits))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.regressor(pooled)

class PersonalityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, scaler=None, fit_scaler=False):
        self.ids = []
        self.texts = []
        ys = []
        for k, text in texts.items():
            if k in labels:
                self.ids.append(k)
                self.texts.append(text)
                ys.append([labels[k][t] for t in traits])
        ys = np.array(ys)
        self.scaler = scaler or MinMaxScaler()
        if fit_scaler:
            ys = self.scaler.fit_transform(ys)
        else:
            ys = self.scaler.transform(ys)
        self.y = torch.tensor(ys, dtype=torch.float32)
        self.encodings = tokenizer(self.texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.y[idx]
        return item
    
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

def train_model():
    texts_train, labels_train, texts_val, labels_val = load_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = PersonalityDataset(texts_train, labels_train, tokenizer, fit_scaler=True)
    val_ds = PersonalityDataset(texts_val, labels_val, tokenizer, scaler=train_ds.scaler)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(train_ds.scaler, f)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    model = BERTRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = nn.MSELoss()
    best_val_loss, wait, patience = float("inf"), 0, 3

    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Train loss: {total_loss / len(train_dl):.4f}")

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids, attention_mask)
                val_preds.append(out.cpu().numpy())
                val_true.append(labels.cpu().numpy())
        val_loss = mean_squared_error(np.vstack(val_true), np.vstack(val_preds))
        print(f"Val loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), "db3_best_model.pt")
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

def extract_top_words():
    with open("../transcription_validation.pkl", "rb") as f:
        transcripts_val = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = BERTRegressor()
    model.load_state_dict(torch.load("db3_best_model.pt", map_location=device))
    model.eval()

    all_tokens = []
    all_trait_words = {trait: [] for trait in traits}

    def interpret(text, trait_index):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        embeddings = model.bert.embeddings(input_ids)

        def forward_func(embeddings):
            outputs = model.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]
            return model.regressor(pooled)[:, trait_index]

        ig = IntegratedGradients(forward_func)
        attributions, _ = ig.attribute(embeddings, return_convergence_delta=True)
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        scores = attributions.detach().numpy()
        return list(zip(input_tokens, scores))

    def clean_token(token):
        return token.replace("Ġ", "").replace("##", "").lower()

    for text in tqdm(transcripts_val.values(), desc="Extracting"):
        for i, trait in enumerate(traits):
            tokens_scores = interpret(text, i)
            filtered = [
                (clean_token(t), s) for t, s in tokens_scores
                if t not in tokenizer.all_special_tokens and clean_token(t) not in stop_words and clean_token(t).isalpha()
            ]
            all_tokens.extend([tok for tok, _ in filtered])
            sorted_tokens = sorted(filtered, key=lambda x: abs(x[1]), reverse=True)
            top_tokens = [tok for tok, _ in sorted_tokens[:20]]
            all_trait_words[trait].extend(top_tokens)

    most_common = {w for w, _ in Counter(all_tokens).most_common(10)}
    final_trait_words = {
        trait: [w for w in words if w not in most_common][:5] for trait, words in all_trait_words.items()
    }

    def format_hints(trait_words):
        return "\n".join(
            f"Words associated with high {trait}: {', '.join(words)}." for trait, words in trait_words.items()
        )

    hint_text = format_hints(final_trait_words)

    with open("trait_hints.txt", "w") as f:
        f.write(hint_text)

    print("Подсказки сохранены в 'trait_hints.txt':")
    print(hint_text)

if __name__ == '__main__':
    train_model()
    extract_top_words()