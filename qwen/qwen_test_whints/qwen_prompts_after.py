import json
import re
import numpy as np
import pickle
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

random.seed(42)
np.random.seed(42)

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cuda")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

with open("./b5_text_files/transcription_training.pkl", "rb") as f:
    transcripts_train = pickle.load(f)
with open("./b5_text_files/annotation_training.pkl", "rb") as f:
    raw_labels_train = pickle.load(f, encoding="latin1")
with open("./b5_text_files/transcription_test.pkl", "rb") as f:
    transcripts_test = pickle.load(f)
with open("./b5_text_files/annotation_test.pkl", "rb") as f:
    raw_labels_test = pickle.load(f, encoding="latin1")

def convert_labels(raw):
    out = {}
    for trait, trait_dict in raw.items():
        for clip_id, value in trait_dict.items():
            out.setdefault(clip_id, {})[trait] = value
    return out

true_scores_train = convert_labels(raw_labels_train)
true_scores_test = convert_labels(raw_labels_test)

text_hints = {}
with open("text_hints.txt", "r", encoding="utf-8") as f:
    for line in f:
        if ":" in line:
            trait, words = line.split(":", 1)
            key = trait.strip().lower()
            text_hints[key] = [w.strip() for w in words.split(",")]

traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

def format_hints():
    return "\n".join(
        f"{trait.capitalize()}: {', '.join(text_hints.get(trait, []))}"
        for trait in traits
    )

def zero_shot_prompt(text):
    return f"""
Evaluate the personality of the person based on the following spoken text using the Big Five personality traits,
in the following order: openness, conscientiousness, extraversion, agreeableness, and neuroticism.

Here are some indicative words for each trait:
{format_hints()}

Text:
\"{text}\"

Return your answer strictly in JSON format with keys: {', '.join(traits)}.
JSON:
"""

EXAMPLE_ID = sorted(list(transcripts_train.keys()))[0]
EXAMPLE_TEXT = transcripts_train[EXAMPLE_ID]
EXAMPLE_LABELS = true_scores_train[EXAMPLE_ID]
example_json = json.dumps(EXAMPLE_LABELS, indent=2)

def one_shot_prompt(text):
    return f"""
Evaluate the personality of the person based on the following spoken text using the Big Five personality traits,
in the following order: openness, conscientiousness, extraversion, agreeableness, and neuroticism.

Here are some indicative words for each trait:
{format_hints()}

Here is an example:
Text:
\"{EXAMPLE_TEXT}\"
JSON:
{example_json}

Now evaluate the following:
\"{text}\"

Return your answer strictly in JSON format with keys: {', '.join(traits)}.
JSON:
"""

def safe_float(val):
    try:
        if isinstance(val, dict):
            val = list(val.values())[0]
        return float(val)
    except:
        return None

def run_experiment(prompt_func, transcripts, runs=5):
    results = {}
    for run in range(runs):
        for clip_id, text in transcripts.items():
            prompt = prompt_func(text)
            out = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
            match = re.search(r"\{.*\}", out, re.DOTALL)
            if not match:
                continue
            parsed = json.loads(match.group())
            results.setdefault(clip_id, []).append(parsed)
    averaged = {}
    for clip_id, runs in results.items():
        avg = {}
        for trait in traits:
            vals = [safe_float(r.get(trait)) for r in runs if safe_float(r.get(trait)) is not None]
            if vals:
                avg[trait] = float(np.mean(vals))
        averaged[clip_id] = avg
    return averaged

def evaluate(predicted, true):
    y_true, y_pred = [], []
    for cid, preds in predicted.items():
        if cid in true:
            for trait in traits:
                pv = safe_float(preds.get(trait))
                if pv is not None:
                    y_true.append(true[cid][trait])
                    y_pred.append(pv)
    return mean_accuracy(y_true, y_pred), concordance_correlation_coefficient(y_true, y_pred)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("Zero-shot with hints")
preds_zero = run_experiment(zero_shot_prompt, transcripts_test, runs=5)
save_json(preds_zero, "qwen25_zero_with_hints.json")

print("One-shot with hints")
preds_one = run_experiment(one_shot_prompt, transcripts_test, runs=5)
save_json(preds_one, "qwen25_one_with_hints.json")
