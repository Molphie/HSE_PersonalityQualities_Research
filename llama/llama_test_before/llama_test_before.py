import json
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

random.seed(42)
np.random.seed(42)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
            if clip_id not in out:
                out[clip_id] = {}
            out[clip_id][trait] = value
    return out

true_scores_train = convert_labels(raw_labels_train)
true_scores_test = convert_labels(raw_labels_test)

# OCEAN
traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

def zero_shot_prompt(text):
    return f"""
Evaluate the personality of the person based on the following spoken text using the Big Five personality traits,
in the following order: openness, conscientiousness, extraversion, agreeableness, and neuroticism.

Rate each trait from 0 to 1 (floating-point numbers).

Return your answer strictly in JSON format with these exact keys.

Text:
\"{text}\"

JSON:
"""

EXAMPLE_ID = sorted(list(transcripts_train.keys()))[0]  # выбираем первый ID (для воспроизводимости)
EXAMPLE_TEXT = transcripts_train[EXAMPLE_ID]
EXAMPLE_LABELS = true_scores_train[EXAMPLE_ID]

example_json = json.dumps(EXAMPLE_LABELS, indent=2)

def one_shot_prompt(text):
    return f"""
Evaluate the personality of the person based on the following spoken text using the Big Five personality traits,
in the following order: openness, conscientiousness, extraversion, agreeableness, and neuroticism.

Each trait must be rated from 0 to 1 (floating-point numbers).

Here is an example:
Text:
\"{EXAMPLE_TEXT}\"
JSON:
{example_json}

Now evaluate the following:
\"{text}\"

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
    results_by_clip = {}
    for run in range(runs):
        print(f"Run {run+1}/{runs}")
        for clip_id, text in transcripts.items():
            prompt = prompt_func(text)
            try:
                out = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]["generated_text"]
                match = re.search(r"\{.*\}", out, re.DOTALL)
                if not match:
                    continue
                parsed = json.loads(match.group())
                if clip_id not in results_by_clip:
                    results_by_clip[clip_id] = []
                results_by_clip[clip_id].append(parsed)
            except Exception as e:
                print(f"{clip_id}: {e}")

    averaged_scores = {}
    for clip_id, runs in results_by_clip.items():
        avg = {}
        for trait in traits:
            vals = [safe_float(run.get(trait)) for run in runs if trait in run and safe_float(run.get(trait)) is not None]
            if vals:
                avg[trait] = float(np.mean(vals))
        averaged_scores[clip_id] = avg

    return averaged_scores

def evaluate(predicted_scores, true_scores):
    all_y_true = []
    all_y_pred = []
    for clip_id in predicted_scores:
        if clip_id in true_scores:
            for trait in traits:
                if trait in predicted_scores[clip_id] and trait in true_scores[clip_id]:
                    pred_val = safe_float(predicted_scores[clip_id][trait])
                    if pred_val is not None:
                        all_y_true.append(true_scores[clip_id][trait])
                        all_y_pred.append(pred_val)
    return mean_accuracy(all_y_true, all_y_pred), concordance_correlation_coefficient(all_y_true, all_y_pred)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

print("Running zero-shot on test")
preds_zero = run_experiment(zero_shot_prompt, transcripts_test, runs=5)
save_json(preds_zero, "llama25_zero_test.json")

print("Running one-shot on test")
preds_one = run_experiment(one_shot_prompt, transcripts_test, runs=5)
save_json(preds_one, "llama25_one_test.json")
