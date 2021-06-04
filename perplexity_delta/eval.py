import math
import os
import subprocess
import json
from datetime import datetime

import torch
from transformers import GPT2LMHeadModel, AutoModelForCausalLM, GPT2TokenizerFast, T5TokenizerFast, AutoTokenizer
from tqdm import tqdm

from preprocess import preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"

language = "fr"

language_models = {
    "en": "gpt2-medium",
    "ja": "rinna/japanese-gpt2-medium",
    "es": "DeepESP/gpt2-spanish-medium",
    "fr": 'antoiloui/belgpt2'
}

params = {
    "LANGUAGE": language,
    "MODEL": language_models[language],
}

# make experiment directory and save experiment params down
date_string = datetime.now().strftime("%Y%m%d_%H%M")
commit_string = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")
experiment_dir = "experiments/" + date_string + "_" + commit_string
os.mkdir(experiment_dir)
os.mkdir(experiment_dir + "/logs")

# save params down
with open(experiment_dir + "/params.json", "w") as f:
    json.dump(params, f, indent=4)

# load pre-trained causal language model
if params["LANGUAGE"] == "en":
    tokenizer = GPT2TokenizerFast.from_pretrained(params["MODEL"])
    model = GPT2LMHeadModel.from_pretrained(params["MODEL"]).to(device)
elif params["LANGUAGE"] == "ja":
    tokenizer = T5TokenizerFast.from_pretrained(params["MODEL"])
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    model = AutoModelForCausalLM.from_pretrained(params["MODEL"]).to(device)
else:
    tokenizer = AutoTokenizer.from_pretrained(params["MODEL"])
    model = AutoModelForCausalLM.from_pretrained(params["MODEL"]).to(device)

# compute perplexity delta on selected corpus
examples = preprocess('four_way_parallel_corpus', params["LANGUAGE"], 'eval')

perplexities_with_context = []
perplexities_without_context =[]
perplexity_deltas = []
percent_perplexity_deltas = []
for i in tqdm(range(len(examples))):
    context_sentence, target_sentence = examples[i]
    context_encoding = tokenizer(context_sentence).input_ids
    target_encoding = tokenizer(target_sentence).input_ids

    # compute perplexity with context
    input_ids_with_context = torch.tensor([[*context_encoding, *target_encoding]]).to(device)
    target_ids_with_context = input_ids_with_context.clone()
    # ignore loss from context sentence tokens
    target_ids_with_context[:,:-len(target_encoding)] = -100

    with torch.no_grad():
        loss_with_context = model(input_ids_with_context, labels=target_ids_with_context)[0]
        # perplexity is the exponentiation of the cross-entropy loss
        perplexity_with_context = torch.exp(loss_with_context).item()

    # compute perplexity without context
    input_ids_without_context = torch.tensor([[*target_encoding]]).to(device)
    target_ids_without_context = input_ids_without_context.clone()

    with torch.no_grad():
        loss_without_context = model(input_ids_without_context, labels=target_ids_without_context)[0]
        # perplexity is the exponentiation of the cross-entropy loss
        perplexity_without_context = torch.exp(loss_without_context).item()

    if math.isfinite(perplexity_with_context) and math.isfinite(perplexity_without_context):
        perplexities_with_context.append(perplexity_with_context)
        perplexities_without_context.append(perplexity_without_context)

        perplexity_deltas.append(perplexity_without_context - perplexity_with_context)
        percent_perplexity_deltas.append((perplexity_without_context - perplexity_with_context) / perplexity_without_context)



def corpus_perplexity(perplexities):
    log_perplexities = [math.log(perplexity) for perplexity in perplexities]
    average_log_perplexity = sum(log_perplexities) / len(log_perplexities)
    
    return math.exp(average_log_perplexity)

corpus_perplexity_with_context = corpus_perplexity(perplexities_with_context)
corpus_perplexity_without_context = corpus_perplexity(perplexities_without_context)
average_perplexity_with_context = sum(perplexities_with_context) / len(perplexities_with_context)
average_perplexity_without_context = sum(perplexities_without_context) / len(perplexities_without_context)
average_perplexity_deltas = sum(perplexity_deltas) / len(perplexity_deltas)
average_percent_perplexity_deltas = sum(percent_perplexity_deltas) / len(percent_perplexity_deltas)

# save results
# save params down
with open(experiment_dir + "/results.json", "w") as f:
    json.dump({
        "CORPUS_PERPLEXITY_WITH_CONTEXT": corpus_perplexity_with_context,
        "CORPUS_PERPLEXITY_WITHOUT_CONTEXT": corpus_perplexity_without_context,
        "AVERAGE_PERPLEXITY_WITH_CONTEXT": average_perplexity_with_context,
        "AVERAGE_PERPLEXITY_WITHOUT_CONTEXT": average_perplexity_without_context,
        "AVERAGE_PERPLEXITY_DELTA": average_perplexity_deltas,
        "AVERAGE_PERCENT_PERPLEXITY_DELTA": average_percent_perplexity_deltas,
    }, f, indent=4)
