import json
import os

import sacrebleu
import pandas as pd

import torch
from simpletransformers.t5 import T5Model, T5Args

from preprocess import preprocess

# helper function to calculate bleu score ignoring context
# sentence (if provided) and removing separator tokens
def target_sentence_bleu(model_translations, gold_translations, break_token):
    # remove context sentence (if provided) and
    # separator tokens from model translations
    # since we only want to evaluate the quality of the (second)
    # target sentence translation, not the (first) context sentence
    target_model_translations = [
        model_translation.split(break_token)[-1]
        for model_translation in model_translations
    ]

    # remove context sentence (if provided) and
    # separator tokens from gold translations
    # since we only want to evaluate the quality of the (second)
    # target sentence translation, not the (first) context sentence
    target_gold_translations = [
        [
            gold_translation.split(break_token)[-1]
            for gold_translation in gold_translation_option
        ]
        for gold_translation_option in gold_translations
    ]

    return sacrebleu.corpus_bleu(target_model_translations, target_gold_translations)

eval_params = {
    # where to output results of evaluation
    "OUTPUT_PATH": "./experiments/20210601_1917_6c01bf8/eval/",

    # which model to evaluate
    "MODEL_DIR": "./experiments/20210601_1917_6c01bf8/best_model",
    "TRAINING_PARAMS_PATH": "./experiments/20210601_1917_6c01bf8/params.json",

    # eval data
    "CORPUS": "four_way_parallel_corpus",
    "LANGUAGE_PAIR": "en-ja",
    "CONTEXT_TYPE": "random-context",

    # eval specifics
    "MAX_SEQ_LENGTH": 256,
    "LENGTH_PENALTY": 1,
    "BEAM_WIDTH": 10,
}

# load tokenizer, model, and training params
with open(eval_params["TRAINING_PARAMS_PATH"], "r") as f:
    training_params = json.load(f)

with open(eval_params["MODEL_DIR"] + "/model_args.json", "r") as f:
    model_params = json.load(f)

with open(eval_params["MODEL_DIR"] + "/tokenizer_config.json", "r") as f:
    tokenizer_params = json.load(f)

# load trained model
model_args = T5Args()
model_args.max_length = eval_params["MAX_SEQ_LENGTH"]
model_args.length_penalty = eval_params["LENGTH_PENALTY"]
model_args.num_beams = eval_params["BEAM_WIDTH"]

model = T5Model(
    model_params["model_type"],
    eval_params["MODEL_DIR"],
    args=model_args,
    use_cuda = torch.cuda.is_available()
)

# load in evaluation data
eval_df = preprocess(eval_params["CORPUS"], eval_params["LANGUAGE_PAIR"], 'eval', eval_params["CONTEXT_TYPE"], training_params["BREAK_TOKEN"])
source_sentences = eval_df["input_text"].tolist()
gold_translations = [eval_df["target_text"].tolist()]

# predict using trained model
model_translations = model.predict(source_sentences)

# calculate bleu score
bleu_score = sacrebleu.corpus_bleu(model_translations, gold_translations)

# calculate bleu score without context sentence (and sep tokens)
contextless_bleu_score = target_sentence_bleu(model_translations, gold_translations, training_params["BREAK_TOKEN"])

# save down results
if not os.path.isdir(eval_params["OUTPUT_PATH"]):
    os.makedirs(eval_params["OUTPUT_PATH"])

with open(eval_params["OUTPUT_PATH"] + eval_params["CONTEXT_TYPE"] + ".json", "w") as f:
    json.dump({
        **eval_params,
        "raw_bleu_score": bleu_score.score,
        "raw_bleu_score_signature": bleu_score.format(),
        "target_bleu_score": contextless_bleu_score.score,
        "target_bleu_score_signature": contextless_bleu_score.format(),
    }, f, indent=4)
