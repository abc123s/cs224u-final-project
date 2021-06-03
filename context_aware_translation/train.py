# heavily inspired by this guide: 
# https://towardsdatascience.com/how-to-train-an-mt5-model-for-translation-with-simple-transformers-30ba5fa66c5f

import os
import subprocess
import json
from datetime import datetime

import torch
from simpletransformers.t5 import T5Model, T5Args

from preprocess import preprocess


params = {
    # model information
    "MODEL_TYPE": "mt5",
    "MODEL_NAME": "google/mt5-base",
    
    # training data
    "CORPUS": "four_way_parallel_corpus",
    "LANGUAGE_PAIR": "en-ja",
    "CONTEXT_TYPE": "2-to-2",
    "BREAK_TOKEN": "<break>",

    # training
    "MAX_SEQ_LENGTH": 128,
    "EPOCHS": 1,
    "BATCH_SIZE": 8,
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

# preprocess data
train_df = preprocess(params["CORPUS"], params["LANGUAGE_PAIR"], 'train', params["CONTEXT_TYPE"], params["BREAK_TOKEN"])
eval_df = preprocess(params["CORPUS"], params["LANGUAGE_PAIR"], 'eval', params["CONTEXT_TYPE"], params["BREAK_TOKEN"])

# set up model
model_args = T5Args()
model_args.max_seq_length = params["MAX_SEQ_LENGTH"]
model_args.train_batch_size = params["BATCH_SIZE"]
model_args.eval_batch_size = params["BATCH_SIZE"]
model_args.num_train_epochs = params["EPOCHS"]
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 10000
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.fp16 = False
model_args.output_dir = experiment_dir
model_args.best_model_dir = experiment_dir + "/best_model"
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.preprocess_inputs = False
model_args.num_return_sequences = 1
model_args.tensorboard_dir = experiment_dir + "/logs"
model_args.special_tokens_list = [params["BREAK_TOKEN"]]

model = T5Model(
    params["MODEL_TYPE"],
    params["MODEL_NAME"],
    args=model_args,
    use_cuda = torch.cuda.is_available()
)

# train model
model.train_model(train_df, eval_data=eval_df)
