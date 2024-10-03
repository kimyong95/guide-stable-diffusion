algo = ["ddpo", "dpok", "d3po"]

import wandb
import pickle
import os

api = wandb.Api()

wandb_path = "kimyong95/finetune-targetdiff"

def get_latest_run_id(path, name):
    runs = api.runs(
        path=path,
        filters={"displayName": name},  
    )
    return runs[0].id

algos = ["ddpo", "dpok", "d3po", "ours"]

names = ["0", "1", "2", "3", "4", "5"]

def run_name(algo,name):
    if algo == "ours":
        return f"finetune-targetdiff-ddim-data={name}"
    else:
        return f"{algo}-data={name}"

def key_name(algo,name):
    return f"{algo}-{name}"

id_dict = {}
for algo in algos:
    id_dict[algo] = {}
    for name in names:
        id_dict[run_name(algo,name)] = get_latest_run_id(wandb_path, run_name(algo,name))

# load cache
cache_path = "results-molecules/history_cache.pkl"
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        history_cache = pickle.load(f)
else:
    history_cache = {}

redownload = []

is_save_cache = False
for algo in algos:
    for name in names:
        _run_name = run_name(algo,name)
        _key_name = key_name(algo,name)
        wandb_run = api.run(f"{wandb_path}/{id_dict[_run_name]}")
        if _key_name not in history_cache \
        or _run_name in redownload \
        or wandb_run.lastHistoryStep > len(history_cache[_key_name]) \
        or wandb_run.summary["_timestamp"] != history_cache[_key_name]["_timestamp"].iloc[-1]:
            print(f"Downloading {_run_name}")
            history = wandb_run.history(wandb_run.lastHistoryStep+1)
            history_cache[_key_name] = history
            is_save_cache = True

if is_save_cache:
    with open(cache_path, 'wb') as f:
        pickle.dump(history_cache, f)