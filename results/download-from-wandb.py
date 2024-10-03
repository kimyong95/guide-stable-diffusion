algo = ["ddpo", "dpok", "d3po"]

import wandb
import pickle
import os

api = wandb.Api()

wandb_path = "kimyong95/finetune-stable-diffusion"

def get_latest_run_id(path, name):
    runs = api.runs(
        path=path,
        filters={"displayName": name},  
    )
    return runs[0].id

algos = ["ddpo", "dpok", "d3po"]

# exclude "logo" for even number 4x3 grid
names = ["cyberdog", "puppynose", "robotplant", "ocean", "sandglass", "penguin", "basket", "icecube", "catbutterfly", "trafficlight", "deerelephant", "apple"]

# 500 epochs
# names_long = ["logo", "trafficlight", "deerelephant", "apple"]

def run_name(algo,name):
    return f"{algo}-{name}"

id_dict = {}
for algo in algos:
    id_dict[algo] = {}
    for name in names:
        id_dict[run_name(algo,name)] = get_latest_run_id(wandb_path, run_name(algo,name))

# load cache
cache_path = "results/history_cache.pkl"
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
        wandb_run = api.run(f"{wandb_path}/{id_dict[_run_name]}")
        if _run_name not in history_cache \
        or _run_name in redownload \
        or wandb_run.lastHistoryStep > len(history_cache[_run_name]) \
        or wandb_run.summary["_timestamp"] != history_cache[_run_name]["_timestamp"].iloc[-1]:
            print(f"Downloading {_run_name}")
            wandb_run.files()
            download_dir = f"./results/wandb-images/{_run_name}"
            for f in wandb_run.files():
                if f.name.startswith("media/images/validation/") and not os.path.exists(f"{download_dir}/{f.name}"):
                    f.download(download_dir)
            history = wandb_run.history(wandb_run.lastHistoryStep+1)
            history_cache[_run_name] = history
            is_save_cache = True

######################## OURS ########################

name_to_run_map = {
    "cyberdog": "sz2euatc",
    "puppynose": "cwp7bwi2",
    "robotplant": "zay28nta",
    "ocean": "fhmn84jo",
    "sandglass": "tyxrdnsx",
    "penguin": "810j06g7",
    "basket": "752mhm4j",
    "catbutterfly": "2k6h2moi",
    "icecube": "xqizprv7",
    "logo": "oex6316w",
    "trafficlight": "rs1vlelu",
    "deerelephant": "dzvajzb6",
    "apple": "x67p4db8",
}

for name in names:
    _run_name = run_name("ours",name)
    if _run_name not in history_cache or _run_name in redownload:
        wandb_run = api.run(f"{wandb_path}/{name_to_run_map[name]}")
        history = wandb_run.history(wandb_run.lastHistoryStep+1)
        history_cache[_run_name] = history
        is_save_cache = True

######################## OURS ########################


if is_save_cache:
    with open(cache_path, 'wb') as f:
        pickle.dump(history_cache, f)