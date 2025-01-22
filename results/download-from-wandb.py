import wandb
import pickle
import os

api = wandb.Api()

wandb_path = "kimyong95/guide-stable-diffusion"
dir_path = os.path.dirname(os.path.realpath(__file__))


def run_name(algo,name):
    
    if algo == "ours":
        # simple reward
        if name in ["compress", "incompress", "aesthetic"]:
            return f"finetune-with-gp-{name}-edm"
        # ablation study
        elif name.startswith("ablation"):
            return f"finetune-with-gp-prompt=deerelephant-{name}"
        # prompts
        else:
            return f"finetune-with-gp-prompt={name}"
    if algo == "ours-ddim":
        # simple reward
        if name in ["compress", "incompress", "aesthetic"]:
            return f"finetune-with-gp-{name}-ddim"
        # prompts
        else:
            return f"finetune-with-gp-prompt={name}-ddim"
    else:
        return f"{algo}-{name}"

def cache_key(algo,name):
    return f"{algo}-{name}"

def get_latest_run_id(path, name):
    runs = api.runs(
        path=path,
        filters={"displayName": name, "tags": { "$ne": "drop" }},
    )
    if len(runs) == 0:
        print(f"Run [{name}] not found.")
        return None
    else:
        return runs[0].id

algos = ["ddpo", "dpok", "d3po", "dno", "ours", "ours-ddim"]

names = ["cyberdog", "puppynose", "robotplant", "ocean", "sandglass", "penguin", "basket", "icecube", "catbutterfly", "trafficlight", "deerelephant", "apple",
    "compress", "incompress", "aesthetic",         
]

sr_names = ["compress", "incompress", "aesthetic"]

# load cache
cache_path = f"{dir_path}/wandb_cache.pkl"
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        wandb_cache = pickle.load(f)
else:
    wandb_cache = {}

def update_cache(cache_key, wandb_run, run_id):
    if cache_key in wandb_cache:
        print(f"Updating cache [{cache_key}] ...")
    else:
        print(f"Adding cache [{cache_key}] ...")
    
    history = wandb_run.history(wandb_run.lastHistoryStep+1)
    wandb_cache[cache_key] = {
        "history": history,
        "run_id": run_id,
        "run_name": wandb_run.name,
        "state": wandb_run.state,
    }

def is_update_cache(cache_key, wandb_run, run_id):
    redownload = []
    is_update = (cache_key not in wandb_cache \
        or wandb_cache[cache_key]["run_id"] != run_id \
        or cache_key in redownload \
        or wandb_run.lastHistoryStep > len(wandb_cache[cache_key]["history"]) \
        or int(wandb_run.summary["_timestamp"]) != int(wandb_cache[cache_key]["history"]["_timestamp"].iloc[-1])
    )

    return is_update

is_save_cache = False
for algo in algos:
    for name in names:
        _cache_key = cache_key(algo,name)
        _run_id = get_latest_run_id(wandb_path, run_name(algo,name))
        _wandb_run = api.run(f"{wandb_path}/{_run_id}")
        if is_update_cache(_cache_key, _wandb_run, _run_id):
            update_cache(_cache_key, _wandb_run, _run_id)
            is_save_cache = True

######################## ablation ########################
ablation_names = [
    "ablation-a=20", "ablation-a=40", "ablation-a=60", "ablation-a=100", "ablation-a=120", "ablation-a=140", "ablation-a=160", "ablation-a=180", "ablation-a=200", "ablation-a=240", "ablation-a=280", "ablation-a=320",
    "ablation-b=2", "ablation-b=4", "ablation-b=8", "ablation-b=16", "ablation-b=32", "ablation-b=40", "ablation-b=48", "ablation-b=56", "ablation-b=64"
]

for name in ablation_names:
    _cache_key = cache_key("ours",name)
    _run_id = get_latest_run_id(wandb_path, run_name("ours",name))
    _wandb_run = api.run(f"{wandb_path}/{_run_id}")
    if is_update_cache(_cache_key, _wandb_run, _run_id):
        update_cache(_cache_key, _wandb_run, _run_id)
        is_save_cache = True
######################## ablation ########################

########################## DNO ###########################
prompt_names = list(set(names) - set(sr_names))
dno_prompt_names = [f"{name}-index={i}" for name in prompt_names for i in range(16)]
dno_sr_names = [f"{name}-index={i}" for name in sr_names for i in range(45)]
dno_names = dno_prompt_names + dno_sr_names
for name in dno_names:
    _cache_key = cache_key("dno",name)
    _run_id = get_latest_run_id(wandb_path, run_name("dno", name))

    if _run_id is None:
        continue

    _wandb_run = api.run(f"{wandb_path}/{_run_id}")
    if is_update_cache(_cache_key, _wandb_run, _run_id):
        update_cache(_cache_key, _wandb_run, _run_id)
        is_save_cache = True

########################## DNO ###########################

if is_save_cache:
    with open(cache_path, 'wb') as f:
        pickle.dump(wandb_cache, f)