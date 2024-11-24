
# plot 3x4 grid for metric
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import seaborn as sns

colors = sns.color_palette("hls", 8)

color_map = {
    "ours": colors[7],
    "ours-ddim": colors[6],

    "dno":  colors[1],
    "ddpo": colors[2],
    "dpok": colors[3],
    "d3po": colors[4],
}

zorder_map = {
    "ours": 100,
    "ours-ddim": 99,

    "dno":  1,
    "ddpo": 1,
    "dpok": 1,
    "d3po": 1,
}

cache_path = "results/wandb_cache.pkl"
with open(cache_path, 'rb') as f:
    wandb_cache = pickle.load(f)

names = ["deerelephant", "trafficlight", "apple", "cyberdog", "puppynose", "robotplant", "ocean", "sandglass", "penguin", "basket", "icecube", "catbutterfly", \
    "compress", "incompress", "aesthetic",
    "compress-eval", "incompress-eval", "aesthetic-eval",
]

prompt_names = ["deerelephant", "trafficlight", "apple", "cyberdog", "puppynose", "robotplant", "ocean", "sandglass", "penguin", "basket", "icecube", "catbutterfly"]

ablation_a_all_names = [
    "ablation-a=20", "ablation-a=40", "ablation-b=32", "ablation-a=60", "ablation-a=100", "ablation-a=120", "ablation-a=140", "ablation-a=160", "ablation-a=180", "ablation-a=200", "ablation-a=240", "ablation-a=280", "ablation-a=320",
]

# "ablation-b=32" use alpha=80
ablation_a_names = [
    "ablation-a=20", "ablation-a=40", "ablation-b=32", "ablation-a=160", "ablation-a=320",
]

ablation_b_all_names = [
    "ablation-b=2", "ablation-b=4", "ablation-b=8", "ablation-b=16", "ablation-b=32", "ablation-b=40", "ablation-b=48", "ablation-b=56", "ablation-b=64"
]

ablation_b_names = [
    "ablation-b=4", "ablation-b=8", "ablation-b=16", "ablation-b=32", "ablation-b=64",
]

long_names = ["deerelephant", "trafficlight", "apple"]
sr_names = ["compress", "incompress", "aesthetic"]
sr_names_eval = ["compress-eval", "incompress-eval", "aesthetic-eval"]

score_dict = {}

def score_dict_key(algo,name):
    return f"{algo}-{name}"

def cache_key(algo,name):
    _name = name
    if name in sr_names_eval:
        _name = name.replace("-eval","")
    return f"{algo}-{_name}"

##################### load scores ######################
algos = ["ddpo", "dpok", "d3po", "dno", "ours-ddim", "ours"]

score_key_map = {
    "dno": "validation/score_mean",
    "ours": "train/score_mean",
    "ours-ddim": "train/score_mean",
    "ddpo": "train/reward_mean",
    "dpok": "train/reward_mean",
    "d3po": "train/reward_mean",
}


def denormalize_score(algo, name, score):
    if algo in ["ours", "ours-ddim"]:
        if name == "incompress":
            return score * (-1e6 / 1e6)
        elif name == "compress":
            return score * (1e6 / 1e6)
        elif name == "aesthetic":
            return score * -10
        else:
            return score * -5
    elif algo == "dno":
        if name.startswith("incompress"):
            return score * (-1e3 / 1e6)
        elif name.startswith("compress"):
            return score * (1e3 / 1e6)
        elif name.startswith("aesthetic"):
            return score * -1
        else:
            return score * -5
    else:
        if name == "incompress":
            return score * (1e3 / 1e6)
        elif name == "compress":
            return score * (-1e3 / 1e6)
        elif name == "aesthetic":
            return score
        else:
            return score * 5

dno_names = None

for algo in algos:
    if algo == "ours":
        _names = names + ablation_a_all_names + ablation_b_all_names
    elif algo == "dno":
        dno_names = []
        for name in list(set(names) - set(sr_names_eval)):
            if name in sr_names:
                total_index = 45
            else:
                total_index = 16
            dno_names.extend([f"{name}-index={i}" for i in range(total_index)])
        _names = dno_names
    else:
        _names = names
    
    for name in _names:

        _cache_key = cache_key(algo,name)
        _score_key = score_dict_key(algo,name)

        _wandb_key = score_key_map[algo]

        if name in sr_names_eval:
            _wandb_key = _wandb_key.replace("train", "validation")
        
        score_ = wandb_cache[_cache_key]["history"][_wandb_key]
        score_k = wandb_cache[_cache_key]["history"][score_.notnull()]["epoch"].values.tolist()
        
        # unnormalize to raw score, refer to related_works/d3po/d3po_pytorch/rewards.py
        score_ = score_[score_.notnull()].values
        score_ = denormalize_score(algo, name.replace("-eval",""), score_)
        score_v = score_.tolist()
        score_dict[_score_key] = dict(zip(score_k, score_v))

##################### load scores ######################


##################### average dno ######################

for name in list(set(names) - set(sr_names_eval)):
    dno_scores = []

    if name in long_names + ["compress", "incompress", "aesthetic"]:
        epoches = np.arange(500)
    else:
        epoches = np.arange(50)

    if name in sr_names:
        total_index = 45
    else:
        total_index = 16

    for i in range(total_index):
        _score_key = score_dict_key("dno", f"{name}-index={i}")
        scores = [ score_dict[_score_key][e] for e in epoches ]
        dno_scores.append(
            scores
        )
    
    dno_scores = np.stack(dno_scores)
    _score_key = score_dict_key("dno", name)
    score_dict[_score_key] = dno_scores.mean(axis=0)


##################### average dno ######################

algos = ["ours", "ours-ddim", "ddpo", "dpok", "d3po", "dno"]

algo_label_map = {
    "ours": "Fast Direct (ours) w/ EDM Sampler",
    "ours-ddim": "Fast Direct (ours) w/ DDIM Sampler",
    "ddpo": "DDPO",
    "dpok": "DPOK",
    "d3po": "D3PO",
    "dno": "DNO",
}

name_label_map = {
    "deerelephant": "deer-elephant",
    "trafficlight": "traffic-light",
    "apple": "apple",
    "cyberdog": "cyber-dog",
    "puppynose": "puppy-nose",
    "robotplant": "robot-plant",
    "ocean": "ocean",
    "sandglass": "sand-glass",
    "penguin": "penguin",
    "basket": "basket",
    "icecube": "ice-cube",
    "catbutterfly": "cat-butterfly",
    "compress": "Compressibility (↓)",
    "incompress": "Incompressibility (↑)",
    "aesthetic": "Aesthetic Quality (↑)",
    "compress-eval": "Compressibility (Unseen Prompts) (↓)",
    "incompress-eval": "Incompressibility (Unseen Prompts) (↑)",
    "aesthetic-eval": "Aesthetic Quality (Unseen Prompts) (↑)",
}

######################## main plot #########################
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
figure_x = np.arange(50) # budget 50 batch queries
for row in range(3):
    for col in range(4):
        index = row * 4 + col
        ax = axes[row, col]
        _name = prompt_names[index]
        for algo_i, algo in enumerate(algos):
            
            _score_key = score_dict_key(algo,_name)

            y = [ score_dict[_score_key][x] for x in figure_x ]
            label = algo_label_map[algo] if index == 0 else None
            linewidth = 1.5 if algo in ["ours","ours-ddim"] else 1.5
            ax.plot(figure_x+1, y, label=label, color=color_map[algo], linewidth=linewidth, zorder=zorder_map[algo])
            
            ax.set_xlim(0, len(figure_x))
            ax.set_ylim(0.9, 5.1)
        
        ax.grid(linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_title(f"Task-{index+1} ({name_label_map[_name]})")

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Batch Queries', ha='center', fontsize=14)
text_y = fig.text(-0.1 / fig.get_figwidth(), 0.5, 'Gemini Rating', va='center', rotation='vertical', fontsize=14)

legend = fig.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig("results/figure.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y), format='jpeg', pil_kwargs={"quality":50})
######################## main plot #########################



######################## simple reward #########################

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 9))
algos = ["ours", "ours-ddim", "ddpo", "dpok", "d3po", "dno"]
y_label_map = {
    "compress": "JPEG Size (MB)",
    "incompress": "JPEG Size (MB)",
    "aesthetic": "Aesthetic Score",
}
algo_label_map["ours"] = "Fast Direct (ours) /w EDM"
algo_label_map["ours-ddim"] = "Fast Direct (ours) /w DDIM"
for col, sr_names_row in enumerate([sr_names, sr_names_eval]):
    for row, name in enumerate(sr_names_row):
        ax = axes[row][col]
        
        for algo_i, algo in enumerate(algos):

            if name in sr_names_eval and algo == "dno":
                continue

            _score_key = score_dict_key(algo,name)

            if algo in ["ours", "ours-ddim"]:
                figure_x = np.arange(100)
            else:
                figure_x = np.arange(100)

            y = [ score_dict[_score_key][x] for x in figure_x ]
            label = algo_label_map[algo] if (row == 0 and col == 0) else None
            linewidth = 1.5 if algo in ["ours","ours-ddim"] else 1.5
            ax.plot(figure_x+1, y, label=label, color=color_map[algo],linewidth=linewidth, zorder=zorder_map[algo])
            ax.set_ylabel(y_label_map[name.replace("-eval","")])
            
            ax.set_xlim(0, len(figure_x))
            # ax.set_ylim(1, 5.1)
        
        ax.grid(linewidth=0.5, linestyle='--', alpha=0.5)
        _title = name_label_map[name]
        ax.set_title(_title)

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Batch Queries', ha='center', fontsize=14)

legend = fig.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig("results/figure-sr.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y), format='jpeg', pil_kwargs={"quality":50})

######################## simple reward #########################

###################### plot accumanlative ######################

y_range_map = {
    "deerelephant": (1, 5.1),
    "trafficlight": (1, 5.1),
    "apple": (1, 5.1),
    "compress-eval": (0, 0.5),
    "incompress-eval": (0.2, 1.3),
    "aesthetic-eval": (5.25, 6.75),
}

algos = ["ours", "ddpo", "dpok", "d3po", "dno"]
algo_label_map["ours"] = "Fast Direct (ours)"
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))  # Adjust figsize as needed
# for row, sr_names_row in enumerate([long_names, sr_names_eval]):
for row, sr_names_row in enumerate([long_names]):
    for col, name in enumerate(sr_names_row):
        ax = axes[col]
        _score_key = score_dict_key("ours",name)
        ours_y = [ score_dict[_score_key][x] for x in np.arange(50) ]
        ours_y = np.asarray(ours_y)
        if name == "compress-eval":
            ours_y = -ours_y
        ours_y_accum = np.maximum.accumulate(ours_y)
        ours_y_max = ours_y.max()
        ours_y_max_id = ours_y.argmax()

        for algo_i, algo in enumerate(algos):
            
            if algo == "dno" and name in sr_names_eval:
                continue

            _score_key = score_dict_key(algo,name)

            figure_x = np.arange(200) if algo not in ["ours", "ours-ddim"] else np.arange(50)

            y = [ score_dict[_score_key][x] for x in figure_x ]
            y = np.asarray(y)
            if name == "compress-eval":
                y = -y

            y_acum = np.maximum.accumulate(y)
            y_max = y.max()
            y_max_id = y.argmax()
            label = algo_label_map[algo] if (row==0 and col==0) else None
            linewidth = 1.5 if algo in ["ours","ours-ddim"] else 1.5
            
            plot_y = np.maximum.accumulate(y)
            if name == "compress-eval":
                plot_y = -y
            ax.step(figure_x+1, plot_y, label=label, color=color_map[algo], where='post', linewidth=linewidth, zorder=zorder_map[algo])
            ax.set_xlim(0, 200)
            ax.set_ylim(y_range_map[name][0], y_range_map[name][1])

            if algo not in ["ours","ours-ddim"]:
                if ours_y_max >= y_max:
                    our_id_surpass_baseline = np.where(ours_y_accum >= y_max)[0][0]
                    print(f"Task [{name}] ours surpass baseline {algo} at {our_id_surpass_baseline+1} batch queries, speed up factor is {len(y)/(our_id_surpass_baseline+1)}")
                    
                    plot_y_max = y_max
                    if name == "compress-eval":
                        plot_y_max = -plot_y_max

                    # horizontal line
                    ax.plot(
                        [our_id_surpass_baseline+1, y_max_id],
                        [plot_y_max, plot_y_max],
                        color=color_map[algo],
                        linewidth=0.5,
                        linestyle='--'
                    )
                    # vertical line
                    ax.plot(
                        [our_id_surpass_baseline+1, our_id_surpass_baseline+1],
                        [plot_y_max, y_range_map[name][0]],
                        color=color_map[algo],
                        linewidth=0.5,
                        linestyle='--'
                    )
                
                else:

                    baseline_id_surpass_our = np.where(y_acum >= ours_y_max)[0][0]
                    print(f"Task [{name}] {algo} surpass ours at {baseline_id_surpass_our+1} batch queries, speed up factor is {(baseline_id_surpass_our+1)/len(ours_y)}")
                    
                    plot_ours_y_max = ours_y_max
                    if name == "compress-eval":
                        plot_ours_y_max = -ours_y_max
                    # horizontal line
                    ax.plot(
                        [ours_y_max_id, baseline_id_surpass_our],
                        [plot_ours_y_max, plot_ours_y_max],
                        color=color_map[algo],
                        linewidth=0.5,
                        linestyle='--'
                    )
                    # vertical line
                    ax.plot(
                        [baseline_id_surpass_our+1, baseline_id_surpass_our+1],
                        [plot_ours_y_max, y_range_map[name][0]],
                        color=color_map[algo],
                        linewidth=0.5,
                        linestyle='--'
                    )
        _name = name
        if name in sr_names_eval:
            _name = name.replace("-eval","")
        ax.set_title(f"Task [{name_label_map[_name]}]")

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Batch Queries', ha='center', fontsize=14)
text_y = fig.text(-0.1 / fig.get_figwidth(), 0.5, 'Accumalated Gemini Rating', va='center', rotation='vertical', fontsize=14)

legend = fig.legend(loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig("results/figure_acum.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y), format='jpeg', pil_kwargs={"quality":50})


###################### plot accumanlative ######################

###################### plot ablation ######################


figure_x = np.arange(50)  # Budget: 50 batch queries
colors_a = sns.color_palette("rocket_r", len(ablation_a_names))
colors_b = sns.color_palette("rocket_r", len(ablation_b_names))

fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)

# First subplot: Alpha ablation
for name_i, name in enumerate(ablation_a_names):
    alpha = 80 if name == "ablation-b=32" else int(re.search(r"ablation-a=(\d+)", name).group(1))
    _score_key = score_dict_key("ours", name)
    y = [score_dict[_score_key][x] for x in figure_x]
    axs[0].plot(figure_x + 1, y, label=f"α={alpha}", color=colors_a[name_i], linewidth=1)
axs[0].grid(linewidth=0.5, linestyle='--', alpha=0.5)
axs[0].set_title("Step Size", fontsize=14)
axs[0].set_xlabel("Number of Batch Queries")
axs[0].set_ylabel("Gemini Rating")
axs[0].legend(loc='lower right', fontsize=8)

# Second subplot: Batch size ablation
for name_i, name in enumerate(ablation_b_names):
    batchsize = int(re.search(r"ablation-b=(\d+)", name).group(1))
    _score_key = score_dict_key("ours", name)
    y = [score_dict[_score_key][x] for x in figure_x]
    axs[1].plot(figure_x + 1, y, label=f"B={batchsize}", color=colors_b[name_i], linewidth=1)
axs[1].grid(linewidth=0.5, linestyle='--', alpha=0.5)
axs[1].set_title("Batch Size", fontsize=14)
axs[1].set_xlabel("Number of Batch Queries")
axs[1].set_ylabel("Gemini Rating")
axs[1].legend(loc='lower right', fontsize=8)

# Third subplot: Runtime vs Batch Size
batchsize = [int(re.search(r"ablation-b=(\d+)", item).group(1)) for item in ablation_b_names]
runtime_list = []
for name in ablation_b_names:
    _cache_key = cache_key("ours", name)
    runtime_ = wandb_cache[_cache_key]["history"]["_runtime"]
    runtime_k = wandb_cache[_cache_key]["history"][runtime_.notnull()]["epoch"].values.tolist()
    runtime_v = (runtime_[runtime_.notnull()].values / 3600).tolist()
    runtime_list.append(dict(zip(runtime_k, runtime_v))[49])
axs[2].plot(batchsize, runtime_list, marker='o', linestyle=':', color=color_map["ours"], linewidth=1)
axs[2].grid(linewidth=0.5, linestyle='--', alpha=0.5)
axs[2].set_title("Runtime vs Batch Size", fontsize=14)
axs[2].set_xlabel("Batch Size")
axs[2].set_ylabel("Run Time (Hours)")
# Save the figure
plt.tight_layout()
plt.savefig("results/ablation.jpeg", dpi=600, format='jpeg', pil_kwargs={"quality":50})


pass

###################### plot ablation ######################