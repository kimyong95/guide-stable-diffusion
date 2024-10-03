
# plot 3x4 grid for metric
import matplotlib.pyplot as plt
import numpy as np
import pickle
from palettable.colorbrewer.qualitative import Dark2_4
colors = Dark2_4.mpl_colors

cache_path = "results/history_cache.pkl"
with open(cache_path, 'rb') as f:
    history_cache = pickle.load(f)

algos = ["ddpo", "dpok", "d3po"]
names = ["deerelephant", "trafficlight", "apple", "cyberdog", "puppynose", "robotplant", "ocean", "sandglass", "penguin", "basket", "icecube", "catbutterfly"]

def run_name(algo,name):
    return f"{algo}-{name}"

long_runs = ["deerelephant", "trafficlight", "apple"]


score_dict = {}


##################### baselines ######################

for algo in algos:
    for name in names:
        run_name_ = run_name(algo,name)
        
        reward_ = history_cache[run_name_]["train/reward_mean"]
        reward_k = history_cache[run_name_][reward_.notnull()]["epoch"].values.tolist()
        
        # unnormalize to raw score, refer to related_works/d3po/d3po_pytorch/rewards.py
        score_ = reward_[reward_.notnull()].values * 5
        score_v = score_.tolist()
        score_dict[run_name_] = dict(zip(reward_k, score_v))
##################### baselines ######################


######################## OURS ########################
for name in names:
    run_name_ = run_name("ours",name)
    
    score_ = history_cache[run_name_]["train/score_mean"]
    score_k = history_cache[run_name_][score_.notnull()]["epoch"].values.tolist()
    
    # unnormalize to raw score, refer to related_works/d3po/d3po_pytorch/rewards.py
    score_ = score_[score_.notnull()].values * -5
    score_v = score_.tolist()
    score_dict[run_name_] = dict(zip(reward_k, score_v))

######################## OURS ########################

algos = ["ours", "ddpo", "dpok", "d3po"]

algo_label_map = {
    "ours": "Fast Direct (ours)",
    "ddpo": "DDPO",
    "dpok": "DPOK",
    "d3po": "D3PO",
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
    "catbutterfly": "cat-butterfly"
}

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
figure_x = np.arange(50) # budget 50 batch queries
for row in range(3):
    for col in range(4):
        index = row * 4 + col
        ax = axes[row, col]
        for algo_i, algo in enumerate(algos):
            run_name_ = run_name(algo,names[index])

            y = [ score_dict[run_name_][x] for x in figure_x ]
            label = algo_label_map[algo] if index == 0 else None
            ax.plot(figure_x+1, y, label=label, color=colors[algo_i])
            
            ax.set_xlim(0, len(figure_x))
            ax.set_ylim(1, 5.1)
        
        ax.grid(linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_title(f"Task-{index+1} ({name_label_map[names[index]]})")

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Batch Queries', ha='center', fontsize=14)
text_y = fig.text(-0.1 / fig.get_figwidth(), 0.5, 'Gemini Rating', va='center', rotation='vertical', fontsize=14)

legend = fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig("results/figure.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y))


###################### plot accumanlative ######################

long_names = ["deerelephant", "trafficlight", "apple"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))  # Adjust figsize as needed
for index in range(len(long_names)):
    ax = axes[index]

    ours_y = [ score_dict[run_name("ours",long_names[index])][x] for x in np.arange(50) ]
    ours_y_accum = np.maximum.accumulate(np.asarray(ours_y))
    ours_y_max = ours_y_accum[-1]

    for algo_i, algo in enumerate(algos):
        run_name_ = run_name(algo,long_names[index])

        figure_x = np.arange(200) if algo != "ours" else np.arange(50)

        y = [ score_dict[run_name_][x] for x in figure_x ]
        y_acum = np.maximum.accumulate(np.asarray(y))
        label = algo_label_map[algo] if index == 0 else None
        ax.step(figure_x+1, y_acum, label=label, color=colors[algo_i], where='post', linewidth=1.0)
        ax.set_xlim(0, len(figure_x))
        ax.set_ylim(1, 5.1)

        if algo != "ours":
            baseline_max = np.asarray(y).max()
            our_id_surpass_baseline = np.where(ours_y_accum >= baseline_max)[0][0]

            print(f"Task-{index+1} ({long_names[index]}) ours surpass baseline {algo} at {our_id_surpass_baseline+1} batch queries, speed up factor is {200/(our_id_surpass_baseline+1)}")

            # horizontal line
            ax.plot(
                [our_id_surpass_baseline+1, 200],
                [baseline_max, baseline_max],
                color=colors[algo_i],
                linewidth=0.5,
                linestyle='--'
            )

            # vertical line
            ax.plot(
                [our_id_surpass_baseline+1, our_id_surpass_baseline+1],
                [baseline_max, 0],
                color=colors[algo_i],
                linewidth=0.5,
                linestyle='--'
            )

    ax.set_title(f"Task-{index+1} ({name_label_map[long_names[index]]})")

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Batch Queries', ha='center', fontsize=14)
text_y = fig.text(-0.1 / fig.get_figwidth(), 0.5, 'Accumalated Gemini Rating', va='center', rotation='vertical', fontsize=14)

legend = fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig("results/figure_acum.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y))

###################### plot accumanlative ######################