
# plot 3x4 grid for metric
import matplotlib.pyplot as plt
import numpy as np
import pickle
from palettable.colorbrewer.qualitative import Dark2_4
colors = Dark2_4.mpl_colors

cache_path = "results-molecules/history_cache.pkl"
with open(cache_path, 'rb') as f:
    history_cache = pickle.load(f)


names = ["0", "1", "2", "3", "4", "5"]

def run_name(algo,name):
    return f"{algo}-{name}"

score_dict = {}
algos = ["ours", "ddpo", "dpok", "d3po"]

##################### load dict ######################
for algo in algos:
    for name in names:
        run_name_ = run_name(algo,name)
        
        score_ = history_cache[run_name_]["train/raw_score_mean"]
        score_k = history_cache[run_name_][score_.notnull()]["epoch"].values.tolist()
        score_ = score_[score_.notnull()].values

        if algo == "d3po":
            # duplicate the scores for d3po because it requires 2 batch queries per epoch
            score_ = np.repeat(score_, 2)

        score_v = score_.tolist()
        score_dict[run_name_] = dict(zip(score_k, score_v))
##################### load dict ######################

task_name = [
    "BSD_ASPTE",
    "GLMU_STRPN",
    "GRK4_HUMAN",
    "GSTP1_HUMAN",
    "GUX1_HYPJE",
    "HDAC8_HUMAN",
]


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))  # Adjust figsize as needed
figure_x = arr = np.arange(50) # budget 50 batch queries

algo_label_map = {
    "ours": "Fast Direct (ours)",
    "ddpo": "DDPO",
    "dpok": "DPOK",
    "d3po": "D3PO",
}

for row in range(2):
    for col in range(3):
        index = row * 3 + col
        ax = axes[row, col]
        for algo_i, algo in enumerate(algos):
            run_name_ = run_name(algo,names[index])
            
            y = [
                score_dict[run_name_][x]
                for x in figure_x
                if x in score_dict[run_name_]    
            ]
            label = algo_label_map[algo] if index == 0 else None
            ax.plot(figure_x+1, y, label=label, color=colors[algo_i])
            ax.set_xlim(0, 50)
        
        ax.grid(linewidth=0.5, linestyle='--', alpha=0.5)
        ax.set_title(f"Task-{index+1} ({task_name[index]})")

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Batch Queries', ha='center', fontsize=16)
text_y = fig.text(-0.1 / fig.get_figwidth(), 0.5, 'Vina Score (kcal/mol)', va='center', rotation='vertical', fontsize=16)

legend = fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig("results-molecules/figure.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y))


###################### plot accumanlative ######################

long_names = ["0", "1"]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))  # Adjust figsize as needed
for index in range(len(long_names)):
    ax = axes[index]

    ours_y = [ score_dict[run_name("ours",long_names[index])][x] for x in np.arange(50) ]
    ours_y_accum = np.minimum.accumulate(np.asarray(ours_y))
    ours_y_min = ours_y_accum[-1]

    for algo_i, algo in enumerate(algos):
        run_name_ = run_name(algo,long_names[index])

        figure_x = np.arange(200) if algo != "ours" else np.arange(50)

        y = [ score_dict[run_name_][x] for x in figure_x ]
        y_acum = np.minimum.accumulate(np.asarray(y))
        label = algo_label_map[algo] if index == 0 else None
        ax.step(figure_x+1, y_acum, label=label, color=colors[algo_i], where='post', linewidth=1.0)
        ax.set_xlim(0, len(figure_x))
        ax.set_ylim(ours_y_min, 0.0)

        if algo != "ours":
            baseline_min = np.asarray(y).min()
            our_id_surpass_baseline = np.where(ours_y_accum <= baseline_min)[0][0]

            print(f"Task-{index+1} ({long_names[index]}) ours surpass baseline {algo} at {our_id_surpass_baseline+1} batch queries, speed up factor is {200/(our_id_surpass_baseline+1)}")

            # horizontal line
            ax.plot(
                [our_id_surpass_baseline+1, 200],
                [baseline_min, baseline_min],
                color=colors[algo_i],
                linewidth=0.5,
                linestyle='--'
            )

            # vertical line
            ax.plot(
                [our_id_surpass_baseline+1, our_id_surpass_baseline+1],
                [baseline_min, ours_y_min],
                color=colors[algo_i],
                linewidth=0.5,
                linestyle='--'
            )

    ax.set_title(f"Task-{index+1} ({task_name[index]})")

text_x = fig.text(0.5, -0.1 / fig.get_figheight(), 'Number of Batch Queries', ha='center', fontsize=14)
text_y = fig.text(-0.1 / fig.get_figwidth(), 0.5, 'Accumalated Vina Score', va='center', rotation='vertical', fontsize=14)

legend = fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.6 / fig.get_figheight()))
plt.tight_layout()
plt.savefig("results-molecules/figure_acum.jpeg", dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,text_x,text_y))

###################### plot accumanlative ######################