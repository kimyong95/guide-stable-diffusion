


import glob
import re
import numpy as np
import yaml
import textwrap
import shutil
import os
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

all_images = defaultdict(list)

######################## OURS ########################

name_to_prompts_map = {
    "cyberdog": "A natural fluffy dog talking to a cybertic dog.",
    "puppynose": "Side view of a puppy lying on floor, one butterfly stopping on its nose.",
    "robotplant": "A cute cybernetic robot plants a tree in the forest.",
    "ocean": "A helicopter floating under the ocean.",
    "sandglass": "A transparent glass filled with a mixture of water and sand, with one feather floating inside.",
    "penguin": "A photo realistic photo showing a penguin standing on a very small floating ice, with a tree is on fire in the background.",
    "basket": "Exactly one orange in a basket of apples.",
    "catbutterfly": "A cat with butterfly wings.",
    "icecube": "A glass of water with exactly one ice cube.",
    "logo": "A board saying \"QOBG\".",
    "trafficlight": "A traffic light with yellow at top, green at middle, and red at bottom.",
    "deerelephant": "A yellow reindeer and a blue elephant.",
    "apple": "Seven red apples arranged in a circle.",
}

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

names = ["deerelephant", "trafficlight", "apple", "cyberdog", "puppynose", "robotplant", "ocean", "sandglass", "penguin", "basket", "icecube", "catbutterfly"]

# 
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--id', default=2, type=int, help='choose the i-th out of the 16 validation saved images to compares')

args = parser.parse_args()

demo_id = int(args.id)

def crop_image(image, i,j, grid_size=1024):
    width, height = image.size
    assert width == height
    start_i = i * grid_size + (i+1)*2
    start_j = j * grid_size + (j+1)*2
    image = image.crop((start_i, start_j, start_i + grid_size, start_j + grid_size))
    return image

def concat_images(image_matrix):

    # assert 2D matrix
    assert all(len(row) == len(image_matrix[0]) for row in image_matrix)

    # Concatenate images within each row horizontally
    rows = []
    for image_row in image_matrix:
        widths, heights = zip(*(i.size for i in image_row))
        total_width = sum(widths)
        max_height = max(heights)

        new_row = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in image_row:
            new_row.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        rows.append(new_row)

    # Concatenate rows vertically
    widths, heights = zip(*(i.size for i in rows))
    max_width = max(widths)
    total_height = sum(heights)

    final_concat = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for row in rows:
        final_concat.paste(row, (0, y_offset))
        y_offset += row.size[1]

    return final_concat

def get_validation_image(run_path, epoch):
    image_path = run_path + f"/ablation/validation/{epoch}/l={epoch}.jpg"
    image = Image.open(image_path)
    return image

def get_validation_texts(run_path, epoch):
    text_path = run_path + f"/ablation/validation/{epoch}/llava.txt"
    with open(text_path, "r") as f:
        texts = f.read().splitlines()
    return texts

def get_prompt(run_path):
    config_path = run_path + "/files/config-cli.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["model"]["init_args"]["generate_prompt"]

for index, name in enumerate(names):
    includes_epoches = [0,9,19,29,39,49]

    run_id = name_to_run_map[name]
    paths = glob.glob(f"wandb/run-*-{run_id}")
    assert len(paths) == 1, paths
    path = paths[0]

    # save final steps to results
    final_image_path = f"{path}/ablation/train/{includes_epoches[-1]}/l={includes_epoches[-1]}.jpg"
    os.makedirs("results/ours-final-compress", exist_ok=True)
    Image.open(final_image_path).save(
        f"results/ours-final-compress/{index}.jpeg",
        quality=22,
        optimize=True
    )
    os.makedirs("results/ours-final", exist_ok=True)
    shutil.copy(final_image_path, f"results/ours-final/{index}.jpeg")

    init_image_path = f"{path}/ablation/train/{includes_epoches[-1]}/l=0.jpg"
    os.makedirs("results/ours-init-compress", exist_ok=True)
    Image.open(init_image_path).save(
        f"results/ours-init-compress/{index}.jpeg",
        quality=22,
        optimize=True
    )
    os.makedirs("results/ours-init", exist_ok=True)
    shutil.copy(init_image_path, f"results/ours-init/{index}.jpeg")

    image_row = []
    for e in includes_epoches:
        image = get_validation_image(path, e)
        demo_row = demo_id // 4
        demo_col = demo_id % 4
        image = crop_image(image, demo_row, demo_col, grid_size=1024)
        image_row.append(image)
    all_images[f"ours-{name}"] = image_row

######################## OURS ########################

##################### baselines ######################
algos = ["ddpo", "dpok", "d3po"]

def random_image():
    return Image.fromarray(np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8))

def run_name(algo,name):
    return f"{algo}-{name}"

cache_path = "results/history_cache.pkl"
with open(cache_path, 'rb') as f:
    history_cache = pickle.load(f)

for algo in algos:
    for name in names:
        run_name_ = run_name(algo,name)

        # load pre-train image
        images_list = [Image.open(f"results/ppo-pretrain-images/{name}/{demo_id}.jpeg")]
        
        image_ = history_cache[run_name_]["validation/images"]
        image_k = history_cache[run_name_][image_.notnull()]["epoch"].values.tolist()
        image_v = image_[image_.notnull()].values.tolist()
        image_dict = dict(zip(image_k, image_v))

        image_dir = f"results/wandb-images/{run_name_}"

        for e in [9,19,29,39,49]:
            image_name = image_dict[e]["filenames"][demo_id]
            image_path = f"{image_dir}/{image_name}"
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                image = random_image()
            images_list.append(image)

        all_images[run_name_] = images_list


##################### baselines ######################


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

algos = ["ddpo", "dpok", "d3po", "ours"]

long_name = []
save_dir = f"results/compare-all/seed={demo_id}"
os.makedirs(save_dir, exist_ok=True)

for index, name in enumerate(names):

    fig = plt.figure(figsize=(12, 8))
    
    gs = gridspec.GridSpec(4, 7, wspace=0.06, hspace=0.0, width_ratios=[1, 0.1, 1, 1, 1, 1, 1])  # Second column is for the gap

    for algo_i, algo in enumerate(algos):
        image_row = all_images[f"{algo}-{name}"]
        for image_i, image in enumerate(image_row):
            ax = fig.add_subplot(gs[algo_i, image_i + (image_i >= 1)])  # Adjust index to skip the second column
            ax.imshow(image.resize((256, 256)))
            ax.axis('off')

            if image_i == 0:
                ax.text(-0.15, 0.5, algo.upper(), ha='right', va='center', fontsize=24, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{index}.jpeg", dpi=600, bbox_inches='tight')