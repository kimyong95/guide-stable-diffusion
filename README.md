This is the official implementation of the ICLR 2025 paper:

[**Fast Direct: Query-Efficient  Online Black-box Guidance  for Diffusion-model Target Generation**](https://openreview.net/forum?id=OmpTdjl7RV)

This project is built on [Pytorch Lightning](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://lightning.ai/docs/pytorch/stable//index.html&ved=2ahUKEwjktJ-P2-OMAxX2zDgGHQ5AJZ0QFnoECCIQAQ&usg=AOvVaw3BfE_f7qL8m3mdqXG5xbGR) framework.

1. Install the conda environment: `conda env create -f environment.yaml`.
2. Setup your Wandb and Google (Gemini) API keys in `.env`.
3. To run the experiment: `python main fit --config configs/finetune_with_model.yaml`.
4. To debug using vscode, simply run the `Finetune with Model` launch configuration.

A self-contained code is available at `demo/fast_direct.py` for easy get started.
