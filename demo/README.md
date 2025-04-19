This is a self-contained code to demostrate Fast-Direct.

To run:
1. `conda env create -f environment.yaml`
2. Setup your Wandb and Google (Gemini) API, and save to `.env`
2. `python fast_direct.py --train` to "train" the GP model.
3. `python fast_direct.py` to load the `data.pth` and generate image `output.jpg`.