# Fashion-MNIST Autoencoder CIA (2087101)

This repository contains a documented PyTorch notebook and concise report for:
- Undercomplete autoencoder reconstruction on Fashion-MNIST
- Bottleneck comparison for latent dimensions 8, 32, and 64
- Denoising autoencoder with noisy inputs and clean targets
- MSE evaluation, dimensional compression ratios, loss curves, and qualitative image grids

## Requirements
```bash
pip install torch torchvision matplotlib numpy notebook reportlab
```

## Run
Open `FashionMNIST_Autoencoder_CIA.ipynb` in Jupyter or Google Colab and run all cells. The first run downloads Fashion-MNIST.

## Important
Use the actual values printed by your run in `Autoencoder_CIA_Report.pdf`. Notebook metrics are not prefilled in the report because they depend on execution settings.
