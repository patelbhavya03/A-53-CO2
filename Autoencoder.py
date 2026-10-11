
        noisy_images = torch.clamp(
            clean_images + NOISE_STD * torch.randn_like(clean_images),
            0.0, 1.0
        )

        reconstructed = denoising_model(noisy_images)

        total_error += ((reconstructed - clean_images) ** 2).sum().item()
        total_pixels += clean_images.numel()

denoising_mse = total_error / total_pixels

log(f"Denoising test MSE: {denoising_mse:.6f}")

plt.figure(figsize=(8, 4))
plt.plot(
    range(1, len(denoising_history) + 1),
    denoising_history,
    marker="o"
)
plt.xlabel("Epoch")
plt.ylabel("MSE against clean target")
plt.title("Denoising Autoencoder Training Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_denoising_loss_curve.png", dpi=160)
plt.show()

torch.manual_seed(2026)
images = X_test[:8].to(device)

with torch.no_grad():
    noisy_images = torch.clamp(
        images + NOISE_STD * torch.randn_like(images),
        0.0, 1.0
    )
    denoised_images = denoising_model(noisy_images)

fig, axes = plt.subplots(3, 8, figsize=(13, 5))

for i in range(8):
    axes[0, i].imshow(images[i].cpu().reshape(28, 28), cmap="gray")
    axes[1, i].imshow(noisy_images[i].cpu().reshape(28, 28), cmap="gray")
    axes[2, i].imshow(denoised_images[i].cpu().reshape(28, 28), cmap="gray")

    for row in range(3):
        axes[row, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=9)
axes[1, 0].set_ylabel("Noisy input", fontsize=9)
axes[2, 0].set_ylabel("Denoised", fontsize=9)

plt.suptitle("Denoising Autoencoder: Actual Model Outputs")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_denoising_visualization.png", dpi=160)
plt.show()

RUN_REGULARIZATION = True

if RUN_REGULARIZATION:
    log("\nOPTIONAL EXPERIMENT: L2 REGULARIZATION")

    regularized_model, regularized_history = train_autoencoder(
        latent_dim=32,
        epochs=EPOCHS,
        noise_std=0.0,
        weight_decay=1e-4
    )

    regularized_mse = evaluate_mse(regularized_model, test_loader)
    log(f"Regularized model test MSE: {regularized_mse:.6f}")

    plt.figure(figsize=(8, 4))
    plt.plot(histories[32], label="Without weight decay")
    plt.plot(regularized_history, label="With L2 weight decay")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title("Effect of L2 Regularization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_regularization_comparison.png", dpi=160)
    plt.show()

log("\nFINAL SUMMARY")
log("=" * 65)

for result in results:
    log(
        f"Latent {result['latent_dim']}: "
        f"MSE={result['test_mse']:.6f}, "
        f"compression={result['compression_ratio']:.2f}:1"
    )

log(f"Denoising test MSE: {denoising_mse:.6f}")

with open(f"{OUTPUT_DIR}/output_logs.txt", "w") as f:
    f.write("\n".join(log_lines))

torch.save(
    {k: v.state_dict() for k, v in models.items()},
    f"{OUTPUT_DIR}/trained_autoencoders.pth"
)
torch.save(
    denoising_model.state_dict(),
    f"{OUTPUT_DIR}/trained_denoising_autoencoder.pth"
)

zip_path = "/content/FashionMNIST_Autoencoder_Real_Outputs.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(filepath):
            z.write(filepath, arcname=filename)

log("\nAll output files saved in: " + OUTPUT_DIR)
log("ZIP file created: " + zip_path)

from google.colab import files
files.download(zip_path)
