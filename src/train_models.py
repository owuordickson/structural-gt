
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def cnn_normalize_images(data_dir: str, img_dim: int):
    """
    The images are not similar to ImageNet, they are SEM images, then it is always advised to calculate the mean and
    std of the dataset and use them to normalize the images. Saves the mean and std to a CSV file.

    :param data_dir: path to your dataset root (which has subfolders "good" and "bad")
    :param img_dim: image dimension (e.g., 224 for ResNet50)
    """

    # Temporary transform: only convert to tensor (no normalization yet)
    temp_transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),  # resize if needed
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=temp_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Compute mean and std
    mean = 0.
    std = 0.
    nb_samples = 0.

    for data, _ in loader:
        batch_samples = data.size(0)  # batch size (the last batch may be smaller!)
        data = data.view(batch_samples, data.size(1), -1)  # (B, C, H*W)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    # Convert to lists
    mean_list = mean.tolist()
    std_list = std.tolist()

    # Save to CSV inside the dataset folder
    out_csv = os.path.join(data_dir, "normalization_stats.csv")
    df = pd.DataFrame({
        "mean": mean_list,
        "std": std_list
    })
    df.to_csv(out_csv, index=False)
    print(f"Saved normalization stats to {out_csv}")


def get_image_normalizations(stats_file: str):
    """Loads the normalization stats from a CSV file."""
    df = pd.read_csv(stats_file)
    return df["mean"].tolist(), df["std"].tolist()


def cnn_fc_layer(num_features):
    fc_layer = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=num_features,
            out_features=1                  # 1 class: 0-bad, 1-good
        ),
        torch.nn.Sigmoid()                  # Converts output to range [0, 1] (<0.5-bad, >0.5-good)
    )
    return fc_layer


def train_cnn_model(ntwk: dict, num_epochs: int, learning_rate: float, train_loader, train_dataset, val_loader, val_dataset, device) -> None:
    """Trains CNN model in the dict and updates the dict with the metrics. Then, saves the model to disk."""
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ntwk["model"].parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        ntwk["model"].train()
        train_loss, train_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y = labels.view(-1, 1).float()

            optimizer.zero_grad()
            y_hat = ntwk["model"](inputs)
            pred_labels = (y_hat >= 0.5).float()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)  # We need to multiply by batch size as loss is the mean loss of the samples in the batch
            train_corrects += torch.sum(pred_labels == y)

        train_loss /= len(train_dataset)
        train_acc = train_corrects / len(train_dataset)

        # Validation
        ntwk["model"].eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                y = labels.view(-1, 1).float()

                y_hat = ntwk["model"](inputs)
                pred_labels = (y_hat >= 0.5).float()
                loss = criterion(y_hat, y)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(pred_labels == y)

        val_loss /= len(val_dataset)
        val_acc = val_corrects / len(val_dataset)

        # Save epoch metrics
        ntwk["train_loss"].append(train_loss)
        ntwk["val_loss"].append(val_loss)
        ntwk["train_acc"].append(train_acc)
        ntwk["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} -- "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save model
    os.makedirs("../models/checkpoints", exist_ok=True)
    torch.save(ntwk["model"].state_dict(), f"../models/checkpoints/sgt_{ntwk["name"]}.pth")
    print("Model saved.")


def run_cnn_model(graph_image_path: str, model_name: str = "resnet50", img_dim: int = 224, norm_file: str = "../train_data/manual/normalization_stats.csv"):
    """
    Load the CNN model and run it on a graph image to get the prediction for good/bad graph.
    :param graph_image_path: Graph image path, e.g., "../train_data/manual/good/1234567890.png.
    :param model_name: Name of the model to use.
    :param img_dim: Image dimension (e.g., 224 for ResNet50).
    :param norm_file: Path to the CSV file containing the normalization stats.
    """
    # Load image
    mean, std = get_image_normalizations(norm_file)
    transform_pipe = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)  # same as training
    ])
    image = Image.open(graph_image_path).convert("RGB")
    image = transform_pipe(image).unsqueeze(0)

    # Load model
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
    elif model_name == "vit":
        model = models.vit_b_16(weights=None)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
    else:
        raise ValueError("Invalid model name.")
    model.fc = cnn_fc_layer(model.fc.in_features)
    model.load_state_dict(torch.load(f"../models/checkpoints/sgt_{model_name}.pth"))
    model.eval()

    # Predict
    with torch.no_grad():
        y_pred = model(image)
        pred_val = y_pred.item()
        return pred_val



def plot_performance(ntwk: dict, num_epochs: int):
    # ---- Plot results ----
    epochs = range(1, num_epochs+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, ntwk["train_loss"], label="Train Loss", color="teal")
    ax1.plot(epochs, ntwk["val_loss"], label="Val Loss", color="orange")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, ntwk["train_acc"], label="Train Acc", color="teal")
    ax2.plot(epochs, ntwk["val_acc"], label="Val Acc", color="orange")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(f"figs/performance_{ntwk["name"]}.png", dpi=300)
    plt.show()



def save_performance_data(model_networks: dict) -> None:
    """"""
    # ---- Convert to DataFrames ----
    dfs = {}
    for name, net in model_networks.items():
        dfs[name] = pd.DataFrame({
            "train_loss": net["train_loss"],
            "val_loss": net["val_loss"],
            "train_acc": net["train_acc"],
            "val_acc": net["val_acc"],
        })

    # Example: save results
    for name, df in dfs.items():
        df.to_csv(f"{name}_performance.csv", index=False)
    print("Results saved in CVS.")