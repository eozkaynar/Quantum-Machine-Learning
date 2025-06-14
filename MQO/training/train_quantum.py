import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import time
import click
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
from datetime import datetime
from tqdm import tqdm 
from torch.utils.data import DataLoader
from torchvision import transforms
from MQO.dataset.mnist_dataset import MNISTDataset
from MQO.models.quantum_mlp import QMLP  # QMLP modelinizin tanımlı olduğu dosya

@click.command("quantum")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="MQO/data")
@click.option("--optimizer", type=click.Choice(["adam", "sgd", "rmsprop", "adamw"]), default="adam")
@click.option("--output", type=click.Path(file_okay=False), default="output/quantum")
@click.option("--run_test/--skip_test", default=True)
@click.option("--num_epochs", type=int, default=30)
@click.option("--lr", type=float, default=0.001)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--num_workers", type=int, default=2)
@click.option("--batch_size", type=int, default=16)
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=0)

def run(data_dir, optimizer, output, run_test, num_epochs, lr, weight_decay, num_workers, batch_size, device, seed):
    
    start_time = time.time()
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(output, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Load datasets
    dataset = {
        "train": MNISTDataset(data_dir=data_dir, split="train",transform=transform),
        "test": MNISTDataset(data_dir=data_dir, split="test",transform=transform),
    }

    model       = QMLP().to(device)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer")
    criterion   = torch.nn.NLLLoss()

    train_loss_list = []  # Eğitim kaybını burada tutacağız

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train_loss, train_acc = run_epoch(model, train_loader, opt, criterion, device, phase="train")
        train_loss_list.append(train_loss)  # Kaybı listeye ekle

        print(f"[Train] Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Eğitim kaybı grafiğini çiz
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_loss_list, marker='o', label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)  
    plt.legend()
    plt.savefig(os.path.join(output, f"train_loss_plot_{optimizer}.png"))  
    plt.show()

    if run_test:
        test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loss, test_acc = run_epoch(model, test_loader, opt, criterion, device, phase="test")
        print(f"[Test] Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    # Save results
    end_time = time.time()
    duration = int(end_time - start_time)

    result = {
        "optimizer": optimizer,
        "accuracy": round(test_acc, 2) if run_test else None,
        "train_time_seconds": duration,
        "timestamp": datetime.now().isoformat()
    }
    log_dir = "/mnt/data/optimizer_results"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{optimizer}_log.csv")
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
    else:
        df_log = pd.DataFrame(columns=["optimizer", "accuracy", "train_time_seconds", "timestamp"])

    df_log = pd.concat([df_log, pd.DataFrame([result])], ignore_index=True)
    df_log.to_csv(log_file, index=False)


def run_epoch(model, dataloader, opt, criterion, device, phase="train"):
    model.train() if phase == "train" else model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(phase == "train"):
        pbar = tqdm(dataloader, desc=f"[{phase.capitalize()}]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device).float(), labels.to(device)
            if phase == "train":
                opt.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if phase == "train":
                loss.backward()
                opt.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    run()
