import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import time
import click
import torch
import numpy as np
import sklearn.metrics
from tqdm import tqdm 
from torch.utils.data import DataLoader
from MQO.dataset.mnist_dataset import MNISTDataset
from MQO.models.quantum_mlp import QMLP  # QMLP modelinizin tanımlı olduğu dosya

@click.command("quantum")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="MQO/data")
@click.option("--output", type=click.Path(file_okay=False), default="output/quantum")
@click.option("--run_test/--skip_test", default=True)
@click.option("--num_epochs", type=int, default=20)
@click.option("--lr", type=float, default=0.001)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--num_workers", type=int, default=2)
@click.option("--batch_size", type=int, default=16)
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=0)

def run(data_dir, output, run_test, num_epochs, lr, weight_decay, num_workers, batch_size, device, seed):
    
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(output, exist_ok=True)

    # Load datasets
    dataset = {
        "train": MNISTDataset(data_dir=data_dir, split="train"),
        "test": MNISTDataset(data_dir=data_dir, split="test"),
    }

    model       = QMLP().to(device)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion   = torch.nn.NLLLoss()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, phase="train")

        print(f"[Train] Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    if run_test:
        test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loss, test_acc = run_epoch(model, test_loader, optimizer, criterion, device, phase="test")
        print(f"[Test] Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

def run_epoch(model, dataloader, optimizer, criterion, device, phase="train"):
    model.train() if phase == "train" else model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(phase == "train"):
        pbar = tqdm(dataloader, desc=f"[{phase.capitalize()}]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device).float(), labels.to(device)
            if phase == "train":
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if phase == "train":
                loss.backward()
                optimizer.step()

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
