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
from scipy.optimize import minimize
from MQO.dataset.mnist_dataset import MNISTDataset
from MQO.models.quantum_mlp import QMLP, qnode, weight_shapes
from pennylane.optimize import SPSAOptimizer


@click.command("quantum")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="MQO/data")
@click.option("--optimizer", type=click.Choice(["adam", "sgd", "rmsprop", "adamw","spsa","cobyla"]), default="cobyla")
@click.option("--output", type=click.Path(file_okay=False), default="output/quantum")
@click.option("--run_test/--skip_test", default=True)
@click.option("--num_epochs", type=int, default=30)
@click.option("--lr", type=float, default=0.05)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--num_workers", type=int, default=2)
@click.option("--batch_size", type=int, default=32)
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
    if optimizer == "spsa":
        spsa_train(dataset, num_epochs, batch_size, output)
        return
    elif optimizer == "cobyla":
        cobyla_train(dataset, num_epochs, batch_size, output)
        return
    elif optimizer == "adam":
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
    log_dir = os.path.join(output, "logs")  # output/quantum/logs
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

def spsa_train(dataset, num_epochs, batch_size, output):

    # Pool  
    model_input = torch.nn.AvgPool2d(7)

    # SPSA optimizer
    opt = SPSAOptimizer(maxiter=10,a=0.5, c=0.05)

    # Initial weights
    weights = {k: 0.01 * np.random.randn(v) for k, v in weight_shapes.items()}
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    losses = []
    total_start = time.time()

    for epoch in range(num_epochs):
        
        W_proj = np.random.randn(16, 10) # projection  
        epoch_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"[SPSA Epoch {epoch+1}]")
        for x_batch, y_batch in pbar:
            x_batch = model_input(x_batch).view(x_batch.size(0), -1).numpy()
            y_batch = y_batch.numpy()

            def loss_fn(w):
                batch_loss = 0.0
                for x, label in zip(x_batch, y_batch):
                    out = qnode(x, **w)
                    logits = np.dot(out, W_proj) 
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)
                    batch_loss += -np.log(probs[label] + 1e-8)
                return np.array([batch_loss / len(x_batch)])

            weights = opt.step(loss_fn, weights)
            l = loss_fn(weights)
            epoch_loss += l

            for x, label in zip(x_batch, y_batch):
                out = qnode(x, **weights)
                logits = np.dot(out, W_proj)
                pred = np.argmax(logits)
                correct += int(pred == label)
                total += 1
            pbar.set_postfix({"loss": l})

        acc = 100.0 * correct / total
        avg_loss = epoch_loss / total
        print(f"[SPSA][Epoch {epoch+1}] Loss: {float(avg_loss):.4f}, Accuracy: {float(acc):.2f}%")
        losses.append(avg_loss)

    duration = int(time.time() - total_start)

    # Loss plot
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SPSA Training Loss")
    plt.grid()
    plt.savefig(os.path.join(output, "train_loss_plot_spsa.png"))
    plt.show()

    # Log
    result = {
        "optimizer": "spsa",
        "accuracy": round(float(acc), 2),
        "train_time_seconds": duration,
        "timestamp": datetime.now().isoformat()
    }

    log_dir = os.path.join(output, "logs")  # output/quantum/logs
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"spsa_log.csv")
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
    else:
        df_log = pd.DataFrame(columns=["optimizer", "accuracy", "train_time_seconds", "timestamp"])
    df_log = pd.concat([df_log, pd.DataFrame([result])], ignore_index=True)
    df_log.to_csv(log_file, index=False)

def cobyla_train(dataset, num_epochs, batch_size, output):

    model_input = torch.nn.AvgPool2d(7)
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    losses = []
    W_proj = np.random.randn(16, 10)  # projection

    # Make weight matrix to vector
    total_params = sum(weight_shapes[k] for k in weight_shapes)
    initial_weights = np.random.randn(total_params)- 0.5 #uniform dist btw -0.5 0.5  

    def unpack_weights(flat_weights):
        weights = {}
        idx = 0
        for k, size in weight_shapes.items():
            weights[k] = flat_weights[idx:idx+size]
            idx += size
        return weights

    total_start = time.time()
    pbar = tqdm(range(num_epochs), desc="[COBYLA Epoch]")
    for epoch in pbar:
 
        epoch_loss = 0
        correct = 0
        total = 0

        x_batch, y_batch = next(iter(train_loader))
        x_batch = model_input(x_batch).view(x_batch.size(0), -1).numpy()
        y_batch = y_batch.numpy()

        def loss_fn(flat_weights):
            w = unpack_weights(flat_weights)
            loss = 0.0
            for x, label in zip(x_batch, y_batch):
                out = qnode(x, **w)
                logits = np.dot(out, W_proj)
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                loss += -np.log(probs[label] + 1e-8)
            return float(loss / len(x_batch))

        result = minimize(loss_fn, initial_weights, method="COBYLA", options={"maxiter": 100})
        final_weights = unpack_weights(result.x)

        for x, label in zip(x_batch, y_batch):
            out = qnode(x, **final_weights)
            logits = np.dot(out, W_proj)
            pred = np.argmax(logits)
            correct += int(pred == label)
            total += 1

        avg_loss = result.fun
        acc = 100.0 * correct / total
        print(f"[COBYLA][Epoch {epoch+1}] Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
        losses.append(avg_loss)
        pbar.set_postfix({"loss": avg_loss, "acc": acc})

    duration = int(time.time() - total_start)

    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("COBYLA Training Loss")
    plt.grid()
    plt.savefig(os.path.join(output, "train_loss_plot_cobyla.png"))
    plt.show()

    result = {
        "optimizer": "cobyla",
        "accuracy": round(acc, 2),
        "train_time_seconds": duration,
        "timestamp": datetime.now().isoformat()
    }

    log_dir = os.path.join(output, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"cobyla_log.csv")
    if os.path.exists(log_file):
        df_log = pd.read_csv(log_file)
    else:
        df_log = pd.DataFrame(columns=["optimizer", "accuracy", "train_time_seconds", "timestamp"])
    df_log = pd.concat([df_log, pd.DataFrame([result])], ignore_index=True)
    df_log.to_csv(log_file, index=False)

if __name__ == "__main__":
    run()
