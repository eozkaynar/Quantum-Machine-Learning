import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import click
import time
import torch
import torchvision
import sklearn.metrics
import tqdm
import pandas               as pd
import matplotlib.pyplot    as plt
import numpy                as np

from torch.utils.data           import DataLoader
from MQO.dataset.mnist_dataset  import MNISTDataset
from MQO.models.classical_mlp   import MLP

@click.command("classical")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="MQO/data")
@click.option("--output", type=click.Path(file_okay=False), default="output/classical")
# @click.option("--hyperparameter_dir", type=click.Path(file_okay=False), default="hyperparam_outputs")
@click.option("--run_test/--skip_test", default=True)
# @click.option("--hyperparameter", type=bool, default=False)
@click.option("--num_epochs", type=int, default=35)
@click.option("--lr", type=float, default=1e-3)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--num_workers", type=int, default=0)
@click.option("--batch_size", type=int, default=16)
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=0)

def run(

    data_dir,
    output,
    run_test,
    num_epochs,
    lr,
    weight_decay,
    num_workers,
    batch_size,
    device,
    seed,
):

    os.makedirs(output, exist_ok=True)  # Ensure the base output directory exists

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

     # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(output):
        os.makedirs(output)

     # Initialize model, optimizer and criterion
    model     = MLP(input_size=784, hidden_sizes=[128,64], num_classes=10)
    model     = model.to(device)
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Load MNIST dataset without normalization (only convert to tensor)
    transform = torchvision.transforms.ToTensor()
    train_dataset = MNISTDataset(data_dir=data_dir, split="train",transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=False)

    # Get the entire training batch
    images, _ = next(iter(train_loader))  # shape: (60000, 1, 28, 28)

    # Compute mean and standard deviation
    mean = images.mean().item()
    std = images.std().item()

    print(f"Mean: {mean:.4f}, Std: {std:.4f}")

    transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((mean,), (std,))
    ])

    # Set up datasets and dataloaders
    dataset     = {}  
    # Load datasets
    dataset["train"]   = MNISTDataset(data_dir=data_dir, split="train",transform=transform)
    dataset["test"]    = MNISTDataset(data_dir=data_dir, split="test",transform=torchvision.transforms.ToTensor())

    # Start training time
    training_start_time = time.time()

    # Number of model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    with open(os.path.join(output, "summary.txt"), "w") as summary_file:
        summary_file.write(f"Trainable Parameters: {total_params}\n")

    log_file_path = os.path.join(output, "log.csv")
    # Run training and testing loops
    with open(os.path.join(log_file_path), "a") as f:
        
        f.write("epoch,phase,epoch_loss,epoch_acc,time\n")
        train_losses    = []

        for epoch in range(num_epochs): 
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train']:
                start_time = time.time()
                ds                             = dataset[phase]
                dataloader                     = DataLoader(ds, batch_size=batch_size, shuffle=(phase=="train"), num_workers=num_workers)
                epoch_loss, epoch_acc, yhat, y = run_epoch(model, dataloader, phase, optimizer, criterion, device,train_losses)
                
                f.write("{},{},{},{},{}\n".format(
                                                epoch,
                                                phase,
                                                epoch_loss,
                                                epoch_acc,
                                                time.time() - start_time))
                f.flush()
            # scheduler.step(epoch + 1)
        total_training_time = time.time() - training_start_time
        print(f"Total Training Time: {total_training_time:.2f} seconds")
        with open(os.path.join(output, "summary.txt"), "a") as summary_file:
            summary_file.write(f"Total Training Time: {total_training_time:.2f} seconds\n")
        if run_test:
            split = "test"
            ds     = dataset[split]
            
            dataloader                     = DataLoader(MNISTDataset(data_dir=data_dir, split=split),batch_size=batch_size, 
                                                 shuffle=False, num_workers=num_workers)
            epoch_loss, epoch_acc, yhat, y = run_epoch(model, dataloader, split, optimizer, criterion, device,train_losses=[])
            # Write full performance to file
            with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                g.write("true_value,prediction\n")
                for (pred, target) in zip(yhat, y):
                        g.write("{},{}\n".format(target,pred))

                g.write("{} Accuracy:   {:.3f} \n".format(split, sklearn.metrics.accuracy_score(y, yhat)))
                g.write("{} Precision:  {:.3f} \n".format(split, sklearn.metrics.precision_score(y, yhat, average='macro')))
                g.write("{} Recall: {:.3f} \n".format(split, sklearn.metrics.recall_score(y, yhat, average='macro')))
                g.write("{} F1 Score: {:.3f} \n".format(split, sklearn.metrics.f1_score(y, yhat, average='macro')))
                g.flush()

    
    np.save(os.path.join(output, "train_losses.npy"), np.array(train_losses))
    print(f"Train and validation losses saved to {output}")

def run_epoch(model, dataloader, phase, optimizer,criterion, device,train_losses):
    """Run one epoch of training/evaluation for classification task (MNIST)."""

    if phase == "train":
        model.train()
    else:  
        model.eval()

    yhat = []          # Prediction 
    y    = []          # Ground truth

    running_loss = 0.0
    n            = 0
    correct      = 0
    with torch.set_grad_enabled(phase== "train"):
        with tqdm.tqdm(total=len(dataloader), desc=f"{phase.capitalize()} Epoch") as pbar:
            for (images, labels) in dataloader:

                images = images.float().to(device)
                labels = labels.long().to(device)

                y.append(labels.cpu().numpy())
                
                outputs         = model(images)
                _, preds        = torch.max(outputs, 1)
                yhat.append(preds.to("cpu").detach().numpy())

                loss            = criterion(outputs,labels)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()  
                
                
                running_loss  += loss.item()
                n             += images.size(0)
                correct       += (preds == labels).sum().item() 
                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}%".format(running_loss / n, loss.item(), 100 * correct / n))
                pbar.update(1)

    epoch_loss = running_loss / n
    epoch_acc  = 100.0 * correct / n

    if (phase== "train"):
        train_losses.append(epoch_loss)
                      
    yhat    = np.concatenate(yhat)
    y       = np.concatenate(y)

    return epoch_loss, epoch_acc, yhat, y
if __name__ == "__main__":
    run()