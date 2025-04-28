import math
import os
import click
import time
import torch
import sklearn.metrics
import tqdm
import pandas               as pd
import matplotlib.pyplot    as plt
import numpy                as np

from torch.utils.data           import DataLoader
from MQO.dataset.mnist_dataset  import MNISTDataset
from MQO.models.classical_mlp   import MLP

@click.command("classical")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="data")
@click.option("--output", type=click.Path(file_okay=False), default="output/classical")
# @click.option("--hyperparameter_dir", type=click.Path(file_okay=False), default="hyperparam_outputs")
@click.option("--run_test/--skip_test", default=True)
# @click.option("--hyperparameter", type=bool, default=False)
@click.option("--num_epochs", type=int, default=35)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--num_workers", type=int, default=4)
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

    # os.makedirs(output, exist_ok=True)  # Ensure the base output directory exists
    # if hyperparameter:
    #     output = hyperparameter_dir
    # output = os.path.join(output, f"lr_{lr}_wd_{weight_decay}_bs_{batch_size}_nh_{num_heads}_nl_{num_layers}_pd_{projection_dim}") if hyperparameter else output
    # os.makedirs(output, exist_ok=True)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Compute mean and std

    # Set up datasets and dataloaders
    dataset     = {}  
    # Load datasets
    dataset["train"]   = MNISTDataset(data_dir=data_dir, split="train")
    dataset["val"]     = MNISTDataset(data_dir=data_dir, split="val")
    dataset["test"]    = MNISTDataset(data_dir=data_dir, split="test")


    log_file_path = os.path.join(output, "log.csv")
    # Run training and testing loops
    with open(os.path.join(log_file_path), "a") as f:
        
        f.write("epoch,phase,epoch_loss,epoch_acc,time\n")
        train_losses    = []
        val_losses      = []

        for epoch in range(num_epochs): 
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                ds                             = dataset[phase]
                dataloader                     = DataLoader(ds, batch_size=batch_size, shuffle=(phase=="train"), num_workers=num_workers)
                epoch_loss, epoch_acc, yhat, y = run_epoch(model, dataloader, phase, optimizer, criterion, device,train_losses,val_losses)
                
                f.write("{},{},{},{},{}\n".format(
                                                epoch,
                                                phase,
                                                epoch_loss,
                                                epoch_acc,
                                                time.time() - start_time))
                f.flush()
            # scheduler.step(epoch + 1)

        if run_test:
            split = "test"
            ds     = dataset[split]
            
            dataloader                     = DataLoader(MNISTDataset(data_dir=data_dir, split=split),batch_size=batch_size, 
                                                 shuffle=False, num_workers=num_workers)
            epoch_loss, epoch_acc, yhat, y = run_epoch(model, dataloader, split, optimizer, criterion, device,train_losses=[],val_losses=[])
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
    np.save(os.path.join(output, "val_losses.npy"), np.array(val_losses))
    print(f"Train and validation losses saved to {output}")

def run_epoch(model, dataloader, phase, optimizer,criterion, device,train_losses, val_losses):
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
                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(running_loss / n, loss.item()))
                pbar.update(1)

    epoch_loss = running_loss / n
    epoch_acc  = 100.0 * correct / n

    if (phase== "train"):
        train_losses.append(epoch_loss)
    elif phase== "val":
        val_losses.append(epoch_loss)
    else:
        pass
                   
    yhat    = np.concatenate(yhat)
    y       = np.concatenate(y)

    return epoch_loss, epoch_acc, yhat, y
if __name__ == "__main__":
    run()