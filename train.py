# train.py

import torch
import pandas as pd
import os
from tqdm import tqdm  # Import tqdm for the progress bar

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = torch.sum(preds == labels).item()
    total = labels.size(0)
    return correct / total

def run_epoch(model, dataloader, loss_fn1, loss_fn2, optimizer=None, device="cuda"):
    model.to(device)
    epoch_loss1, epoch_loss2 = 0.0, 0.0
    epoch_correct1, epoch_correct2 = 0, 0
    epoch_total = 0

    is_training = optimizer is not None

    # Use tqdm to display the progress bar for the dataloader
    with tqdm(dataloader, unit="batch") as tepoch:
        for inputs, targets1, targets2 in tepoch:
            tepoch.set_description("Training" if is_training else "Validation")
            inputs, targets1, targets2 = inputs.to(device), targets1.to(device), targets2.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs1, outputs2 = model(inputs)

            # Calculate losses
            loss1 = loss_fn1(outputs1, targets1)
            loss2 = loss_fn2(outputs2, targets2)
            total_loss = loss1 + loss2

            if is_training:
                total_loss.backward()
                optimizer.step()

            # Accumulate loss
            epoch_loss1 += loss1.item() * inputs.size(0)
            epoch_loss2 += loss2.item() * inputs.size(0)

            # Calculate accuracy
            epoch_correct1 += calculate_accuracy(outputs1, targets1) * inputs.size(0)
            epoch_correct2 += calculate_accuracy(outputs2, targets2) * inputs.size(0)

            epoch_total += inputs.size(0)

            # Update the progress bar with the current loss values
            tepoch.set_postfix(loss1=loss1.item(), loss2=loss2.item())

    avg_loss1 = epoch_loss1 / epoch_total
    avg_loss2 = epoch_loss2 / epoch_total
    avg_acc1 = epoch_correct1 / epoch_total
    avg_acc2 = epoch_correct2 / epoch_total

    return avg_loss1, avg_loss2, avg_acc1, avg_acc2

def train_and_validate(model, loss_fn1, loss_fn2, optimizer, train_loader, val_loader, num_epochs=20, save_dir):
    device = torch.device("cuda")  # You can change to 'cuda' if using a GPU
    os.makedirs(save_dir, exist_ok=True)

    history = []

    for epoch in range(num_epochs + 1):  # Including epoch 0
        if epoch == 0:
            print(f"Running epoch {epoch} (before training)...")
            train_loss1, train_loss2, train_acc1, train_acc2 = run_epoch(model, train_loader, loss_fn1, loss_fn2, device=device)
            val_loss1, val_loss2, val_acc1, val_acc2 = run_epoch(model, val_loader, loss_fn1, loss_fn2, device=device)
        else:
            print(f"Training epoch {epoch}...")
            train_loss1, train_loss2, train_acc1, train_acc2 = run_epoch(model, train_loader, loss_fn1, loss_fn2, optimizer=optimizer, device=device)
            val_loss1, val_loss2, val_acc1, val_acc2 = run_epoch(model, val_loader, loss_fn1, loss_fn2, device=device)

        # Save the model state
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))

        # Save epoch results
        epoch_results = {
            'epoch': epoch,
            'train_loss1': train_loss1, 'train_loss2': train_loss2,
            'train_acc1': train_acc1, 'train_acc2': train_acc2,
            'val_loss1': val_loss1, 'val_loss2': val_loss2,
            'val_acc1': val_acc1, 'val_acc2': val_acc2
        }
        history.append(epoch_results)
        print(epoch_results)

    # Save the history of results
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    print("Training complete.")
