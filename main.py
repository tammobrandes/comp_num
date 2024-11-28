from dataset_setup import get_dataloaders
from model import get_model_and_optimizer
from train import train_and_validate

def main():
    # Paths to the CSV file and image directory
    csv_file = '/Users/tammo/Desktop/Project/AltDatasetLarger/dataset.csv'
    img_dir = '/Users/tammo/Desktop/Project/AltDatasetLarger'

    model_name = "MT"

    for i in range(4):
        # Get DataLoaders for training, validation, and test sets
        print(f'Training Instance {i+1}/8')
        
        train_loader, val_loader, test_loader = get_dataloaders(csv_file, img_dir, batch_size=32)

        # Get the model, loss functions, and optimizer
        model, loss_fn1, loss_fn2, optimizer = get_model_and_optimizer()

        # Train and validate the model, save results
        train_and_validate(model, loss_fn1, loss_fn2, optimizer, train_loader, val_loader, num_epochs=20, save_dir=f"{model_name}_{i}_checkpoints")

if __name__ == "__main__":
    main()

