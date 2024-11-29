from dataset_setup import get_dataloaders
from model import get_model_and_optimizer
from train import train_and_validate

def main():
    # Path to the ZIP file
    zip_file = 'C:/Users/jiskg/OneDrive/Documenten/MBSC/NCM/dataMiniProject/AltDatasetLarger/AltDatasetLarger.zip'
    subfolder_name = 'AltDatasetLarger'  # Subfolder inside the ZIP
    csv_filename = 'dataset.csv'  # Name of the CSV inside the subfolder

    model_name = "MT"

    for i in range(10):
        print(f'Training Instance {i+1}/10')

        # Get DataLoaders
        train_loader, val_loader, test_loader = get_dataloaders(
            zip_file=zip_file,
            subfolder_name=subfolder_name,
            csv_filename=csv_filename,
            batch_size=32
        )

        # Get the model and optimizer
        model, loss_fn1, loss_fn2, optimizer = get_model_and_optimizer()

        # Train and validate the model
        train_and_validate(model, loss_fn1, loss_fn2, optimizer, train_loader, val_loader, num_epochs=20, save_dir=f"{model_name}_{i}_checkpoints")

if __name__ == "__main__":
    main()
