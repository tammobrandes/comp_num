import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from collections import defaultdict
import os
import gc
from tqdm import tqdm

device = torch.device('cuda')

# Function to get softmax values and corresponding ground truth labels
def get_softmax_values_and_ground_truth(model, test_loader):
    num_softmax = []
    object_softmax = []
    device = torch.device('cuda')
    model.to(device)
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        for inputs, targets1, targets2 in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs1, outputs2 = model(inputs)
            softmax = F.softmax(outputs2, dim=1)  # Softmax on the numerosity head (15-channel output)
            softmax2 = F.softmax(outputs1, dim = 1)
            for i in range(len(targets2)):
                num_softmax.append((softmax[i].cpu().numpy(), targets1[i].item(), targets2[i].item()))
                object_softmax.append((softmax2[i].cpu().numpy(), targets1[i].item(), targets2[i].item()))
    return num_softmax, object_softmax

# Function to summarize softmax values by numerosity and object type
def summarize_softmax(softmax_results,):
    summary = {}
    for softmax_val, object_type, numerosity in softmax_results:
        if (numerosity, object_type) not in summary:
            summary[(numerosity, object_type)] = []
        summary[(numerosity, object_type)].append(softmax_val)

    summarized_data = {}
    spread_data = {}
    for key, values in summary.items():
        values = np.array(values)
        mean_softmax = values.mean(axis=0)  # Mean distribution
        spread = kurtosis(values, axis=0)  # Measure of spread (kurtosis)
        summarized_data[key] = mean_softmax
        spread_data[key] = spread
    

    return summarized_data, spread_data

def load_best_model(history_path, model, model_name, optimizer, checkpoints_dir):
    """
    Load the best model based on the lowest validation loss sum from the history CSV.
    """
    # Load the training history
    history_df = pd.read_csv(history_path)
    
    # Find the best epoch based on the lowest sum of validation losses
    history_df['val_loss_sum'] = history_df['val_loss1'] + history_df['val_loss2']
    best_epoch = history_df['val_loss_sum'].idxmin()
    
    best_epoch_path = os.path.join(checkpoints_dir, f'{model_name}_epoch_{best_epoch}.pth')

    # Load the model from the best epoch checkpoint
    model.load_state_dict(torch.load(best_epoch_path))
    model.eval()

    # Print the selected best epoch
    print(f"Best epoch selected: {best_epoch} (based on lowest validation loss sum)")

    return model, best_epoch

# Function to compute accuracy from softmax values and ground truth labels
def compute_accuracy(model_name, condition, num_softmax, object_softmax, csv_save_dir):
    """
    Compute test accuracy by comparing predicted classes (from softmax) to ground truth.
    """
    # Confusion matrix data
    
    numerosity_confusion = {i: np.zeros(16, dtype=int) for i in range(16)}
    object_confusion = {i: np.zeros(10, dtype=int) for i in range(10)}

    numerosity_correct = {i: 0 for i in range(16)}
    numerosity_total = {i: 0 for i in range(16)}
    object_correct = {i: 0 for i in range(10)}
    object_total = {i: 0 for i in range(10)}

    for softmax_val, object_label, numerosity_label in num_softmax:
        # Convert softmax to predicted class (argmax)
        predicted_numerosity = np.argmax(softmax_val)
        
        # Track numerosity accuracy
        if numerosity_label not in numerosity_correct:
            numerosity_correct[numerosity_label] = 0
            numerosity_total[numerosity_label] = 0
        numerosity_total[numerosity_label] += 1
        if predicted_numerosity == numerosity_label:
            numerosity_correct[numerosity_label] += 1

        # Update confusion matrix for numerosity
        numerosity_confusion[numerosity_label][predicted_numerosity] += 1

    for softmax_val, object_label, numerosity_label in object_softmax:

        predicted_object = np.argmax(softmax_val)

        # Track object accuracy (assuming predicted numerosity is the "task" for now)
        if object_label not in object_correct:
            object_correct[object_label] = 0
            object_total[object_label] = 0
        object_total[object_label] += 1
        if predicted_object == object_label:
            object_correct[object_label] += 1

        # Update confusion matrix for object
        object_confusion[object_label][predicted_object] += 1

    # Calculate accuracies as percentages
    numerosity_accuracy = {k: numerosity_correct[k] / numerosity_total[k] for k in numerosity_correct}
    object_accuracy = {k: object_correct[k] / object_total[k] for k in object_correct}

    # Save numerosity accuracy to CSV
    numerosity_df = pd.DataFrame(list(numerosity_accuracy.items()), columns=['Numerosity_Label', 'Accuracy'])
    numerosity_df.to_csv(os.path.join(csv_save_dir, f'{model_name}_numerosity_accuracy{condition}.csv'), index=False)

    # Save object accuracy to CSV
    object_df = pd.DataFrame(list(object_accuracy.items()), columns=['Object_Label', 'Accuracy'])
    object_df.to_csv(os.path.join(csv_save_dir, f'{model_name}_object_accuracy{condition}.csv'), index=False)

    # Convert confusion matrix dictionaries to pandas DataFrames and save them
    numerosity_confusion_df = pd.DataFrame(numerosity_confusion).fillna(0).astype(int)
    numerosity_confusion_df.to_csv(os.path.join(csv_save_dir, f'{model_name}_numerosity_confusion_matrix{condition}.csv'), index=False, header=False)

    object_confusion_df = pd.DataFrame(object_confusion).fillna(0).astype(int)
    object_confusion_df.to_csv(os.path.join(csv_save_dir, f'{model_name}_object_confusion_matrix{condition}.csv'), index=False, header=False)

    return numerosity_accuracy, object_accuracy, numerosity_confusion, object_confusion


def calculate_entropy(softmax_values):
    epsilon = 1e-10  # Small constant to avoid log(0)
    return -np.sum(softmax_values * np.log(softmax_values + epsilon))

def extract_layer_activations(model, layer, inputs):
    """
    Extract activation maps from a specified layer in the model.
    """
    activations = []
    device = torch.device('cuda')
    model.to(device)
    
    def hook_fn(module, input, output):
        activations.append(F.relu(output).detach().clone())

    handle = layer.register_forward_hook(hook_fn)
    
    inputs = inputs.to(device)

    model(inputs)  # Perform a forward pass to collect the activations

    handle.remove()

    return activations[0]  # Return the collected activations

def initialize_decoder(model, layer, num_classes):
    """
    Train a linear decoder on the activation maps to classify the numerosities.
    The decoder will include an AdaptiveAvgPool2d layer before the linear layer.
    """

    # Define the linear decoder model
    class Decoder(nn.Module):
        def __init__(self, input_channels, num_classes):
            super(Decoder, self).__init__()
            
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling
            self.fc = nn.Linear(input_channels, num_classes)  # Linear layer for classification

        def forward(self, x):
            
            x = self.avg_pool(x)  # Apply adaptive average pooling            
            x = torch.flatten(x, 1)  # Flatten to [batch_size, channels]            
            x = self.fc(x)  # Apply linear layer
            
            return x
    
    input_channels = layer.out_channels

    # Initialize the decoder model
    decoder = Decoder(input_channels, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    return decoder, criterion, optimizer


def layer_decoding_analysis(model, train_dataloader, test_dataloader, layers, num_classes):
    """
    Perform decoding analysis for each specified layer in the model in smaller chunks.
    Returns accuracies, softmax values, entropy for each numerosity class across layers,
    and predicted/true labels for each layer.
    """
    accuracies = {numerosity: [[] for _ in range(len(layers))] for numerosity in range(num_classes)}
    softmax_values = {numerosity: [[] for _ in range(len(layers))] for numerosity in range(num_classes)}
    entropy_values = {numerosity: [[] for _ in range(len(layers))] for numerosity in range(num_classes)}
    predictions_per_layer = [[] for _ in range(len(layers))]
    true_labels_per_layer = [[] for _ in range(len(layers))]
    
    device = torch.device('cuda')
    num_epochs = 10

    # Iterate over the layers with a progress bar
    for layer_idx, layer in enumerate(tqdm(layers, desc="Analyzing Layers")):
        
        decoder, criterion, optimizer = initialize_decoder(model = model, layer = layer, num_classes = 16)
        decoder.to(device)
        
        decoder.train()
        # Extract features for the dataset
        for e in tqdm(range(num_epochs)):
            tqdm.set_description(f'Training Decoder: Epoch {e+1}/{num_epochs}')
            for inputs, _, labels in tqdm(train_dataloader, desc="Epoch Progress", leave=False):
                optimizer.zero_grad()
                
                activations = extract_layer_activations(model, layer, inputs)
                output = decoder(activations)
                
                labels.to(device)
                
                loss = criterion(output, labels)
                
                loss.backward()
                optimizer.step()   

        # Evaluate the decoder for each numerosity
        decoder.eval()
        with torch.no_grad():
            for inputs, _, labels in tqdm(test_dataloader, desc="Testing Progress", leave=False):
            
                activations = extract_layer_activations(model, layer, inputs)
                output = decoder(activations)
                
                labels.to(device)
            
                _, predicted = torch.max(output, 1)

            # Collect predictions and true labels for confusion matrix
                predictions_per_layer[layer_idx].extend(predicted.cpu().numpy())
                true_labels_per_layer[layer_idx].extend(labels.cpu().numpy())

            # Compute softmax values for confidence
                outputs_softmax = torch.softmax(output, dim=1)

            # Calculate accuracy, softmax values, and entropy
                for numerosity in range(num_classes):
                    mask = labels == numerosity
                    if mask.sum() > 0:
                        acc = (predicted[mask] == labels[mask]).float().mean().item()
                        mean_softmax = outputs_softmax[mask].mean(dim=0).cpu().numpy()
                        mean_entropy = np.mean([calculate_entropy(p.cpu().numpy()) for p in outputs_softmax[mask]])

                        accuracies[numerosity][layer_idx].append(acc)
                        softmax_values[numerosity][layer_idx].append(mean_softmax)
                        entropy_values[numerosity][layer_idx].append(mean_entropy)

    # Average accuracies, softmax values, and entropy across chunks for each layer
    for numerosity in accuracies:
        for layer_idx in range(len(layers)):
            accuracies[numerosity][layer_idx] = np.mean(accuracies[numerosity][layer_idx])
            softmax_values[numerosity][layer_idx] = np.mean(softmax_values[numerosity][layer_idx], axis=0)
            entropy_values[numerosity][layer_idx] = np.mean(entropy_values[numerosity][layer_idx])

    return accuracies, softmax_values, entropy_values, predictions_per_layer, true_labels_per_layer


def compute_tuning_curves(model, test_dataloader, layer, num_numerosities, layer_idx, plots_save_dir, csv_save_dir):
    """
    Compute and plot the tuning curves for each kernel in a specific layer, showing the average
    response (activation) to different numerosities.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): The DataLoader for the dataset.
        layer (nn.Module): The layer from which to extract the feature maps.
        num_numerosities (int): The number of different numerosities.
        layer_idx (int): The index of the layer being analyzed.
        save_dir (str): The directory to save the plots.
    """
    
    device = torch.device('cuda')
    
    # Create a directory for the tuning curves
    layer_save_dir = os.path.join(plots_save_dir, f'tuning_curves/layer_{layer_idx}')
    os.makedirs(layer_save_dir, exist_ok=True)
    model.eval()

    # Get the number of output channels (kernels) from the layer
    num_kernels = layer.out_channels

    # Initialize the tuning curves dictionary with an empty list for each kernel
    tuning_curves = {kernel_idx: [0] * num_numerosities for kernel_idx in range(num_kernels)}
    counts = np.zeros(num_numerosities)

    # Extract activations for each input
    with torch.no_grad():
        for inputs, _, labels in tqdm(test_dataloader, desc = f'Extracting Feature Maps for Layer {layer_idx}'):
            
            inputs = inputs.to(device)
            
            # Forward pass to get feature maps using the extract_layer_activations function
            
            feature_maps = extract_layer_activations(model, layer, inputs)

            # Average pooling over spatial dimensions
            avg_feature_maps = feature_maps.mean(dim=[2, 3])  # Shape: [batch_size, num_kernels]

            # Update tuning curves for each numerosity
            for i in range(len(labels)):
                numerosity = labels[i].item()

                # Iterate over each kernel and add its mean activation to the corresponding numerosity list
                for kernel_idx in range(num_kernels):
                    tuning_curves[kernel_idx][numerosity] += avg_feature_maps[i, kernel_idx].cpu().item()

                counts[numerosity] += 1
                

    # Normalize the tuning curves by dividing each kernel's response by the counts for each numerosity
    for kernel_idx in range(num_kernels):
        for numerosity in range(num_numerosities):
            if counts[numerosity] > 0:  # Avoid division by zero
                tuning_curves[kernel_idx][numerosity] /= counts[numerosity]

    # Plot tuning curves for each kernel
    plot_tuning_curves(tuning_curves, num_numerosities, layer_save_dir)
    
    # Save the tuning curves as a CSV
    csv_path = os.path.join(csv_save_dir, f'tuning_curves_layer_{layer_idx}.csv')
    save_tuning_curves_as_csv(tuning_curves, csv_path)


def plot_tuning_curves(tuning_curves, num_numerosities, layer_save_dir):
    """
    Plot tuning curves for each kernel and save them in the layer directory.

    Args:
        tuning_curves (dict): Dictionary containing tuning curve data for each kernel.
        num_numerosities (int): The number of different numerosities.
        layer_save_dir (str): The directory to save the plots for the current layer.
    """
    # Ensure the directory for the current layer exists
    os.makedirs(layer_save_dir, exist_ok=True)

    for kernel_idx, curve in tuning_curves.items():
        # Plot the tuning curve for the current kernel
        plt.figure()
        plt.plot(range(num_numerosities), curve, marker='o')
        plt.xlabel('Numerosity')
        plt.ylabel('Average Activation')
        plt.title(f'Tuning Curve for Kernel {kernel_idx}')
        plt.grid()
        plt.tight_layout()

        # Save the plot in the current layer's directory
        plt.savefig(os.path.join(layer_save_dir, f'ST_Num_tuning_curve_kernel_{kernel_idx}.png'))
        plt.close()
        
def save_tuning_curves_as_csv(tuning_curves, file_path):
    """
    Save the tuning curves dictionary as a CSV file.

    Args:
        tuning_curves (dict): The tuning curves dictionary where keys are kernel indices 
                              and values are lists of average activations per numerosity.
        file_path (str): Path to save the CSV file.
    """
    # Convert the tuning curves dictionary to a DataFrame
    df = pd.DataFrame.from_dict(tuning_curves, orient='index')
    df.columns = [f'Numerosity_{i}' for i in range(df.shape[1])]  # Name columns
    df.index.name = 'Kernel'
    
    # Save the DataFrame as a CSV
    df.to_csv(file_path)



def compute_selectivity_index_and_save_csv(model, dataloader, layers, num_numerosities=16, save_dir='selectivity_indices'):
    """
    Compute the selectivity index (Swidth) for each kernel in each layer and save the results as CSV files.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): The DataLoader for the dataset.
        layers (list): The list of layers to analyze.
        num_numerosities (int): The number of different numerosities.
        save_dir (str): The directory where the CSV files will be saved.

    Returns:
        selectivity_indices (dict): A dictionary containing the selectivity index for each kernel in each layer.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
    selectivity_indices = {}

    # Loop over layers
    for layer_idx, layer in enumerate(layers):
        print(f"Analyzing layer {layer_idx}")
        num_kernels = layer.out_channels  # Get the number of kernels in the layer

        # Initialize data structures
        tuning_curves = {kernel_idx: np.zeros(num_numerosities) for kernel_idx in range(num_kernels)}
        counts = np.zeros(num_numerosities)

        # Extract activations for each input
        with torch.no_grad():
            for inputs, _, labels in tqdm(dataloader, desc=f'Extracting Feature Maps for Layer {layer_idx}'):
                feature_maps = extract_layer_activations(model, layer, inputs)

                # Average pooling over spatial dimensions to get [batch_size, num_kernels]
                avg_feature_maps = feature_maps.mean(dim=[2, 3])

                # Update tuning curves for each numerosity
                for i in range(len(labels)):
                    numerosity = labels[i].item()

                    # Iterate over each kernel and add its mean activation to the corresponding numerosity list
                    for kernel_idx in range(num_kernels):
                        tuning_curves[kernel_idx][numerosity] += avg_feature_maps[i, kernel_idx].item()

                    # Track how many times each numerosity appears
                    counts[numerosity] += 1

        # Normalize tuning curves by the counts of each numerosity
        for numerosity in range(num_numerosities):
            if counts[numerosity] > 0:
                for kernel_idx in range(num_kernels):
                    tuning_curves[kernel_idx][numerosity] /= counts[numerosity]

        # Compute the selectivity index for each kernel
        layer_selectivity_indices = []
        layer_numerosity_preferences = []
        for kernel_idx in range(num_kernels):

            responses = tuning_curves[kernel_idx]
            
            responses_tensor = torch.tensor(responses)

            # Find r_max (maximum response of the kernel across all numerosities)
            r_max, max_idx = torch.max(responses_tensor, dim=0)
            max_idx = int(max_idx.item())  # Ensure max_idx is an integer

            mapping = {0:1, 1:2, 2:4, 3:6, 4:8, 5:10, 6:12, 7:14, 8:16, 9:18, 10:20, 11:22, 12:24, 13:26, 14:28, 15:30}

            # Compute the selectivity index using the formula
            if r_max > 0:  # Avoid division by zero
                sum_responses = np.sum(responses)
                selectivity_index = (num_numerosities - (sum_responses / float(r_max))) / (num_numerosities - 1)
                numerosity_preference = mapping[int(max_idx)]
            else:
                selectivity_index = 0  # If the kernel has no response, set selectivity to 0
                numerosity_preference = None

            # Store the selectivity index for this kernel
            layer_selectivity_indices.append(selectivity_index)
            layer_numerosity_preferences.append(numerosity_preference)

        # Save the selectivity indices for this layer as a CSV file
        df = pd.DataFrame(layer_selectivity_indices, columns=[f'Layer_{layer_idx}_Selectivity_Index'])
        df[f'Layer_{layer_idx}_Numerosity_Preference'] = pd.DataFrame(layer_numerosity_preferences)
        csv_path = os.path.join(save_dir, f'MT_selectivity_indices_layer_{layer_idx}_ood.csv')
        df.to_csv(csv_path, index_label='Kernel_Index')
        print(f'Saved selectivity indices for Layer {layer_idx} to {csv_path}')

        # Store in selectivity_indices dictionary for later use
        selectivity_indices[layer_idx] = layer_selectivity_indices

    return selectivity_indices
