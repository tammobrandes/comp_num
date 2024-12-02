import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from scipy.stats import entropy
import json
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from collections import defaultdict
import os
import gc
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, shapiro
from plot_utils import(
    plot_accuracy_vs_selectivity,
    plot_kernel_selectivity_histogram,
    plot_sensitivity, 
    plot_tuning_curves)

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
def summarize_softmax(softmax_results):
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
    
    best_epoch_path = os.path.join(checkpoints_dir, f'model_epoch_{best_epoch}.pth')

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

def precompute_layer_activations_and_labels_from_dataloader(model, dataloader, layer):
    """
    Precompute activations and labels for a specific layer using the provided dataloader,
    while considering only the second label.
    """
    activations = []
    labels = []

    for inputs, _, labels in tqdm(dataloader, desc=f"Extracting Activations for Layer {layer}"):
        layer_activations = extract_layer_activations(model, layer, inputs)
        activations.append(layer_activations.cpu().numpy())
        labels.append(labels.numpy())

    # Concatenate all activations and labels
    activations = np.concatenate(activations)
    labels = np.concatenate(labels)

    return activations, labels


def custom_k_fold_with_two_val_folds(dataset_size = 16000, num_folds = 10):
    """
    Custom splitting function to create train and validation splits
    with two folds used for validation in each iteration, ensuring exactly 10 splits.
    """
    indices = np.arange(dataset_size)
    fold_size = dataset_size // num_folds
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]

    splits = []

    # Create 10 splits with 2 folds for validation
    for i in range(num_folds):
        val_folds = [folds[i], folds[(i + 1) % num_folds]]  # Use 2 consecutive folds as validation
        val_indices = np.concatenate(val_folds)
        train_indices = np.setdiff1d(indices, val_indices)
        splits.append((train_indices, val_indices))

    return splits


def layer_decoding_analysis(
    model, train_val_loader, test_loader, layers, condition, num_classes = 16, dataset_size = 16000, batch_size=32, model_name="model", csv_save_dir=None
):
    """
    Perform decoding analysis with two folds left out for validation,
    considering only the second label from the dataset, and compute validation metrics.
    """
    label_to_numerosity = {
        0: 1, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14,
        8: 16, 9: 18, 10: 20, 11: 22, 12: 24, 13: 26, 14: 28, 15: 30
    }

    # Custom splits with 2 folds left out for validation
    splits = custom_k_fold_with_two_val_folds(dataset_size=dataset_size, num_folds=10)

    final_metrics = []

    for layer in tqdm(layers, desc="Processing Layers"):
        # Precompute activations for the combined train+val set
        train_val_activations, train_val_labels = precompute_layer_activations_and_labels_from_dataloader(
            model, train_val_loader, layer
        )

        # Precompute activations for the test set
        test_activations, test_labels = precompute_layer_activations_and_labels_from_dataloader(
            model, test_loader, layer
        )

        # SVM models and validation results across folds
        svms = []
        validation_accuracies = []

        for split_idx, (train_indices, val_indices) in enumerate(tqdm(splits, desc="Custom Splits", leave=False)):
            # Split the precomputed activations and labels
            train_activations = train_val_activations[train_indices]
            train_labels = train_val_labels[train_indices]

            val_activations = train_val_activations[val_indices]
            val_labels = train_val_labels[val_indices]

            # Train SVM
            svm = SVC(kernel='linear', probability=True)
            svm.fit(train_activations, train_labels)
            svms.append(svm)

            # Evaluate on the validation set
            val_predictions = svm.predict(val_activations)
            val_accuracy = accuracy_score(val_labels, val_predictions)
            validation_accuracies.append(val_accuracy)

        # Model averaging (coefficients and intercepts)
        avg_coef = np.mean([svm.coef_ for svm in svms], axis=0)
        avg_intercept = np.mean([svm.intercept_ for svm in svms], axis=0)

        # Create the averaged SVM
        averaged_svm = SVC(kernel='linear', probability=True)
        averaged_svm.coef_ = avg_coef
        averaged_svm.intercept_ = avg_intercept

        # Evaluate on the test set
        predictions = averaged_svm.predict(test_activations)
        softmax_outputs = averaged_svm.predict_proba(test_activations)

        # Compute test set metrics
        accuracy = accuracy_score(test_labels, predictions)
        conf_matrix = confusion_matrix(test_labels, predictions).tolist()
        report = classification_report(test_labels, predictions, target_names=[f"Class {i}" for i in range(num_classes)], output_dict=True)

        # Compute per-class metrics like entropy and error
        entropy_per_class = {}
        error_per_class = {}
        for cls in range(num_classes):
            true_mask = (test_labels == cls)
            incorrect_mask = (predictions != test_labels) & true_mask
            predicted_labels_for_class = predictions[incorrect_mask]
            true_numerosity = label_to_numerosity[cls]

            # Compute entropy
            class_probs = softmax_outputs[true_mask]
            class_entropy = np.mean([entropy(prob) for prob in class_probs]) if len(class_probs) > 0 else None
            entropy_per_class[f"Class {cls}"] = class_entropy

            # Compute error ignoring correct predictions
            if len(predicted_labels_for_class) > 0:
                predicted_numerosities = [label_to_numerosity[pred] for pred in predicted_labels_for_class]
                absolute_errors = [abs(predicted - true_numerosity) for predicted in predicted_numerosities]
                error_per_class[f"Class {cls}"] = np.mean(absolute_errors)
            else:
                error_per_class[f"Class {cls}"] = None

        # Store metrics
        final_metrics.append({
            'layer': str(layer),
            'validation_accuracies': validation_accuracies,
            'test_accuracy': accuracy,
            'conf_matrix': conf_matrix,
            'report': report,
            'entropy_per_class': entropy_per_class,
            'error_per_class': error_per_class
        })
        
    os.makedirs(csv_save_dir, exist_ok=True)

    with open(os.path.join(csv_save_dir, f"{model_name}{condition}_layer_decoding.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)


def compute_tuning_curves(model, dataloader, layer, num_numerosities, layer_idx, plots_dir, csv_dir):
    """
    Compute tuning curves for a given layer and save them as plots and CSV files,
    including summary statistics like standard deviation and variance.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): DataLoader for the test set.
        layer (nn.Module): The layer to analyze.
        num_numerosities (int): Number of numerosities.
        layer_idx (int): Index of the layer.
        plots_dir (str): Directory to save tuning curve plots.
        csv_dir (str): Directory to save tuning curve data as CSV.

    Returns:
        None
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    # Initialize storage for tuning curves and statistics
    tuning_curves = {kernel_idx: np.zeros(num_numerosities) for kernel_idx in range(layer.out_channels)}
    all_kernel_activations = {kernel_idx: [[] for _ in range(num_numerosities)] for kernel_idx in range(layer.out_channels)}
    counts = np.zeros(num_numerosities)

    with torch.no_grad():
        for inputs, _, labels in dataloader:
            feature_maps = extract_layer_activations(model, layer, inputs)
            avg_feature_maps = feature_maps.mean(dim=[2, 3])
            
            for i in range(len(labels)):
                numerosity = labels[i].item()
                for kernel_idx in range(layer.out_channels):
                    tuning_curves[kernel_idx][numerosity] += avg_feature_maps[i, kernel_idx].item()
                    all_kernel_activations[kernel_idx][numerosity].append(avg_feature_maps[i, kernel_idx].item())
                counts[numerosity] += 1
    
    # Normalize tuning curves
    for numerosity in range(num_numerosities):
        if counts[numerosity] > 0:
            for kernel_idx in range(layer.out_channels):
                tuning_curves[kernel_idx][numerosity] /= counts[numerosity]

    # Initialize summary statistics arrays
    summary_stats = {
        'mean': np.zeros((layer.out_channels, num_numerosities)),
        'std': np.zeros((layer.out_channels, num_numerosities)),
        'variance': np.zeros((layer.out_channels, num_numerosities)),
        'min': np.zeros((layer.out_channels, num_numerosities)),
        'max': np.zeros((layer.out_channels, num_numerosities))
    }
    
    # Calculate summary statistics (mean, std, variance)
    for kernel_idx in range(layer.out_channels):
        for numerosity in range(num_numerosities):
            activations = np.array(all_kernel_activations[kernel_idx][numerosity])
            if len(activations) > 0:
                summary_stats['mean'][kernel_idx, numerosity] = np.mean(activations)
                summary_stats['std'][kernel_idx, numerosity] = np.std(activations)
                summary_stats['variance'][kernel_idx, numerosity] = np.var(activations)
                summary_stats['min'][kernel_idx, numerosity] = np.min(activations)
                summary_stats['max'][kernel_idx, numerosity] = np.max(activations)

    # Prepare data for CSV: column-wise
    csv_data = []
    for kernel_idx in range(layer.out_channels):
        for numerosity in range(num_numerosities):
            row = [
                kernel_idx,  # Kernel ID
                numerosity,  # Numerosity
                summary_stats['mean'][kernel_idx, numerosity],  # Mean
                summary_stats['std'][kernel_idx, numerosity],  # Std
                summary_stats['variance'][kernel_idx, numerosity],  # Variance
                summary_stats['min'][kernel_idx, numerosity],  # Min
                summary_stats['max'][kernel_idx, numerosity]   # Max
            ]
            csv_data.append(row)
    
    # Convert to DataFrame and save as CSV
    csv_path = os.path.join(csv_dir, f'tuning_curves_and_stats_layer_{layer_idx}.csv')
    csv_df = pd.DataFrame(csv_data, columns=['kernel id', 'numerosity', 'mean', 'std', 'variance', 'min', 'max'])
    csv_df.to_csv(csv_path, index=False)
    print(f'Saved tuning curves and summary statistics for layer {layer_idx} to {csv_path}')
    
    # Plot tuning curves (you can leave this part unchanged if it works as expected)
    plot_tuning_curves(tuning_curves, num_numerosities, plots_dir)



        
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



def compute_selectivity_index_and_save_csv(model, model_name, dataloader, condition, layers, num_numerosities, save_dir):
    """
    Compute the selectivity index for all kernels in given layers and save as CSV.

    Args:
        model (nn.Module): The trained model.
        model_name (str): Name of the model.
        dataloader (DataLoader): DataLoader for the dataset.
        condition (str): Experimental condition (e.g., '', '_samesize', '_ood').
        layers (list): List of layers to analyze.
        num_numerosities (int): Number of numerosities.
        save_dir (str): Directory to save CSV files.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset_name = 'Base' if condition == '' else ('SameSize' if condition == '_samesize' else 'OOD')
    
    for layer_idx, layer in enumerate(layers):
        print(f"Analyzing layer {layer_idx}")
        tuning_curves = {kernel_idx: np.zeros(num_numerosities) for kernel_idx in range(layer.out_channels)}
        counts = np.zeros(num_numerosities)

        with torch.no_grad():
            for inputs, _, labels in dataloader:
                feature_maps = extract_layer_activations(model, layer, inputs)
                avg_feature_maps = feature_maps.mean(dim=[2, 3])
                
                for i in range(len(labels)):
                    numerosity = labels[i].item()
                    for kernel_idx in range(layer.out_channels):
                        tuning_curves[kernel_idx][numerosity] += avg_feature_maps[i, kernel_idx].item()
                    counts[numerosity] += 1
        
        # Normalize tuning curves
        for numerosity in range(num_numerosities):
            if counts[numerosity] > 0:
                for kernel_idx in range(layer.out_channels):
                    tuning_curves[kernel_idx][numerosity] /= counts[numerosity]

        # Compute selectivity index
        layer_selectivity_indices = []
        layer_numerosity_preferences = []
        for kernel_idx in range(layer.out_channels):
            responses = tuning_curves[kernel_idx]
            r_max = max(responses)
            selectivity_index = (num_numerosities - sum(responses) / r_max) / (num_numerosities - 1) if r_max > 0 else 0
            numerosity_preference = np.argmax(responses)
            layer_selectivity_indices.append(selectivity_index)
            layer_numerosity_preferences.append(numerosity_preference)
        
        # Save to CSV
        df = pd.DataFrame({
            'Selectivity_Index': layer_selectivity_indices,
            'Numerosity_Preference': layer_numerosity_preferences,
            'Layer': layer_idx,
            'Dataset': dataset_name
        })
        csv_path = os.path.join(save_dir, f'{model_name}_selectivity_indices_layer_{layer_idx}_{dataset_name}.csv')
        df.to_csv(csv_path, index_label='Kernel_Index')
        print(f"Saved selectivity indices for layer {layer_idx} to {csv_path}")


def estimate_correlation(model_name, dataset_name, data_path):
    """
    Estimates the correlation between the percentage of selective neurons and model accuracy for the specified model and dataset.
    Performs a normality test and uses Pearson or Spearman correlation accordingly, returning the correlation coefficient, p-value, and test used.
    Also returns p-values from the normality test for both variables.
    
    Parameters:
    - model_name (str): Name of the model ('MT', 'ST_Num', 'ST_Obj')
    - dataset_name (str or None): Name of the dataset to filter ('base', 'ood', 'ss'). If None, all datasets will be used.
    - data_path (str): Path to the directory containing the CSV files.
    
    Returns:
    - correlations (dict): Dictionary with dataset names as keys and the correlation results as values (correlation, p-value, test used, normality p-values).
    """
    
    # Mapping of dataset names to actual file naming conventions
    dataset_mapping = {
        'base': 'base',
        'ood': 'AltDatasetOOD',
        'ss': 'AltDatasetLarger_SameSize'
    }
    
    # If dataset_name is provided, filter to only that dataset, otherwise use all
    if dataset_name:
        dataset_names = [dataset_mapping[dataset_name]]
    else:
        dataset_names = dataset_mapping.values()

    # Initialize dictionary to store correlations
    correlations = {}

    # Loop through each dataset
    for dataset in dataset_names:
        file_name = f"numerosity_accuracy_model_{model_name}_{dataset}.csv"
        file_path = os.path.join(data_path, file_name)

        # Check if the file exists
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # The first column is the number of selective neurons, unnamed
        num_selective_neurons = df['Kernel_Count']
        
        # The 'accuracy' column contains model accuracies
        accuracies = df['Accuracy']
        
        # Calculate the percentage of selective neurons (out of total neurons)
        total_neurons = num_selective_neurons.sum()
        percentage_selective_neurons = num_selective_neurons 
        
        # Perform normality tests (Shapiro-Wilk test)
        stat_neurons, p_value_neurons = shapiro(percentage_selective_neurons)
        stat_accuracy, p_value_accuracy = shapiro(accuracies)
        
        # Use Pearson if both datasets are normally distributed, otherwise use Spearman
        if p_value_neurons > 0.05 and p_value_accuracy > 0.05:
            # Normally distributed, use Pearson correlation
            correlation, p_value_corr = pearsonr(percentage_selective_neurons, accuracies)
            test_used = 'Pearson'
        else:
            # Not normally distributed, use Spearman correlation
            correlation, p_value_corr = spearmanr(percentage_selective_neurons, accuracies)
            test_used = 'Spearman'
        
        # Store the result in the dictionary with correlation, p-value, test used, and normality p-values
        correlations[dataset] = {
            'correlation': correlation,
            'p_value_corr': p_value_corr,
            'test_used': test_used,
            'normality_p_value_neurons': p_value_neurons,
            'normality_p_value_accuracy': p_value_accuracy
        }
    
    return correlations

def compute_acc_sel_corr(sel_thr1, sel_thr2, numerosity_accuracy,model_name, dataset_name, csv_dir, plot_dir):
    # Should be changed to extract from save file
    df_sel_index = pd.read_csv(os.path.join(csv_dir, f'{model_name}_final_concatenated_selectivity_indices.csv'))
    df_numerosities = pd.read_csv(os.path.join(csv_dir,f'{model_name}_numerosity_accuracies.csv'))
    num_accuracies = df_numerosities['accuracy']
    if dataset_name == '_samesize':
        dataset = 'ss'
    elif dataset_name == '_ood':
        dataset = 'ood'
    else:
        dataset = 'base'
    plot_accuracy_vs_selectivity(numerosity_accuracy, df_sel_index, dataset, sel_thr1, sel_thr2, plot_dir, model_name)
    plot_kernel_selectivity_histogram(df_sel_index, dataset,5, sel_thr1, sel_thr2,plot_dir, model_name, num_accuracies)
    plot_sensitivity(plot_dir)
    correlations = estimate_correlation('MT', 'base', os.path.join(csv_dir, 'num_accuracies'))
    print(correlations)