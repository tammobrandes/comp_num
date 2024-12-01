import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis_utils import calculate_entropy
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_colormap(num_classes):
    """
    Generate a colormap with darker shades for higher classes.
    """
    colormap = plt.cm.viridis(np.linspace(0, 1, num_classes))
    return colormap

# Plot Train and Validation Losses
def plot_train_val_losses(history_df, plots_save_dir):
    plt.figure(figsize=(10, 6))

    # Plot Loss1 for training and validation
    plt.plot(history_df['epoch'], history_df['train_loss1'], label='OBJ Train Loss', linestyle='-', color='blue')
    plt.plot(history_df['epoch'], history_df['val_loss1'], label='OBJ Val Loss', linestyle='--', color='blue')

    # Plot Loss2 for training and validation
    plt.plot(history_df['epoch'], history_df['train_loss2'], label='NUM Train Loss', linestyle='-', color='green')
    plt.plot(history_df['epoch'], history_df['val_loss2'], label='NUM Val Loss', linestyle='--', color='green')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(plots_save_dir, 'train_val_losses.png'))
    plt.close()

# Plot Train and Validation Accuracies
def plot_train_val_accuracies(history_df, plots_save_dir):
    plt.figure(figsize=(10, 6))

    # Plot Accuracy1 for training and validation
    plt.plot(history_df['epoch'], history_df['train_acc1'], label='OBJ Train Acc', linestyle='-', color='blue')
    plt.plot(history_df['epoch'], history_df['val_acc1'], label='OBJ Val Acc', linestyle='--', color='blue')

    # Plot Accuracy2 for training and validation
    plt.plot(history_df['epoch'], history_df['train_acc2'], label='NUM Train Acc', linestyle='-', color='green')
    plt.plot(history_df['epoch'], history_df['val_acc2'], label='NUM Val Acc', linestyle='--', color='green')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracies')
    plt.legend()
    plt.savefig(os.path.join(plots_save_dir, 'train_val_accuracies.png'))
    plt.close()

# Plot Test Accuracy by Numerosity
def plot_test_accuracy_by_numerosity(condition, test_results, plots_save_dir):
    numerosity_classes = sorted(set([num for num, acc in test_results]))
    accuracies = [acc for num, acc in sorted(test_results, key=lambda x: x[0])]

    plt.figure(figsize=(10, 6))
    plt.bar(numerosity_classes, accuracies, color='blue')
    plt.xlabel('Numerosity')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy by Numerosity')
    plt.savefig(os.path.join(plots_save_dir, f'test_accuracy_by_numerosity{condition}.png'))
    plt.close()

# Plot Test Accuracy by Object Type
def plot_test_accuracy_by_object(condition, test_results, plots_save_dir):
    object_classes = sorted(set([obj for obj, acc in test_results]))
    accuracies = [acc for obj, acc in sorted(test_results, key=lambda x: x[0])]

    plt.figure(figsize=(10, 6))
    plt.bar(object_classes, accuracies, color='green')
    plt.xlabel('Object Classes')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy by Object Type')
    plt.savefig(os.path.join(plots_save_dir, f'test_accuracy_by_object{condition}.png'))
    plt.close()

# Combined Numerosity Accuracies by Object Type
def plot_combined_numerosity_accuracies_by_object(condition, test_results, plots_save_dir):
    """
    Plot combined numerosity accuracies for each object type.
    test_results: List of tuples (softmax, object_type, ground_truth_numerosity)
    """
    # Convert softmax vectors to predicted numerosity classes using argmax
    numerosity_classes = sorted(set([np.argmax(num) for num, _, _ in test_results]))  # Use argmax on softmax values
    object_types = sorted(set([obj for _, obj, _ in test_results]))

    plt.figure(figsize=(10, 6))

    for obj in object_types:
        accuracies = []
        for numerosity in numerosity_classes:
            relevant_results = [(np.argmax(num), num_label) for num, obj_label, num_label in test_results if obj_label == obj]
            correct_predictions = sum([1 for pred_num, true_num in relevant_results if pred_num == true_num])
            total_predictions = len(relevant_results)
            if total_predictions > 0:
                accuracies.append(correct_predictions / total_predictions)
            else:
                accuracies.append(0)

        plt.plot(numerosity_classes, accuracies, label=f'Object Type {obj}')
    
    plt.xlabel('Numerosity Classes')
    plt.ylabel('Accuracy')
    plt.title('Numerosity Accuracies by Object Type')
    plt.legend()
    plt.savefig(os.path.join(plots_save_dir, f'numerosity_accuracies_by_object{condition}.png'))
    plt.close()

# Plot softmax distributions for each object type and numerosity
def plot_softmax_distributions(condition, summarized_data, plots_save_dir, num_numerosities=16):
    """
    Plot softmax distributions for each object type across different numerosities,
    using a colormap where higher numerosities have darker shades.
    """
    object_types = sorted(set(key[1] for key in summarized_data.keys()))
    fig, axs = plt.subplots(len(object_types), 1, figsize=(10, 40), sharex=True)
    colormap = get_colormap(num_numerosities)  # Get colormap for numerosities

    for i, object_type in enumerate(object_types):
        ax = axs[i]
        for numerosity in range(num_numerosities):
            if (numerosity, object_type) in summarized_data:
                color = colormap[numerosity]  # Set color based on numerosity
                ax.plot(range(num_numerosities), summarized_data[(numerosity, object_type)], label=f'Numerosity {numerosity + 1}', color=color)
        
        ax.set_title(f'Object Type: {object_type}')
        ax.set_xlabel('Numerosity')
        ax.set_ylabel('Softmax Value')
        ax.legend(title='Numerosity', loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_save_dir, f'softmax_distributions{condition}.png'))
    plt.close()


# Plot entropy of softmax distributions vs numerosity
def plot_entropy_vs_numerosity(condition, summarized_data, plots_save_dir, num_numerosities=16):
    plt.figure(figsize=(10, 6))
    object_types = sorted(set(key[1] for key in summarized_data.keys()))  # Get unique object types
    
    for object_type in object_types:
        entropies = []
        for numerosity in range(num_numerosities):
            if (numerosity, object_type) in summarized_data:
                softmax_values = summarized_data[(numerosity, object_type)]
                # Calculate entropy for the softmax values
                entropy = calculate_entropy(softmax_values)
                entropies.append(entropy)
            else:
                entropies.append(0)  # Default to 0 if there's no data

        # Plot entropy vs numerosity for each object type
        plt.plot(range(num_numerosities), entropies, label=f'Object Type: {object_type}')
    
    plt.title('Entropy vs Numerosity for Each Object Type')
    plt.xlabel('Numerosity')
    plt.ylabel('Entropy (Uncertainty)')
    plt.legend()
    plt.savefig(os.path.join(plots_save_dir, f'entropy_vs_numerosity{condition}.png'))
    plt.close()

def plot_decoding_accuracy_vs_layer(condition, accuracies, layers, plots_save_dir, num_numerosities=16):
    """
    Plot decoding accuracy with numerosity on the x-axis and accuracy on the y-axis,
    using a colormap where deeper layers have darker shades.
    """
    plt.figure(figsize=(12, 8))
    colormap = get_colormap(len(layers))  # Get colormap for layers

    # Plot accuracy for each layer
    for layer_idx in range(len(layers)):
        color = colormap[layer_idx]
        accuracy_list = [accuracies[numerosity][layer_idx] for numerosity in range(num_numerosities)]
        plt.plot(range(num_numerosities), accuracy_list, label=f'Layer {layer_idx}', color=color)

    plt.xlabel('Numerosity')
    plt.ylabel('Decoding Accuracy')
    plt.title('Decoding Accuracy vs. Numerosity for Different Layers')
    plt.xticks(range(num_numerosities), [str(n) for n in range(1, num_numerosities + 1)], rotation=45)
    plt.legend(title='Layers')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_save_dir, f'decoding_accuracy_vs_numerosity{condition}.png'))
    plt.close()


def plot_softmax_confidence_vs_layer(condition, softmax_values, layers, plots_save_dir, num_numerosities=16):
    """
    Plot softmax confidence with numerosity on the x-axis and softmax confidence on the y-axis,
    using a colormap where deeper layers have darker shades.
    """
    plt.figure(figsize=(12, 8))
    colormap = get_colormap(len(layers))  # Get colormap for layers

    # Plot softmax confidence for each layer
    for layer_idx in range(len(layers)):
        color = colormap[layer_idx]
        confidence_list = [softmax_values[numerosity][layer_idx][numerosity] for numerosity in range(num_numerosities)]
        plt.plot(range(num_numerosities), confidence_list, label=f'Layer {layer_idx}', color=color)

    plt.xlabel('Numerosity')
    plt.ylabel('Softmax Confidence')
    plt.title('Softmax Confidence vs. Numerosity for Different Layers')
    plt.xticks(range(num_numerosities), [str(n) for n in range(1, num_numerosities + 1)], rotation=45)
    plt.legend(title='Layers')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_save_dir, f'layerwise_softmax_confidence_vs_numerosity{condition}.png'))
    plt.close()


def plot_layerwise_entropy_vs_numerosity(condition, entropy_values, num_classes, layers, plots_save_dir):
    """
    Plot the entropy for different layers on the same plot, with different layers as separate lines.
    Deeper layers will have darker shades.
    """
    plt.figure(figsize=(12, 8))
    colormap = get_colormap(len(layers))  # Get colormap for layers

    # Plot entropy for each layer
    for layer_idx in range(len(layers)):
        color = colormap[layer_idx]
        entropy_list = [entropy_values[numerosity][layer_idx] for numerosity in range(num_classes)]
        plt.plot(range(num_classes), entropy_list, label=f'Layer {layer_idx}', color=color)

    plt.xlabel('Numerosity')
    plt.ylabel('Entropy')
    plt.title('Entropy vs. Numerosity for Different Layers')
    plt.xticks(range(num_classes), [str(n) for n in range(1, num_classes + 1)], rotation=45)
    plt.legend(title='Layers')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_save_dir, f'layerwise_entropy_vs_numerosity{condition}.png'))
    plt.close()

def plot_confusion_matrices_per_layer(condition, predictions_per_layer, true_labels_per_layer, layers, num_classes, plots_save_dir, csv_save_dir):
    """
    Plot confusion matrix for each layer and save the plots.
    """

    for layer_idx, (predictions, true_labels) in enumerate(zip(predictions_per_layer, true_labels_per_layer)):
        # Compute the confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=list(range(num_classes)))

        # Save the confusion matrix as an array (CSV format)
        cm_array_path = os.path.join(csv_save_dir, f'confusion_matrix_layer_{layer_idx}{condition}.csv')
        pd.DataFrame(cm).to_csv(cm_array_path, index=False, header=False)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1, num_classes + 1), yticklabels=range(1, num_classes + 1))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Layer {layer_idx}')
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(plots_save_dir, f'confusion_matrix_layer_{layer_idx}{condition}.png'))
        plt.close()


def plot_softmax_values_per_layer(condition, softmax_values, plots_save_dir, num_numerosities=16, layers=None):
    """
    Plot the softmax values for each numerosity per layer.
    
    Args:
        softmax_values (dict): The extracted softmax values for each numerosity per layer.
        num_numerosities (int): Number of different numerosity classes.
        layers (list): The list of layers (for naming purposes).
        save_dir (str): The directory where the plots will be saved.
    """
    colormap = get_colormap(num_numerosities)  # Get colormap for different numerosities

    # Iterate over layers
    for layer_idx in range(len(layers)):
        plt.figure(figsize=(12, 8))

        # Iterate over each true numerosity
        for true_numerosity in range(num_numerosities):
            if true_numerosity in softmax_values:
                # Get the average softmax values for the current layer and true numerosity
                avg_softmax = softmax_values[true_numerosity][layer_idx]

                # Plot the softmax distribution
                color = colormap[true_numerosity]
                plt.plot(range(num_numerosities), avg_softmax, label=f'True Numerosity {true_numerosity + 1}', color=color)


        # Plot formatting
        plt.xlabel('Predicted Numerosity')
        plt.ylabel('Average Softmax Value')
        plt.title(f'Softmax Distribution by Predicted Numerosity for Layer {layer_idx}')
        plt.xticks(range(num_numerosities), [str(n) for n in range(1, num_numerosities + 1)], rotation=45)
        plt.legend(title='True Numerosity', loc='upper right')
        plt.grid()
        plt.tight_layout()
        
        # Save the plot
        # layer_name = layers[layer_idx] if layers else f'Layer_{layer_idx}'
        plt.savefig(os.path.join(plots_save_dir, f'softmax_distribution_{layer_idx}{condition}.png'))
        plt.close()



