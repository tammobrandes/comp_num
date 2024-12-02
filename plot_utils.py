import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

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
    from analysis_utils import calculate_entropy
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
        plt.tight_layout()

        # Save the plot in the current layer's directory
        plt.savefig(os.path.join(layer_save_dir, f'ST_Num_tuning_curve_kernel_{kernel_idx}.png'))
        plt.close()

def plot_sensitivity(base_dir, output_dir):
    # Load the data for each model
    df_models = {
        'MT': pd.read_csv(os.path.join(base_dir, 'MT_final_concatenated_selectivity_indices.csv')),
        'ST_Num': pd.read_csv(os.path.join(base_dir, 'ST_Num_final_concatenated_selectivity_indices.csv')),
        'ST_Obj': pd.read_csv(os.path.join(base_dir, 'ST_Obj_final_concatenated_selectivity_indices.csv'))
    }
 
    # Plot the averaged selectivity index distributions
    plot_averaged_selectivity_index_distributions(df_models, output_dir)
    for model_name in ['MT', 'ST_Num', 'ST_Obj']:
        for dataset in ['base', 'ood', 'ss']:
            plot_selectivity_pie_chart(df_models[model_name], dataset, output_dir, model_name)
    print(f"Plots saved in {output_dir}")
    
def plot_preferred_numerosity(df, dataset, layer, selectivity_threshold, output_dir, model_name):
    filtered_df = df[(df['Dataset'] == dataset) & 
                     (df['Layer'] == layer) & 
                     (df['selectivity_index'] > selectivity_threshold)]
    
    total_Units = len(filtered_df)
    numerosity_counts = filtered_df['Prefered_Numerosity'].value_counts().sort_index()
    numerosity_percentages = (numerosity_counts / total_Units) * 100
    
    plt.figure(figsize=(8, 6))
    plt.bar(numerosity_percentages.index, numerosity_percentages.values, color='steelblue', edgecolor='black')
    plt.xlabel('Preferred numerosity', fontsize=12)
    plt.ylabel('Percentage of Units', fontsize=12)
    plt.title(f'n = {total_Units}', fontsize=14)
    plt.xticks(range(0, 35, 5), fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, f'{model_name}_{dataset}_Layer_{layer}_Preferred_Numerosity.png')
    plt.savefig(file_path)
    plt.close()

def plot_number_selective_Units_heatmap(df, dataset, selectivity_threshold, output_dir, model_name):
    # Existing code for heatmap
    filtered_df = df[(df['Dataset'] == dataset) & (df['selectivity_index'] > selectivity_threshold)]
    Unit_counts = filtered_df.groupby(['Layer', 'Prefered_Numerosity']).size().unstack(fill_value=0)
    Unit_percentages = Unit_counts.div(Unit_counts.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(Unit_percentages, annot=True, fmt=".1f", cbar_kws={'label': 'Percentage of Units'})
    plt.xlabel('Preferred Numerosity', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.title(f'Percentage of Unit Selectivity by Layer and Numerosity ({dataset} dataset)', fontsize=14)
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, f'{model_name}_{dataset}_Number_Selective_Units_Heatmap.png')
    plt.savefig(file_path)
    plt.close()

def plot_averaged_selectivity_index_distributions(df_models,output_dir):
    # Colors for each model
    model_colors = {'MT': 'blue', 'ST_Num': 'orange', 'ST_Obj': 'green'}
    
    # Create the figure with 3 columns and 6 rows
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 18), sharex=True, sharey=True)

    # Set the titles for the columns (models)
    column_titles = ['MultiTask Net', 'Num Net', 'Obj Net']

    # Loop over the models and layers to plot averaged selectivity index distribution
    for col_idx, (model_name, df) in enumerate(df_models.items()):
        for layer in range(6):
            ax = axes[layer, col_idx]

            # Filter the dataframe for the current layer
            layer_df = df[df['Layer'] == layer]

            # Group by kernel (unit) and calculate the mean selectivity index across datasets
            averaged_df = layer_df.groupby('Kernel_Index').agg({'selectivity_index': 'mean'}).reset_index()

            # Plot the averaged selectivity index distribution
            sns.histplot(averaged_df['selectivity_index'], bins=30, kde=True, color=model_colors[model_name], 
                         edgecolor='black', ax=ax)

            # Add a title for each column (model names)
            if layer == 0:
                ax.set_title(column_titles[col_idx], fontsize=14)

            # Add y-axis label for "Number of Units" to each plot
            ax.set_ylabel('Number of Units', fontsize=10)

            # Customize x-axis labels and grid
            ax.set_xlabel('Averaged Selectivity Index', fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Add layer number as text on the first column only
            if col_idx == 0:
                ax.text(-0.2, 0.5, f'Layer {layer + 1}', fontsize=16, ha='center', va='center', 
                        rotation=90, transform=ax.transAxes)

    # Adjust layout to fit everything
    plt.tight_layout(rect=[0.05, 0, 1, 1])

    # Save the figure
    file_path = os.path.join(output_dir, 'Averaged_Selectivity_Index_Distributions_Models_Layers.png')
    plt.savefig(file_path)
    plt.close()

    print(f"Plot saved to {file_path}")

    
def plot_selectivity_pie_chart(df, dataset, output_dir, model_name):
    # Filter the DataFrame by the specified dataset
    filtered_df = df[df['Dataset'] == dataset]

    # Thresholds for selectivity index
    not_selective_threshold = 0.3
    low_selectivity_threshold = 0.6
    high_selectivity_threshold = 0.9

    # Create groups for selectivity_index == 0 and selectivity_index == 1
    zero_selectivity = filtered_df[filtered_df['selectivity_index'] == 0].shape[0]

    # Existing groups based on selectivity thresholds
    not_selective = filtered_df[(filtered_df['selectivity_index'] < not_selective_threshold) & 
                                (filtered_df['selectivity_index'] > 0)].shape[0]
    low_selective = filtered_df[(filtered_df['selectivity_index'] >= not_selective_threshold) & 
                                (filtered_df['selectivity_index'] < low_selectivity_threshold)].shape[0]
    high_selective = filtered_df[(filtered_df['selectivity_index'] >= low_selectivity_threshold) & 
                                 (filtered_df['selectivity_index'] < high_selectivity_threshold)].shape[0]
    perfect_selectivity = filtered_df[filtered_df['selectivity_index'] >= high_selectivity_threshold].shape[0]

    # Define labels and sizes for the pie chart
    labels = ['Not Selective (0)', 'Low Selectivity (0 - 0.3)', 
              'Moderate Selectivity (0.3 - 0.6)', 'High Selectivity (0.6 - 0.9)', 'Perfect Selectivity (0.9 - 1.0)']
    sizes = [zero_selectivity, not_selective, low_selective, high_selective, perfect_selectivity]
    colors = ['lightblue', 'lightcoral', 'gold', 'lightgreen', 'darkgreen']

    # Create the pie chart
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
    plt.title(f'Unit Selectivity Distribution ({model_name}, {dataset} dataset)', fontsize=14)

    # Save the plot
    plt.tight_layout()
    file_path = os.path.join(output_dir, f'{model_name}_{dataset}_Selectivity_Pie_Chart.png')
    plt.savefig(file_path)
    plt.close()

def plot_selectivity_index_distribution(df, dataset, layer, output_dir, model_name):
    # Existing code for selectivity index distribution
    filtered_df = df[(df['Dataset'] == dataset) & (df['Layer'] == layer)]

    plt.figure(figsize=(8, 6))
    sns.histplot(filtered_df['selectivity_index'], bins=30, kde=True, color='purple', edgecolor='black')
    plt.xlabel('Selectivity Index', fontsize=12)
    plt.ylabel('Number of Units', fontsize=12)
    plt.title(f'Distribution of Selectivity Indices for {dataset} - Layer {layer}', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    file_path = os.path.join(output_dir, f'{model_name}_{dataset}_Layer_{layer}_Selectivity_Index_Distribution.png')
    plt.savefig(file_path)
    plt.close()

def plot_accuracy_vs_selectivity(accuracies, df, dataset, selectivity_threshold, selectivity_threshold_up,output_dir, model_name):
    # Filter DataFrame based on selectivity threshold
    filtered_df = df[(df['Dataset'] == dataset) & (df['selectivity_index'] >= selectivity_threshold) & (df['Layer'] == 5) & (df['selectivity_index'] < selectivity_threshold_up)]

    # Calculate total units and percentages for each numerosity
    total_units = len(filtered_df)
    numerosity_counts = filtered_df['Prefered_Numerosity'].value_counts().sort_index()
    numerosity_percentages = (numerosity_counts / total_units) * 100

    # Prepare data for plotting
    x_values = []
    y_values = []
    numerosity_labels = []

    for num in range(1, 16):  # Assuming numerosity ranges from 1 to 15
        if num in accuracies:
            x_values.append(numerosity_percentages.get(num, 0))  # Get percentage or 0 if not present
            y_values.append(accuracies[num])
            numerosity_labels.append(num)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_values, y_values, c=numerosity_labels, cmap='viridis', s=100, edgecolor='black')
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Numerosity', fontsize=12)

    # Labeling the plot
    plt.xlabel('Percentage of Selective Neurons (%)', fontsize=12)
    plt.ylabel('Model Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs. Percentage of Selective Neurons ({dataset} dataset)', fontsize=14)

    # Display grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    plt.tight_layout()
    file_path = os.path.join(output_dir, f'{model_name}_{dataset}_Accuracy_vs_Selectivity_Scatter.png')
    plt.savefig(file_path)
    plt.close()


def plot_kernel_selectivity_histogram(df, dataset, layer, selectivity_threshold, selectivity_threshold_up, output_dir, model_name, numerosity_accuracy):
    """
    Plots a histogram of the number of kernels selectively tuned to each numerosity.
    Additionally, saves the numerosity counts and corresponding model accuracies.

    Parameters:
    - df: DataFrame containing the kernel data, including 'Prefered_Numerosity' and 'selectivity_index'.
    - dataset: Dataset name to filter the dataframe.
    - layer: The layer to filter kernels from.
    - selectivity_threshold: Lower threshold for the selectivity index.
    - selectivity_threshold_up: Upper threshold for the selectivity index.
    - output_dir: Directory to save the plot and results.
    - model_name: Name of the model being analyzed.
    - numerosity_accuracy: List or array containing accuracies for each numerosity from 1 to 30.
    """

    # Filter DataFrame based on selectivity index threshold and the layer
    filtered_df = df[(df['Dataset'] == dataset) & (df['Layer'] == layer) & 
                     (df['selectivity_index'] >= selectivity_threshold) & 
                     (df['selectivity_index'] < selectivity_threshold_up)]


    # Replace preferred numerosity values using the ma
    numerosities = [1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    # Count the number of kernels selectively tuned to each numerosity
    numerosity_counts = filtered_df['Prefered_Numerosity'].value_counts().reindex(numerosities, fill_value=0).sort_index()

    # Create a DataFrame to store numerosity counts and accuracies
    results_df = pd.DataFrame({
        'Numerosity': numerosities,
        'Kernel_Count': numerosity_counts.values,
        'Accuracy': numerosity_accuracy  # Assuming numerosity_accuracy is indexed from 1 to 30
    })

    # Save results to a CSV file
    results_file_path = os.path.join(output_dir, f'numerosity_accuracy_{model_name}_{dataset}.csv')
    results_df.to_csv(results_file_path, index=False)

    # Set colors for the histogram based on the model name
    color_map = {
        'model_MT': 'blue',
        'model_ST_Num': 'green',
        'model_ST_Obj': 'orange'
    }
    bar_color = color_map.get(model_name, 'gray')  # Default to gray if model name doesn't match

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(numerosity_counts.index, numerosity_counts.values, color=bar_color, edgecolor='black')
    plt.xlabel('Numerosity', fontsize=12)
    plt.ylabel('Number of Selectively Tuned Kernels', fontsize=12)
    plt.title(f'Selectively Tuned Kernels to Each Numerosity (Layer {layer}, threshold {selectivity_threshold})', fontsize=14)
    plt.xticks(range(1, 31, 2), fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plot_file_path = os.path.join(output_dir, f'{model_name}_Layer_{layer}_Numerosity_Selectivity_Histogram.png')
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()

    print(f"Plot saved to {plot_file_path}")
    print(f"Results saved to {results_file_path}")
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
        
