import os
import pandas as pd
from model import get_model_and_optimizer
from dataset_setup import get_dataloaders
import torch
from analysis_utils import (
    get_softmax_values_and_ground_truth, 
    summarize_softmax,
    compute_accuracy,
    load_best_model,
    calculate_entropy,  
    layer_decoding_analysis,
    compute_tuning_curves,
    compute_selectivity_index_and_save_csv
)
from plot_utils import (
    plot_train_val_losses,
    plot_train_val_accuracies,
    plot_test_accuracy_by_numerosity,
    plot_test_accuracy_by_object,
    plot_combined_numerosity_accuracies_by_object,
    plot_softmax_distributions,
    plot_entropy_vs_numerosity, 
    plot_decoding_accuracy_vs_layer,
    plot_softmax_confidence_vs_layer,
    plot_layerwise_entropy_vs_numerosity,
    plot_confusion_matrices_per_layer,
    plot_softmax_values_per_layer
)

def main():
    
    conditions = ['', '_samesize', '_ood']
    N = 10
    
    for i in range(N):
        
        for condition in conditions:
        
            print(f'Assessing Instance {i+1}/{N}')
            
            
            model_type = 'MT'
            model_name = f'{model_type}_{i}'
            
            
            # Load the best model based on validation losses
            model, _, _, optimizer = get_model_and_optimizer()
            model, best_epoch = load_best_model(history_path = f'{model_name}_checkpoints/training_history.csv', model = model, model_name = model_name, optimizer = optimizer, checkpoints_dir = f'{model_name}_checkpoints')
        
            if condition == '':
                # Change depending on dataset
                csv_file = '/Users/tammo/Desktop/Project/AltDatasetLarger/dataset.csv'
                img_dir = '/Users/tammo/Desktop/Project/AltDatasetLarger'
            
            elif condition == '_samesize':
                # Change depending on dataset
                csv_file = '/Users/tammo/Desktop/Project/AltDatasetLarger_SameSize/dataset.csv'
                img_dir = '/Users/tammo/Desktop/Project/AltDatasetLarger_SameSize'
                
            elif condition == '_ood':
                # Change depending on dataset
                csv_file = '/Users/tammo/Desktop/Project/AltDatasetLarger/dataset.csv'
                img_dir = '/Users/tammo/Desktop/Project/AltDatasetLarger'
                
            else:
                raise Exception(f'{condition} is not a valid condition!')
        
            # Get DataLoader for the test set
            train_loader, _, test_loader = get_dataloaders(csv_file, img_dir, batch_size=32)
        
            # Get softmax results and ground truth from the test set
            num_softmax, object_softmax = get_softmax_values_and_ground_truth(model, test_loader)
        
            # Summarize softmax distributions by numerosity and object type
            summarized_data, spread_data = summarize_softmax(num_softmax)
        
            # Set up directory to save plots
            plots_save_dir = f'{model_name}_plots'
            csv_save_dir = f'{model_name}_csv'
            os.makedirs(plots_save_dir, exist_ok=True)
            os.makedirs(csv_save_dir, exist_ok=True)
        
            # Load training history for plotting
      #      history_df = pd.read_csv(f'{model_name}_checkpoints/training_history.csv')
        
            # Plot and save train/validation losses and accuracies
     #       plot_train_val_losses(history_df, save_dir)
     #       plot_train_val_accuracies(history_df, save_dir)
        
            # Compute test accuracy by numerosity and object type
            numerosity_accuracy, object_accuracy,_,_ = compute_accuracy(model_name = model_name, condition = condition, num_softmax = num_softmax, object_softmax = object_softmax, csv_save_dir=csv_save_dir)
        
            # Convert accuracy dictionaries to lists of tuples for plotting
      #      test_results_by_numerosity = [(n, numerosity_accuracy[n]) for n in sorted(numerosity_accuracy.keys())]
      #      test_results_by_object = [(o, object_accuracy[o]) for o in sorted(object_accuracy.keys())]
        
            # Plot and save test accuracies
       #     plot_test_accuracy_by_numerosity(test_results_by_numerosity, save_dir)
       #     plot_test_accuracy_by_object(test_results_by_object, save_dir)
        
            # Plot and save combined numerosity accuracies by object type
            plot_combined_numerosity_accuracies_by_object(condition=condition, test_results=num_softmax, plots_save_dir=plots_save_dir)
        
            # Plot and save softmax distributions
            plot_softmax_distributions(condition=condition, summarized_data=summarized_data, plots_save_dir=plots_save_dir)
        
            # Plot and save entropy vs numerosity
            plot_entropy_vs_numerosity(condition=condition, summarized_data=summarized_data, plots_save_dir=plots_save_dir)  # New entropy plot
        
            # Specify the layers to analyze
            layers_to_analyze = [model.conv1, model.conv2, model.conv3, model.conv4, model.conv5, model.conv6]
        
            # Perform the decoding analysis
            accuracies, softmax_values, entropy_values, predictions_per_layer, true_labels_per_layer = layer_decoding_analysis(
                model=model, train_loader=train_loader, test_loader=test_loader, layers=layers_to_analyze, num_classes=16
            )
        
            # Plot decoding accuracy vs. layer
            plot_decoding_accuracy_vs_layer(condition=condition, accuracies=accuracies, layers=layers_to_analyze, plots_save_dir=plots_save_dir)
        
            # Plot softmax confidence vs. layer
         #   plot_softmax_confidence_vs_layer(condition=condition, softmax_values=softmax_values, layers=layers_to_analyze, plots_save_dir=plots_save_dir)
        
            # Plot entropy vs. numerosity for each layer
         #   plot_layerwise_entropy_vs_numerosity(condition=condition, entropy_values=entropy_values, num_classes=16, layers=layers_to_analyze, plots_save_dir=plots_save_dir)
        
            # Plot confusion matrices for each layer
            plot_confusion_matrices_per_layer(condition=condition, predictions_per_layer=predictions_per_layer, true_labels_per_layer=true_labels_per_layer, layers=layers_to_analyze, num_classes=16, csv_save_dir=csv_save_dir, plots_save_dir=plots_save_dir)
        
            # Plot softmax distribution for each layer
            plot_softmax_values_per_layer(condition=condition, softmax_values=softmax_values, layers=layers_to_analyze, num_numerosities=16, plots_save_dir=plots_save_dir)
        
            # Perform tuning curve analysis for each layer
            
            
            for layer_idx, layer in enumerate(layers_to_analyze):
                compute_tuning_curves(model, test_loader, layer, num_numerosities=16, layer_idx=layer_idx, plots_save_dir=plots_save_dir, csv_save_dir=csv_save_dir)
        
         #   print(f"Plots saved in {save_dir}.")
        
            # Example usage of the function
            selectivity_indices = compute_selectivity_index_and_save_csv(model, test_loader, layers_to_analyze, num_numerosities=16, save_dir=f'{model_name}_selectivity_indices')



if __name__ == "__main__":
    main()