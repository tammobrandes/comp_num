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
    plot_softmax_values_per_layer,
    plot_tuning_curves
)

def main():
    
    conditions = ['','_samesize']#['', '_samesize', '_ood']
    N = 1#10
    print(f'Assessing Instance {1}/{N}')
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
                zip_file = '/home/jgroot1/ArithmeticProj/datasets/AltDatasetLarger.zip'
                subfolder_name = 'AltDatasetLarger'  # Subfolder inside the ZIP
                csv_filename = 'dataset.csv'  # Name of the CSV inside the subfolder
            
            elif condition == '_samesize':
                # Change depending on dataset
                zip_file = '/home/jgroot1/ArithmeticProj/datasets/AltDatasetLarger_SameSize.zip'
                subfolder_name = 'AltDatasetLarger_SameSize'  # Subfolder inside the ZIP
                csv_filename = 'dataset.csv'  # Name of the CSV inside the subfolder
                
            elif condition == '_ood':
                # Change depending on dataset
                zip_file = '/home/jgroot1/ArithmeticProj/datasets/AltDatasetOOD.zip'
                subfolder_name = 'AltDatasetOOD'  # Subfolder inside the ZIP
                csv_filename = 'dataset.csv'  # Name of the CSV inside the subfolder
                
            else:
                raise Exception(f'{condition} is not a valid condition!')
        
            # Get DataLoader for the test set
            train_loader, _, test_loader, train_val_loader = get_dataloaders(
                zip_file=zip_file,
                subfolder_name=subfolder_name,
                csv_filename=csv_filename,
                batch_size=32,
                seed = i # Important to get random splits
            )
        
            # Get softmax results and ground truth from the test set
            num_softmax, object_softmax = get_softmax_values_and_ground_truth(model, test_loader)
        
            # Summarize softmax distributions by numerosity and object type
            summarized_data, spread_data = summarize_softmax(num_softmax)
        
            # Set up directory to save plots
            plots_save_dir = f'{model_name}_plots'
            csv_save_dir = f'{model_name}_csv'
            os.makedirs(plots_save_dir, exist_ok=True)
            os.makedirs(csv_save_dir, exist_ok=True)
        
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
            layer_decoding_analysis(
                    model=model,
                    train_val_loader=train_val_loader, 
                    test_loader=test_loader,
                    layers=layers_to_analyze, 
                    condition=condition,
                    model_name=model_name,
                    csv_save_dir=csv_save_dir
                )
        
        
            # Perform tuning curve analysis for each layer
            for layer_idx, layer in enumerate(layers_to_analyze):
                compute_tuning_curves(
                    model=model,
                    dataloader=test_loader,
                    layer=layer,
                    num_numerosities=16,
                    layer_idx=layer_idx,
                    plots_dir=os.path.join(plots_save_dir, f'tuning_curves/layer_{layer_idx}'),
                    csv_dir=os.path.join(csv_save_dir, 'tuning_curves')
                )

            # Compute selectivity index
            compute_selectivity_index_and_save_csv(
                model=model,
                model_name=model_name,
                dataloader=test_loader,
                condition=condition,
                layers=layers_to_analyze,
                num_numerosities=16,
                save_dir=os.path.join(csv_save_dir, 'selectivity_index')
            )
            
            #sel_thr1 = 0.3
            #sel_thr2 = 0.6
            #compute_acc_sel_corr(sel_thr1, sel_thr2, numerosity_accuracy,model_name, condition, csv_dir,os.path.join(plots_save_dir, 'acc_sel_corr'))

if __name__ == "__main__":
    main()
