import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gc

from physical_parameters_SciNet.model_instances.n3_setting_mast_mag_prob import config

from physical_parameters_SciNet.utils.build_dataset import mast_dataset
from torch.utils.data import DataLoader
from physical_parameters_SciNet.model.scinet import PendulumNet
from physical_parameters_SciNet.ml_tools.train_callbacks import EarlyStopping, GradientClipping, LRScheduling
from physical_parameters_SciNet.utils.train_scinet import train_scinet, plot_history

from physical_parameters_SciNet.utils.mast_scinet_evaluation import load_trained_model, full_inference, plot_reconstructions_answers_observations, plot_latent_variables


def plot_latent_sizes(results_df: pd.DataFrame) -> None:
    # Plot final KLD and Reconstruction losses vs Latent Space Size
    # On plot for KLD loss first, another for Recon loss, saved on two figures

    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Latent Size'], results_df['Final KLD Loss'], marker='o', label='Final KLD Loss', color='blue')
    plt.title('Final KLD Loss vs Latent Space Size')
    plt.xlabel('Latent Space Size')
    plt.ylabel('Final KLD Loss')
    plt.xticks(results_df['Latent Size'].astype(int))
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = config.DIR_FIGURES_CHANNEL / f"final_kld_loss_vs_latent_size.png"
    plt.savefig(path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(results_df['Latent Size'], results_df['Final Recon Loss'], marker='o', label='Final Reconstruction Loss', color='orange')
    plt.title('Final Reconstruction Loss vs Latent Space Size')
    plt.xlabel('Latent Space Size')
    plt.ylabel('Final Reconstruction Loss')
    # convert to int for xticks
    plt.xticks(results_df['Latent Size'].astype(int))
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = config.DIR_FIGURES_CHANNEL / f"final_recon_loss_vs_latent_size.png"
    plt.savefig(path)
    plt.close()
    return None
   



if __name__ == "__main__":
    device = config.DEVICE
    print(f"\n----------- Using device: {device} -----------")
    
    
    # Clear GPU cache before starting training
    torch.cuda.empty_cache()
    gc.collect()

    
    # Load datasets
    path_train = config.DIR_PREPROCESSED_DATA / f"{config.MODEL_NAME}_train_dataset.pt"
    path_valid = config.DIR_PREPROCESSED_DATA / f"{config.MODEL_NAME}_valid_dataset.pt"
    dataset_train = torch.load(path_train, weights_only=False)
    print(f"\nTraining dataset loaded with {len(dataset_train)} samples.")
    dataset_valid = torch.load(path_valid, weights_only=False)
    print(f"Validation dataset loaded with {len(dataset_valid)} samples.\n")

    # Load normalization stats
    path_stats = config.DIR_OTHERS_DATA_CHANNEL / f"{config.MODEL_NAME}_normalization_stats.pt"
    normalization_stats = torch.load(path_stats)
    print(f"Normalization stats loaded from {path_stats}.\n")
    print(f"Normalization stats: {normalization_stats}\n")

    # Create DataLoader
    train_loader = DataLoader(dataset_train, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
    print(f"Training DataLoader created with batch size {config.BATCH_SIZE_TRAIN}.")
    valid_loader = DataLoader(dataset_valid, batch_size=config.BATCH_SIZE_EVAL, shuffle=True)
    print(f"Validation DataLoader created with batch size {config.BATCH_SIZE_EVAL}.\n")


    # Create a pandas DataFrame to store final losses for each latent size
    results_df = pd.DataFrame(columns=['Latent Size', 'Final KLD Loss', 'Final Recon Loss'])

    # Train the model for different latent sizes
    for latent_size in [i for i in range(1, 16)]:
        print(f"\n--- Training with latent size: {latent_size} ---\n")
    
        # Initialize model, optimizer, and callbacks
        mast_net = PendulumNet(
            input_size=config.M_INPUT_SIZE,
            enc_hidden_sizes=config.M_ENC_HIDDEN_SIZES,
            latent_size=latent_size,
            question_size=config.M_QUESTION_SIZE,
            dec_hidden_sizes=config.M_DEC_HIDDEN_SIZES,
            output_size=config.M_OUTPUT_SIZE
        )
        optimizer = torch.optim.Adam(mast_net.parameters(), lr=config.FIRST_LEARNING_RATE*1.3, weight_decay=config.WEIGHT_DECAY)
        early_stopper = EarlyStopping(patience=config.ES_PATIENCE, min_delta=config.ES_MIN_DELTA)
        gradient_clipper = GradientClipping(max_norm=config.GC_MAX_NORM)
        lr_scheduler = LRScheduling(optimizer, factor=0.5, patience=3, min_lr=config.LRS_MIN_LR, min_delta=config.LRS_MIN_DELTA)
        # Train the model
        history = train_scinet(
            train_loader, 
            valid_loader, 
            mast_net, 
            optimizer,
            normalization_stats,
            num_epochs=40, #config.NUM_EPOCHS, 
            kld_beta=config.KLD_BETA, 
            early_stopper=early_stopper, 
            gradient_clipper=gradient_clipper, 
            lr_scheduler=lr_scheduler,
            device=device
        )
        print("\nTraining completed.")
        #path = config.DIR_PARAMS_CHANNEL / f"{config.MODEL_NAME}_final.pth"
        #torch.save(mast_net.state_dict(), path)

        kld_loss_final = history['kld_loss'][-1]
        recon_loss_final = history['recon_loss'][-1]

        # Append results to DataFrame
        new_data = pd.DataFrame({'Latent Size': latent_size,
                                 'Final KLD Loss': kld_loss_final,
                                 'Final Recon Loss': recon_loss_final}, index=[0])
        if latent_size == 1:
            results_df = results_df._append(new_data, ignore_index=True)
        else:   
            results_df = pd.concat([results_df, new_data], ignore_index=True)
        
        del mast_net, optimizer, early_stopper, gradient_clipper, lr_scheduler
        torch.cuda.empty_cache()
        gc.collect()

    
    # Save results to CSV
    results_csv_path = config.DIR_OTHERS_DATA_CHANNEL / f"{config.MODEL_NAME}_latent_sizes-results.csv"
    results_df.to_csv(results_csv_path, index=False)
    #results_df = pd.read_csv(results_csv_path)
    print(f"Results loaded from {results_csv_path}.")

    # Plot results
    plot_latent_sizes(results_df)