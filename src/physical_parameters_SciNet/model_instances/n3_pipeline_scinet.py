import torch
import numpy as np

import gc

from physical_parameters_SciNet.model_instances.n3_setting_mast_mag_prob import config

from physical_parameters_SciNet.utils.build_dataset import build_dataset

from physical_parameters_SciNet.utils.build_dataset import mast_dataset
from torch.utils.data import DataLoader
from physical_parameters_SciNet.model.scinet import PendulumNet
from physical_parameters_SciNet.ml_tools.train_callbacks import EarlyStopping, GradientClipping, LRScheduling
from physical_parameters_SciNet.utils.train_scinet import train_scinet, plot_history

from physical_parameters_SciNet.utils.mast_scinet_evaluation import load_trained_model, full_inference, plot_reconstructions_answers_observations, plot_latent_variables



if __name__ == "__main__":
    device = config.DEVICE
    print(f"\n----------- Using device: {device} -----------")
    """
    build_dataset()

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

    # Initialize model, optimizer, and callbacks
    mast_net = PendulumNet(
        input_size=config.M_INPUT_SIZE,
        enc_hidden_sizes=config.M_ENC_HIDDEN_SIZES,
        latent_size=config.M_LATENT_SIZE,
        question_size=config.M_QUESTION_SIZE,
        dec_hidden_sizes=config.M_DEC_HIDDEN_SIZES,
        output_size=config.M_OUTPUT_SIZE
    )
    print(f"Model initialized with {sum(p.numel() for p in mast_net.parameters() if p.requires_grad)} trainable parameters.\n")
    print(mast_net, "\n")

    optimizer = torch.optim.Adam(mast_net.parameters(), lr=config.FIRST_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    early_stopper = EarlyStopping(patience=config.ES_PATIENCE, min_delta=config.ES_MIN_DELTA)
    gradient_clipper = GradientClipping(max_norm=config.GC_MAX_NORM)
    lr_scheduler = LRScheduling(optimizer, factor=config.LRS_FACTOR, patience=config.LRS_PATIENCE, min_lr=config.LRS_MIN_LR, min_delta=config.LRS_MIN_DELTA)


    # Train the model
    try:
        history = train_scinet(
            train_loader, 
            valid_loader, 
            mast_net, 
            optimizer,
            normalization_stats,
            num_epochs=config.NUM_EPOCHS, 
            kld_beta=config.KLD_BETA, 
            early_stopper=early_stopper, 
            gradient_clipper=gradient_clipper, 
            lr_scheduler=lr_scheduler,
            device=device
        )

        print("\nTraining completed.")

        path = config.DIR_PARAMS_CHANNEL / f"{config.MODEL_NAME}_final.pth"
        torch.save(mast_net.state_dict(), path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        path = config.DIR_PARAMS_CHECKPOINTS / f"{config.MODEL_NAME}_interrupted.pth"
        torch.save(mast_net.state_dict(), path)
        print("Model state saved.")


    plot_history(history['train_loss'], history['valid_loss'])
    """


    # Load model
    model_path = config.DIR_PARAMS_CHANNEL / f"{config.BEST_MODEL_NAME}.pth"
    pendulum_net = load_trained_model(model_path, device)

    # Load test dataset
    path = config.DIR_PREPROCESSED_DATA / f"{config.MODEL_NAME}_test_dataset.pt"
    test_dataset = torch.load(path, weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE_EVAL, shuffle=True)

    # Load normalization stats
    path = config.DIR_OTHERS_DATA_CHANNEL / f"{config.MODEL_NAME}_normalization_stats.pt"
    normalization_stats = torch.load(path)

    # Full inference on test set
    reconstructions, means, logvars, all_observations, all_questions, all_answers = full_inference(pendulum_net, test_loader, normalization_stats, device)

    # Plot latent variables distributions
    plot_latent_variables(means, n_max_cols=5)

    # Plot a random reconstruction
    sample_idx = np.random.choice(config.TEST_SIZE)
    plot_reconstructions_answers_observations(
        observations=all_observations, 
        reconstructions=reconstructions,
        answers=all_answers,
        questions=all_questions, 
        sample_idx=sample_idx
    )