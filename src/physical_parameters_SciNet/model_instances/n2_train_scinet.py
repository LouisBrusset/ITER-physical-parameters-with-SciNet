import torch

from physical_parameters_SciNet.model_instances.n2_setting_mast_constant_time import config
from physical_parameters_SciNet.utils.build_dataset import mast_dataset
from torch.utils.data import DataLoader
from physical_parameters_SciNet.model.scinet import PendulumNet
from physical_parameters_SciNet.ml_tools.train_callbacks import EarlyStopping, GradientClipping, LRScheduling
from physical_parameters_SciNet.utils.train_scinet import train_scinet, plot_history




if __name__ == "__main__":

    device = config.DEVICE
    print(f"----------- Using device: {device} -----------")

    # Load datasets
    path_train = config.DIR_PREPROCESSED_DATA / "mast_scinet_train_dataset.pt"
    path_valid = config.DIR_PREPROCESSED_DATA / "mast_scinet_valid_dataset.pt"
    dataset_train = torch.load(path_train, weights_only=False)
    dataset_valid = torch.load(path_valid, weights_only=False)

    # Create DataLoader
    train_loader = DataLoader(dataset_train, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=config.BATCH_SIZE_EVAL, shuffle=True)
    print(f"\nDataLoaders created with batch size {config.BATCH_SIZE_TRAIN} for training and {config.BATCH_SIZE_EVAL} for evaluation.\n")

    # Initialize model, optimizer, and callbacks
    mast_net = PendulumNet(
        input_size=config.M_INPUT_SIZE,
        enc_hidden_sizes=config.M_ENC_HIDDEN_SIZES,
        latent_size=config.M_LATENT_SIZE,
        question_size=config.M_QUESTION_SIZE,
        dec_hidden_sizes=config.M_DEC_HIDDEN_SIZES,
        output_size=config.M_OUTPUT_SIZE
    )
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
            num_epochs=config.NUM_EPOCHS, 
            kld_beta=config.KLD_BETA, 
            early_stopper=early_stopper, 
            gradient_clipper=gradient_clipper, 
            lr_scheduler=lr_scheduler,
            device=device
        )

        print("\nTraining completed.")

        path = config.DIR_MODEL_PARAMS / "mast_scinet_final.pth"
        torch.save(mast_net.state_dict(), path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        path = config.DIR_PARAMS_CHECKPOINTS / "mast_scinet_interrupted.pth"
        torch.save(mast_net.state_dict(), path)
        print("Model state saved.")


    plot_history(history['train_loss'], history['valid_loss'])
    