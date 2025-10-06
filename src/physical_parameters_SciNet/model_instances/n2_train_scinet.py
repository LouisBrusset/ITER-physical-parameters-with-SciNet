import torch

from pathlib import Path

from physical_parameters_SciNet.model_instances.n2_setting_mast_constant_time import config
from physical_parameters_SciNet.model.scinet import PendulumNet
from physical_parameters_SciNet.ml_tools.train_callbacks import EarlyStopping, GradientClipping, LRScheduling




if __name__ == "__main__":

    device = config.DEVICE
    print(f"----------- Using device: {device} -----------")

    # Load datasets
    path_train = config.DIR_SYNTHETIC_DATA / "pendulum_scinet_train.pt"
    path_valid = config.DIR_SYNTHETIC_DATA / "pendulum_scinet_valid.pt"
    train_dataset = torch.load(path_train)
    valid_dataset = torch.load(path_valid)

    # Create dataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE_VALID, shuffle=False)

    # Initialize model, optimizer, and callbacks
    pendulum_net = PendulumNet(
        input_size=config.M_INPUT_SIZE,
        enc_hidden_sizes=config.M_ENC_HIDDEN_SIZES,
        latent_size=config.M_LATENT_SIZE,
        question_size=config.M_QUESTION_SIZE,
        dec_hidden_sizes=config.M_DEC_HIDDEN_SIZES,
        output_size=config.M_OUTPUT_SIZE
    )
    optimizer = torch.optim.Adam(pendulum_net.parameters(), lr=config.FIRST_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    early_stopper = EarlyStopping(patience=config.ES_PATIENCE, min_delta=config.ES_MIN_DELTA)
    gradient_clipper = GradientClipping(max_norm=config.GC_MAX_NORM)
    lr_scheduler = LRScheduling(optimizer, factor=config.LRS_FACTOR, patience=config.LRS_PATIENCE, min_lr=config.LRS_MIN_LR, min_delta=config.LRS_MIN_DELTA)


    # Train the model
    try:
        history = train_scinet(
            train_loader, 
            valid_loader, 
            pendulum_net, 
            optimizer, 
            num_epochs=config.NUM_EPOCHS, 
            kld_beta=config.KLD_BETA, 
            early_stopper=early_stopper, 
            gradient_clipper=gradient_clipper, 
            lr_scheduler=lr_scheduler,
            device=device
        )

        print("\nTraining completed.")

        path = config.DIR_MODEL_PARAMS / "pendulum_scinet_final.pth"
        torch.save(pendulum_net.state_dict(), path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        path = config.DIR_PARAMS_CHECKPOINTS / "pendulum_scinet_interrupted.pth"
        torch.save(pendulum_net.state_dict(), path)
        print("Model state saved.")


    plot_history(history['train_loss'], history['valid_loss'])
    