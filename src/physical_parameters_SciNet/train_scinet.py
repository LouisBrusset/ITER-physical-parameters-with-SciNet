
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import gc
from tqdm import tqdm
from pathlib import Path

from magnetics_diagnostic_analysis.project_scinet.setting_scinet import config
from magnetics_diagnostic_analysis.project_scinet.utils.build_dataset import PendulumDataset
from magnetics_diagnostic_analysis.project_scinet.model.scinet import PendulumNet
from magnetics_diagnostic_analysis.ml_tools.metrics import scinet_loss
from magnetics_diagnostic_analysis.ml_tools.train_callbacks import EarlyStopping, GradientClipping, LRScheduling


# Training loop

def train_scinet(
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        num_epochs: int = 150, 
        kld_beta: float = 0.001, 
        early_stopper: EarlyStopping = None, 
        gradient_clipper: GradientClipping = None, 
        lr_scheduler: LRScheduling = None,
        device: torch.device = torch.device('cpu')
        ) -> None:
    
    torch.cuda.empty_cache()
    model.to(device)
    print("------training on {}-------\n".format(device))
    history = {'train_loss': [], 'valid_loss': []}
    print(f"{'Epoch':<20} ||| {'Train Loss':<15} ||| {'KLD Loss':<12} {'Recon Loss':<12} ||||||| {'Valid Loss':<15}")

    # Training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        kld_loss, recon_loss = 0.0, 0.0
        for observations, questions, a_corr, _ in tqdm(train_loader, desc="Training", leave=False):
            observations = observations.to(device)
            questions = questions.to(device).unsqueeze(-1)
            a_corr = a_corr.to(device).unsqueeze(-1)

            optimizer.zero_grad()
            possible_answer, mean, logvar = model(observations, questions)
            loss, l_kld, l_recon = scinet_loss(possible_answer, a_corr, mean, logvar, beta=kld_beta)
            loss.backward()
            if gradient_clipper is not None:
                gradient_clipper.on_backward_end(model)
            optimizer.step()

            train_loss += loss.item() * observations.size(0)
            kld_loss += l_kld.item() * observations.size(0)
            recon_loss += l_recon.item() * observations.size(0)
        train_loss /= len(train_loader.dataset)
        kld_loss /= len(train_loader.dataset)
        recon_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Evaluation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for observations, questions, a_corr, _ in tqdm(valid_loader, desc="Validation", leave=False):
                observations = observations.to(device)
                questions = questions.to(device).unsqueeze(-1)
                a_corr = a_corr.to(device).unsqueeze(-1)

                possible_answer, mean, logvar = model(observations, questions)
                loss = scinet_loss(possible_answer, a_corr, mean, logvar, beta=kld_beta)[0]
                valid_loss += loss.item() * observations.size(0)
        
        valid_loss /= len(valid_loader.dataset)
        history['valid_loss'].append(valid_loss)

        print(f"{f'{epoch+1}/{num_epochs}':<20}  |  {train_loss:<15.6f}  |  {kld_loss:<12.6f} {recon_loss:<12.6f}    |    {valid_loss:<15.6f}")

        if early_stopper is not None:
            if early_stopper.check_stop(valid_loss, model):
                print(f"Early stopping at epoch {epoch + 1} with loss {valid_loss:.4f}")
                print(f"Restoring best weights for model.")
                early_stopper.restore_best_weights(model)
                break

        if lr_scheduler is not None:
            lr_scheduler.step(valid_loss)

        path = config.DIR_PARAMS_CHECKPOINTS / "pendulum_scinet_checkpointed.pth"
        torch.save(model.state_dict(), path)

        torch.cuda.empty_cache()
        gc.collect()
    
    return history


def plot_history(history_train: list, history_valid: list) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_train, 'b-', linewidth=2, label='Train Loss')
    plt.plot(history_valid, 'r-', linewidth=2, label='Valid Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = config.DIR_FIGURES / "training_validation_loss.png"
    plt.savefig(path)
    return None







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
    