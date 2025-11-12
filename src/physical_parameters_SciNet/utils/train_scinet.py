import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import gc
from tqdm import tqdm

# from physical_parameters_SciNet.model_instances.n2_setting_mast_constant_time import config
from physical_parameters_SciNet.model_instances.n3_setting_mast_mag_prob import config

from physical_parameters_SciNet.ml_tools.metrics import scinet_loss_forced_pendulum as scinet_loss
from physical_parameters_SciNet.ml_tools.train_callbacks import EarlyStopping, GradientClipping, LRScheduling


# Training loop

def train_scinet(
        train_loader: DataLoader, 
        valid_loader: DataLoader,
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        normalization_stats: dict,
        num_epochs: int = 150, 
        kld_beta: float = 0.001, 
        early_stopper: EarlyStopping = None, 
        gradient_clipper: GradientClipping = None, 
        lr_scheduler: LRScheduling = None,
        device: torch.device = torch.device('cpu')
        ) -> None:
    
    torch.cuda.empty_cache()
    model.to(device)
    obs_factor = normalization_stats['obs_mean_max']
    que_factor = normalization_stats['que_mean_max']
    ans_factor = normalization_stats['ans_mean_max']

    print("------training on {}-------\n".format(device))
    history = {'train_loss': [], 'valid_loss': [], 'kld_loss': [], 'recon_loss': []}
    print(f"{'Epoch':<20} ||| {'Train Loss':<15} ||| {'KLD Loss':<12} {'Recon Loss':<12} ||||||| {'Valid Loss':<15}")
    # Training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        kld_loss, recon_loss = 0.0, 0.0
        for observations, questions, a_corr in tqdm(train_loader, desc="Training", leave=False):
            observations = observations.to(device) / obs_factor
            questions = questions.to(device) / que_factor
            a_corr = a_corr.to(device) / ans_factor

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
        history['kld_loss'].append(kld_loss)
        history['recon_loss'].append(recon_loss)

        # Evaluation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for observations, questions, a_corr in tqdm(valid_loader, desc="Validation", leave=False):
                observations = observations.to(device) / obs_factor
                questions = questions.to(device) / que_factor
                a_corr = a_corr.to(device) / ans_factor

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

        path = config.DIR_PARAMS_CHECKPOINTS / f"{config.MODEL_NAME}_checkpointed.pth"
        torch.save(model.state_dict(), path)

        del observations, questions, a_corr, possible_answer, mean, logvar, loss, l_kld, l_recon
        gc.collect()
        torch.cuda.empty_cache()
    
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
    path = config.DIR_FIGURES_CHANNEL / f"train_valid_loss_{config.MODEL_NAME}.png"
    plt.savefig(path)
    return None