import torch
import numpy as np
import matplotlib.pyplot as plt

# from physical_parameters_SciNet.model_instances.n2_setting_mast_constant_time import config
from physical_parameters_SciNet.model_instances.n3_setting_mast_mag_prob import config

from physical_parameters_SciNet.utils.build_dataset import mast_dataset
from torch.utils.data import DataLoader
from physical_parameters_SciNet.model.scinet import PendulumNet


def load_trained_model(model_path: str, device: torch.device = torch.device('cpu')) -> PendulumNet:
    model = PendulumNet(
        input_size=config.M_INPUT_SIZE,
        enc_hidden_sizes=config.M_ENC_HIDDEN_SIZES,
        latent_size=config.M_LATENT_SIZE,
        question_size=config.M_QUESTION_SIZE,
        dec_hidden_sizes=config.M_DEC_HIDDEN_SIZES,
        output_size=config.M_OUTPUT_SIZE
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def full_inference(model: PendulumNet, data_loader: DataLoader, normalization_stats: dict, device: torch.device, save: bool = False) -> tuple:
    obs_factor, que_factor, ans_factor = normalization_stats['obs_mean_max'], normalization_stats['que_mean_max'], normalization_stats['ans_mean_max']
    all_reconstructions, all_means, all_logvars = [], [], []
    all_observations, all_questions, all_answers = [], [], []
    
    model.eval()
    with torch.no_grad():
        for observations, questions, answers in data_loader:
            observations = observations.to(device) / obs_factor
            questions = questions.to(device) / que_factor
            possible_answer, mean, logvar = model(observations, questions)
            all_reconstructions.append(possible_answer.cpu().numpy() * ans_factor)
            all_means.append(mean.cpu().numpy())
            all_logvars.append(logvar.cpu().numpy())
            all_observations.append(observations.cpu().numpy() * obs_factor)
            all_questions.append(questions.cpu().numpy() * que_factor)
            all_answers.append(answers.cpu().numpy())
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    all_means = np.concatenate(all_means, axis=0)
    all_logvars = np.concatenate(all_logvars, axis=0)
    all_observations = np.concatenate(all_observations, axis=0)
    all_questions = np.concatenate(all_questions, axis=0)
    all_answers = np.concatenate(all_answers, axis=0)
    if save:
        path = config.DIR_PROCESSED_DATA / f"{config.MODEL_NAME}_test_reconstructions.npy"
        np.save(path, all_reconstructions)
        path = config.DIR_PROCESSED_DATA / f"{config.MODEL_NAME}_test_latent.npy"
        np.save(path, all_means)
    return all_reconstructions, all_means, all_logvars, all_observations, all_questions, all_answers



def plot_reconstructions_answers_observations(observations: np.ndarray, reconstructions: np.ndarray, questions: np.ndarray, answers: np.ndarray, sample_idx: int):
    time = np.linspace(config.MIN_TIME, config.MAX_TIME, config.M_INPUT_SIZE)
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, observations[sample_idx], label='b_field_probe_ccbv', color='black', alpha=0.3)
    plt.title(f'Observations: magnetic probe (Sample Index: {sample_idx})')
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic field intensity (T)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(time, answers[sample_idx], label='Plasma current', color='blue', linestyle='--')
    #plt.plot(time, reconstructions[sample_idx], label='Reconstruction', color='orange')
    plt.title(f'Reconstruction vs Answer: plasma current (Sample Index: {sample_idx})')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(time, questions[sample_idx], label='Schedule', color='green')
    #plt.plot(time, reconstructions[sample_idx], label='Reconstruction', color='orange')
    plt.title(f'Forcing Amplitude: plasma schedule (Sample Index: {sample_idx})')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = config.DIR_FIGURES_CHANNEL / f"reconstruction_vs_answer_sample_{sample_idx}_without_recon.png"
    plt.savefig(path)
    #plt.show()
    return None



def plot_latent_variables(means: np.ndarray, n_max_cols: int = 5):
    latent_dim = means.shape[1]

    n_cols = min(latent_dim, n_max_cols)
    n_rows = (latent_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i in range(latent_dim):
        axes[i].hist(means[:, i], bins=30, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Latent Variable {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    for i in range(latent_dim, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    path = config.DIR_FIGURES_CHANNEL / f"latent_variables_distribution.png"
    plt.savefig(path)
    #plt.show()
    return None








if __name__ == "__main__":
    device = config.DEVICE

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

    # Plot n random reconstruction
    n = 6
    for _ in range(n):
        sample_idx = np.random.choice(config.TEST_SIZE)
        plot_reconstructions_answers_observations(
            observations=all_observations, 
            reconstructions=reconstructions, 
            questions=all_questions,
            answers=all_answers,
            sample_idx=sample_idx
        )




