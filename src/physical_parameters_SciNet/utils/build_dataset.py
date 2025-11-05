import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import xarray as xr

# from physical_parameters_SciNet.model_instances.n2_setting_mast_constant_time import config
from physical_parameters_SciNet.model_instances.n3_setting_mast_mag_prob import config


class mast_dataset(Dataset):
    def __init__(self, 
                 dataset: xr.Dataset, 
                 chosen_signals: str = "ip", 
                 compute_stats: bool = False
                 ) -> None:
        self.dataset = dataset
        if "::" in chosen_signals:
            self.chosen_var, self.chosen_channel = chosen_signals.split("::")
        else:
            self.chosen_var, self.chosen_channel = chosen_signals, None
        self.shot_ids = dataset.shot_id.values


    def __len__(self):
        return len(self.shot_ids)

    def __getitem__(self, idx):
        shot_id = self.shot_ids[idx]
        answers = self.dataset["ip_ref"].sel(shot_id=shot_id).values        # shape: (time,)
        questions = self.dataset["i_plasma"].sel(shot_id=shot_id).values    # shape: (time,)
        if self.chosen_channel is not None:     # Multi-channel variable
            if "b_field" in self.chosen_var:
                observations = self.dataset[self.chosen_var].sel(shot_id=shot_id, **{f"{self.chosen_var[:-6]}_channel": self.chosen_channel}).values
            elif "flux" in self.chosen_var:
                observations = self.dataset[self.chosen_var].sel(shot_id=shot_id, **{f"{self.chosen_var[:-5]}_channel": self.chosen_channel}).values
        else:                                   # Single-channel variable
            observations = self.dataset[self.chosen_var].sel(shot_id=shot_id).values
        
        subsampling_factor = config.SUBSAMPLING_FACTOR
        self.observations_tensor = torch.tensor(observations[::subsampling_factor], dtype=torch.float32)
        self.questions_tensor = torch.tensor(questions[::subsampling_factor], dtype=torch.float32)
        self.answers_tensor = torch.tensor(answers[::subsampling_factor], dtype=torch.float32)

        return self.observations_tensor, self.questions_tensor, self.answers_tensor

    def compute_normalization_stats(self) -> None:
        obs_max = torch.max(self.observations_tensor, dim=1).values
        self.obs_mean_max = torch.mean(obs_max).item()
        que_max = torch.max(self.questions_tensor, dim=1).values
        self.que_mean_max = torch.mean(que_max).item()
        ans_max = torch.max(self.answers_tensor, dim=1).values
        self.ans_mean_max = torch.mean(ans_max).item()
        normalization_stats = {
            "metadata": f"Normalization stats computed from training dataset for model {config.MODEL_NAME}.\nNormalization performed with two bounds: min=0 and max=mean of max values of timeseries over samples.",
            "obs_mean_max": self.obs_mean_max,
            "que_mean_max": self.que_mean_max,
            "ans_mean_max": self.ans_mean_max
        }
        path = config.DIR_OTHERS_DATA_CHANNEL / f"{config.MODEL_NAME}_normalization_stats.pt"
        torch.save(normalization_stats, path)
        print(f"\nNormalization stats saved to {path}.\n")
        return None


def compute_normalization_stats_from_subset(subset_dataset: Dataset) -> None:
    """
    Compute normalization stats from a PyTorch subset dataset.
    
    Args:
        subset_dataset: Subset PyTorch dataset containing the data.
    """
    obs_maxes, que_maxes, ans_maxes = [], [], []
    for i in range(len(subset_dataset)):
        observations, questions, answers = subset_dataset[i]
        
        obs_maxes.append(torch.max(observations).item())
        que_maxes.append(torch.max(questions).item())
        ans_maxes.append(torch.max(answers).item())
    obs_mean_max = sum(obs_maxes) / len(obs_maxes)
    que_mean_max = sum(que_maxes) / len(que_maxes)
    ans_mean_max = sum(ans_maxes) / len(ans_maxes)
    
    normalization_stats = {
        "metadata": f"Normalization stats computed from training dataset for model {config.MODEL_NAME}.\nNormalization performed with two bounds: min=0 and max=mean of max values of timeseries over samples.",
        "obs_mean_max": obs_mean_max,
        "que_mean_max": que_mean_max,
        "ans_mean_max": ans_mean_max
    }
    path = config.DIR_OTHERS_DATA_CHANNEL / f"{config.MODEL_NAME}_normalization_stats.pt"
    torch.save(normalization_stats, path)
    print(f"\nNormalization stats saved to {path}.\n")
    return None
    

def build_dataset() -> None:
    """Build Custom Dataset for MAST SciNet model and save it as .pt file."""

    # Load xarray dataset
    path = config.DIR_RAW_DATA / "mast_magnetics_data_constant_time.nc"
    with xr.open_dataset(path) as ds:
        dataset = ds.load()
    print(f"\nDataset loaded from {path}.")

    # Create custom Dataset
    dataset_custom = mast_dataset(
        dataset,
        chosen_signals=config.CHOSEN_SIGNAL
    )
    train_dataset, val_dataset, test_dataset = random_split(dataset_custom, [config.TRAIN_SIZE, config.VALID_SIZE, config.TEST_SIZE])
    print(f"\nDataset split into train ({len(train_dataset)} samples), valid ({len(val_dataset)} samples), and test ({len(test_dataset)} samples).")

    # Compute and save normalization stats from training dataset
    compute_normalization_stats_from_subset(train_dataset)
    
    # Save Dataset
    path_train = config.DIR_PREPROCESSED_DATA / f"{config.MODEL_NAME}_train_dataset.pt"
    path_valid = config.DIR_PREPROCESSED_DATA / f"{config.MODEL_NAME}_valid_dataset.pt"
    path_test = config.DIR_PREPROCESSED_DATA / f"{config.MODEL_NAME}_test_dataset.pt"
    torch.save(train_dataset, path_train)
    print(f"\nDataset saved to {path_train}.")
    torch.save(val_dataset, path_valid)
    print(f"\nDataset saved to {path_valid}.")
    torch.save(test_dataset, path_test)
    print(f"\nDataset saved to {path_test}.")

    return None


if __name__ == "__main__":
    build_dataset()