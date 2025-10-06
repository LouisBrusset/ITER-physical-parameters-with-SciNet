import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import xarray as xr

from physical_parameters_SciNet.model_instances.n2_setting_mast_constant_time import config


class mast_dataset(Dataset):
    def __init__(self, dataset, chosen_signals: str = "ip"):
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
            observations = self.dataset[self.chosen_var].sel(shot_id=shot_id, **{f"{self.chosen_var}_channel": self.chosen_channel}).values
        else:                                   # Single-channel variable
            observations = self.dataset[self.chosen_var].sel(shot_id=shot_id).values
        
        subsampling_factor = config.SUBSAMPLING_FACTOR
        observations_tensor = torch.tensor(observations[::subsampling_factor], dtype=torch.float32)
        questions_tensor = torch.tensor(questions[::subsampling_factor], dtype=torch.float32)
        answers_tensor = torch.tensor(answers[::subsampling_factor], dtype=torch.float32)
        return observations_tensor, questions_tensor, answers_tensor
    

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
        chosen_signals="ip"
    )
    train_dataset, val_dataset, test_dataset = random_split(dataset_custom, [config.TRAIN_SIZE, config.VALID_SIZE, config.TEST_SIZE])
    print(f"\nDataset split into train ({len(train_dataset)} samples), valid ({len(val_dataset)} samples), and test ({len(test_dataset)} samples).")
    
    # Save Dataset
    path_train = config.DIR_PREPROCESSED_DATA / "mast_scinet_train_dataset.pt"
    path_valid = config.DIR_PREPROCESSED_DATA / "mast_scinet_valid_dataset.pt"
    path_test = config.DIR_PREPROCESSED_DATA / "mast_scinet_test_dataset.pt"
    torch.save(train_dataset, path_train)
    print(f"\nDataset saved to {path_train}.")
    torch.save(val_dataset, path_valid)
    print(f"\nDataset saved to {path_valid}.")
    torch.save(test_dataset, path_test)
    print(f"\nDataset saved to {path_test}.")

    return None


if __name__ == "__main__":
    build_dataset()