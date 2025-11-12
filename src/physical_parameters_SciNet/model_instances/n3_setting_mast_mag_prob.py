from pathlib import Path
import torch
# import yaml

from physical_parameters_SciNet.ml_tools.random_seed import seed_everything
from physical_parameters_SciNet.ml_tools.pytorch_device_selection import select_torch_device

class Config:
    """Global variables configuration"""

    ##########################################    

    ### PyTorch device & set seed for reproducibility
    # DEVICE = select_torch_device(temporal_dim="parallel")  # GPU 0 is saturated
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # Use GPU 1 instead
    SEED = 42
    seed_everything(SEED)


    ### Data parameters
    # Real tokamak data: FAIR-MAST API
    N_SAMPLES = 6102
    GROUPS = ["magnetics"]
    CHOSEN_SIGNAL = "b_field_pol_probe_ccbv_field::AMB_CCBV03"   # "ip" or "all"
    MIN_TIME = -0.05
    MAX_TIME = 0.45
    TIMESTEPS = 1000
    SUBSAMPLING_FACTOR = 5   # To reduce the input size (e.g. from 1000 to 200)


    ### DataLoader parameters
    SPLIT_RATIO = [0.7, 0.15, 0.15]   # train, valid, test
    TRAIN_SIZE = int(N_SAMPLES * SPLIT_RATIO[0])
    VALID_SIZE = int(N_SAMPLES * SPLIT_RATIO[1])
    TEST_SIZE = N_SAMPLES - TRAIN_SIZE - VALID_SIZE


    ### SCINET architecture
    M_INPUT_SIZE = 200
    M_ENC_HIDDEN_SIZES = [500, 200]
    M_LATENT_SIZE = 10
    M_QUESTION_SIZE = 200
    M_DEC_HIDDEN_SIZES = [500, 500, 500]
    M_OUTPUT_SIZE = 200

    ### Hyperparameters
    BATCH_SIZE_TRAIN = 512
    BATCH_SIZE_EVAL = 512
    FIRST_LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    KLD_BETA = 0.001

    ### Train parameters
    NUM_EPOCHS = 200
    ES_PATIENCE = 12
    ES_MIN_DELTA = 1e-5
    GC_MAX_NORM = 1.5
    LRS_FACTOR = 0.75
    LRS_PATIENCE = 6
    LRS_MIN_LR = 1e-7
    LRS_MIN_DELTA = 1e-4

    ### Paths
    DIR_DATA = Path(__file__).absolute().parent.parent.parent.parent / "data"
    DIR_RAW_DATA = DIR_DATA / f"raw"
    DIR_PREPROCESSED_DATA = DIR_DATA / f"preprocessed/{CHOSEN_SIGNAL}_channel"
    DIR_PROCESSED_DATA = DIR_DATA / f"processed/{CHOSEN_SIGNAL}_channel"
    DIR_OTHERS_DATA = DIR_DATA / f"others"
    DIR_OTHERS_DATA_CHANNEL = DIR_OTHERS_DATA / f"{CHOSEN_SIGNAL}_channel"
    DIR_PREPROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    DIR_PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    DIR_OTHERS_DATA_CHANNEL.mkdir(parents=True, exist_ok=True)

    DIR_RESULTS = Path(__file__).absolute().parent.parent.parent.parent / f"results"
    DIR_MODEL_PARAMS = DIR_RESULTS / f"model_params"
    DIR_PARAMS_CHANNEL = DIR_MODEL_PARAMS / f"{CHOSEN_SIGNAL}_channel"
    DIR_PARAMS_CHECKPOINTS = DIR_PARAMS_CHANNEL / f"checkpoints"
    DIR_PARAMS_CHANNEL.mkdir(parents=True, exist_ok=True)
    DIR_PARAMS_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    DIR_FIGURES = DIR_RESULTS / f"figures"
    DIR_FIGURES_CHANNEL = DIR_FIGURES / f"{CHOSEN_SIGNAL}_channel"
    DIR_FIGURES_CHANNEL.mkdir(parents=True, exist_ok=True)


    ### Others
    MODEL_NAME = "mast_scinet"
    BEST_MODEL_NAME = f"{MODEL_NAME}_final"

   

    # Method to update parameters
    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

# Global instance
config = Config()
