from pathlib import Path
# import yaml

#from magnetics_diagnostic_analysis.ml_tools.random_seed import seed_everything
#from magnetics_diagnostic_analysis.ml_tools.pytorch_device_selection import select_torch_device

class Config:
    """Global variables configuration"""

    ##########################################    
    ### Paths
    SUFFIX = "scinet"
    DIR_DATA = Path(__file__).absolute().parent.parent.parent.parent / "data"
    DIR_SYNTHETIC_DATA = DIR_DATA / f"synthetic/{SUFFIX}"
    # DIR_RAW_DATA = DIR_DATA / f"raw"
    # DIR_PREPROCESSED_DATA = DIR_DATA / f"preprocessed/{SUFFIX}"
    # DIR_PROCESSED_DATA = DIR_DATA / f"processed/{SUFFIX}"
    DIR_PARAMS_CHECKPOINTS = Path(__file__).absolute().parent / f"checkpoints"
    DIR_MODEL_PARAMS = Path(__file__).absolute().parent.parent.parent.parent / f"results/model_params/{SUFFIX}"
    DIR_FIGURES = Path(__file__).absolute().parent.parent.parent.parent / f"results/figures/{SUFFIX}"

    ### PyTorch device & set seed for reproducibility
    DEVICE = select_torch_device(temporal_dim="parallel")
    SEED = 42
    seed_everything(SEED)


    ### Data parameters
    N_SAMPLES = 50000
    KAPA_RANGE = (1.0, 10.0)
    B_RANGE = (0.01, 0.1)
    MAXTIME = 10.0
    TIMESTEPS = 100


    ### DataLoader parameters
    TRAIN_VALID_SPLIT = 0.8


    ### SCINET architecture
    M_INPUT_SIZE = TIMESTEPS
    M_ENC_HIDDEN_SIZES = [500, 100]
    M_LATENT_SIZE = 3
    M_QUESTION_SIZE = 1
    M_DEC_HIDDEN_SIZES = [400, 400]
    M_OUTPUT_SIZE = 1

    ### Hyperparameters
    BATCH_SIZE_TRAIN = 512
    BATCH_SIZE_VALID = 512
    FIRST_LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5     # if needed
    KLD_BETA = 0.003

    ### Train parameters
    NUM_EPOCHS = 150
    ES_PATIENCE = 12
    ES_MIN_DELTA = 5e-4
    GC_MAX_NORM = 1.0
    LRS_FACTOR = 0.66
    LRS_PATIENCE = 5
    LRS_MIN_LR = 1e-7
    LRS_MIN_DELTA = 1e-4




    ### Data scrapping from MAST API

    ### Others
    BEST_MODEL_NAME = "pendulum_scinet_final"

   

    # Method to update parameters
    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

# Global instance
config = Config()
