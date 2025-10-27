import torch

def print_torch_info():
    print("\nTorch version? ", torch.__version__)
    print("Cuda?          ", torch.cuda.is_available())

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\nGPU number : {device_count}")

        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\nNo NVIDIA GPU is available for PyTorch.")


def select_torch_device(temporal_dim: str = "sequential", device_id: int | None = None) -> torch.device:
    """
    Select the appropriate PyTorch device based on the availability of CUDA and the specified temporal dimension.

    Args:
        temporal_dim (str): The temporal dimension type ("sequential" or "parallel").
            - If "sequential", we want to force the use of one specific GPU if available and only one.
              Indeed, the batches must be treated in chronological order, due to the ConvLSTM module and the time deque between batches.
            - If "parallel", the function will allow for data parallelism across multiple GPUs.

    Returns:
        torch.device: The selected PyTorch device.
    """

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            if temporal_dim == "sequential":
                if device_id is not None:
                    torch.cuda.set_device(device_id)
                    device = torch.device(f"cuda:{device_id}")
                else:
                    torch.cuda.set_device(0)
                    device = torch.device("cuda:0")
            elif temporal_dim == "parallel":
                device = torch.device("cuda")
            else:
                raise ValueError("temporal_dim must be either 'sequential' or 'parallel'")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("\n\nChoosen device =", device)
    return device


if __name__ == "__main__":

    print_torch_info()
    
    device = select_torch_device()

    print(f"Using device: {device}")
