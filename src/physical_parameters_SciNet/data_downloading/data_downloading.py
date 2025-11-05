# Third party imports
import numpy as np
import pandas as pd
import xarray as xr
import tqdm

# Standard library imports
import time
import json
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import partial

# Local package imports
# from physical_parameters_SciNet.model_instances.n2_setting_mast_constant_time import config
from physical_parameters_SciNet.model_instances.n3_setting_mast_mag_prob import config



def to_dask(shot: int, group: str, level: int = 2) -> xr.Dataset:
    """
    Return a Dataset from the MAST Zarr store.

    Args:
        shot (int): Shot ID to retrieve data for.
        group (str): Diagnostic group to retrieve data from.
        level (int): Data level to retrieve (default is 2).

    Returns:
        xr.Dataset: The Dask Dataset for the specified shot and group.
    """
    return xr.open_zarr(
        f"https://s3.echo.stfc.ac.uk/mast/level{level}/shots/{shot}.zarr",
        group=group
    )

def retry_to_dask(shot_id: int, group: str, level: int = 2, retries: int = 5, delay: int = 1) -> xr.Dataset:
    """
    Retry loading a shot's data as a Dask Dataset with exponential backoff.

    Args:
        shot_id (int): Shot ID to retrieve data for.
        group (str): Diagnostic group to retrieve data from.
        level (int): Data level to retrieve (default is 2).
        retries (int): Number of retry attempts (default is 5).
        delay (int): Delay in seconds between retries (default is 1).

    Returns:
        xr.Dataset: The Dask Dataset for the specified shot and group.
    or Error
    """
    for attempt in range(retries):
        try:
            return to_dask(shot_id, group, level=level)
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying connection to {shot_id} in group {group} (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                raise e


def filter_xr_dataset_channels(data: xr.Dataset, var_channel: list[str]) -> xr.Dataset:
    """
    Filters the channels of the data_train variables according to var_channel_df.
    Removes variables not listed in var_channel_df.

    Args:
        data_train (xr.Dataset): Dataset containing variables with or without the “channel” dimension.
        var_channel (list[str]): List of variable::channel strings to keep.

    Returns:
        xr.Dataset: Filtered dataset.
    """
    # Find good var::channels for the chosen groups
    var_channel_df = pd.DataFrame(var_channel, columns=["full_name"])
    var_channel_df[["variable", "channel"]] = var_channel_df["full_name"].str.split("::", expand=True)
    # Variables for the loop
    variables_to_keep = set(var_channel_df["variable"])
    
    filtered_data = {}
    for variable in variables_to_keep:
        da = data[variable]
        da_dims = da.dims
        coord = [dim for dim in da_dims if 'channel' in dim]
        if not coord:
            da_filtered = da
        else:
            coord_names = da[coord[0]].values
            coord_names_to_keep = var_channel_df.loc[var_channel_df["variable"] == variable, "channel"].tolist()
            channel_indices = np.where(np.isin(coord_names, coord_names_to_keep))[0]
            da_filtered = da.isel({coord[0]: channel_indices})
        filtered_data[variable] = da_filtered
    final = xr.Dataset(filtered_data)

    return final

def impute_to_zero(data: xr.Dataset) -> xr.Dataset:
    """
    It may rest some NaN values in the dataset, due to the interpolation.
    This function imputes missing values in the dataset by replacing them with zero.
    We loop over the variables, so that we do not modify the coordinates and the dataset structure (attrs for instance)
    """
    result = data.copy()
    for var in result.data_vars:
        result[var] = result[var].fillna(0)
    return result


def single_shot_constant_time(shot_id: int, groups: list[str], channels: list[str], time_base: xr.DataArray, verbose: bool = False):
    """
    One shot processing for build_level_2_data_constant_time_parallel.
    Retrieve specified groups of diagnostics from shots previously selected.
        - The "summary" group is used to get the reference plasma current (ip_ref).
        - The "pulse_schedule" group is used to get the scheduled plasma current (i_plasma).
        - Other specified groups are used to get the observation signals.
          Here, we focus on the "magnetics" group.

    Args:
        shots (list[int]): List of shot IDs to retrieve data for.
        groups (list[str]): List of diagnostic groups to retrieve.
        channels (list[str]): List of diagnostic channels to retrieve.
        time_base (xr.DataArray): Time base to interpolate data to.
        verbose (bool): If True, print additional information during processing.

    Returns:
        xr.Dataset: An xarray Dataset containing the requested diagnostic data.
    """
    signal = []
    try:
        # Load "plasma current" reference
        answer = retry_to_dask(shot_id, "summary", level=2)
        ip_ref = answer['ip'].rename('ip_ref')
        ip_ref = ip_ref.interp({"time": time_base}, method="linear")
        ip_ref = ip_ref.assign_coords({"time": time_base})
        ip_ref.attrs |= {"group": "summary"}
        signal.append(ip_ref.to_dataset())

        # Load "pulse_schedule"
        question = retry_to_dask(shot_id, "pulse_schedule", level=2)
        ip_scheduled = question['i_plasma']
        ip_scheduled = ip_scheduled.interp({"time": time_base}, method="linear")
        ip_scheduled = ip_scheduled.assign_coords({"time": time_base})
        ip_scheduled.attrs |= {"group": "pulse_schedule"}
        signal.append(ip_scheduled.to_dataset())

        # Load observation groups
        for group in groups:
            if verbose:
                print(f"Loading group {group} for shot {shot_id}...")
            try:
                data = retry_to_dask(shot_id, group, level=2)
            except (IndexError, KeyError):
                continue
            data = filter_xr_dataset_channels(data, channels)

            other_times = set()
            for var in data.data_vars:
                time_dim = next((dim for dim in data[var].dims if dim.startswith('time')), 'time')
                data[var] = data[var].interp({time_dim: time_base}, method="linear")
                data[var] = data[var].assign_coords({time_dim: time_base})
                data[var].attrs |= {"group": group}
                data[var] = data[var].transpose("time_base", ...)
                other_times.add(time_dim)
            data = data.drop_vars(other_times)
            signal.append(data)

        signal_ds = xr.merge(signal, join="outer")
        signal_ds = signal_ds.drop_vars("time").rename({"time_base": "time"})
        signal_ds = signal_ds.assign_coords(shot_id=shot_id)
        return signal_ds
        
    except Exception as e:
        print(f"Error processing shot {shot_id}: {e}")
        return None
    

def build_level_2_data_constant_time_parallel(shots: list[int], groups: list[str], channels: list[str], verbose: bool = False, max_workers: int = None) -> xr.Dataset:
    """
    Parallel version of build_level_2_data_constant_time using ThreadPoolExecutor.

    Data are interpolated to the time base of an arbitrary reference.
        -> Goal: have a constant time base for all variables.
    Selection is done on the channels listed in previously selected 'channels' (format: variable::channel).
    Missing values are imputed to zero.

    Args:
        shots (list[int]): List of shot IDs to retrieve data for.
        groups (list[str]): List of diagnostic groups to retrieve.
        channels (list[str]): List of diagnostic channels to retrieve.
        verbose (bool): If True, print additional information during processing.
        max_workers (int): Maximum number of threads to use (default is None, which uses the number of processors on the machine).

    Returns:
        xr.Dataset: An xarray Dataset containing the requested diagnostic data.
    """
    time_base = np.linspace(config.MIN_TIME, config.MAX_TIME, config.TIMESTEPS)
    time_base = xr.DataArray(time_base, dims=["time_base"], coords={"time_base": time_base})
    
    process_func = partial(
        single_shot_constant_time,
        groups=groups,
        channels=channels,
        time_base=time_base,
        verbose=verbose
    )       # Partial function with fixed arguments
    
    datasets = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Using {executor._max_workers} workers for parallel processing.")
        future_to_shot = {executor.submit(process_func, shot_id): shot_id for shot_id in shots}

        for future in tqdm.tqdm(as_completed(future_to_shot), total=len(shots), desc="Processing shots (parallel)"):
            shot_id = future_to_shot[future]
            try:
                result = future.result()
                if result is not None:
                    datasets.append(result)
                    if verbose:
                        print(f"Successfully processed shot {shot_id}")
            except Exception as e:
                print(f"Shot {shot_id} generated an exception: {e}")
    if not datasets:
        raise ValueError("No shots were successfully processed")
    
    final = xr.concat(datasets, dim="shot_id", join="outer", coords="minimal", combine_attrs="drop_conflicts")
    final = impute_to_zero(final)
    final = final.sortby("shot_id")
    if 'license' in final.attrs:
        del final.attrs['license']
    
    print(f"Dataset ok - processed {len(datasets)} shots")
    return final


def single_shot_variable_time(shot_id: int, groups: list[str], channels: list[str], verbose: bool = False):
    """
    One shot processing for build_level_2_data_variable_time_parallel.
    Retrieve specified groups of diagnostics from shots previously selected.
        - The "summary" group is used to get the reference plasma current (ip_ref).
        - The "pulse_schedule" group is used to get the scheduled plasma current (i_plasma).
        - Other specified groups are used to get the observation signals.
          Here, we focus on the "magnetics" group.

    Args:
        shots (list[int]): List of shot IDs to retrieve data for.
        groups (list[str]): List of diagnostic groups to retrieve.
        channels (list[str]): List of diagnostic channels to retrieve.
        verbose (bool): If True, print additional information during processing.

    Returns:
        xr.Dataset: An xarray Dataset containing the requested diagnostic data.
    """
    signal = []
    try:
        # Load "plasma current" reference
        answer = retry_to_dask(shot_id, "summary", level=2)
        time_ref = answer.time
        ip_ref = answer['ip'].rename('ip_ref')
        ip_ref.attrs |= {"group": "summary"}
        signal.append(ip_ref.to_dataset())

        # Load "pulse_schedule"
        question = retry_to_dask(shot_id, "pulse_schedule", level=2)
        ip_scheduled = question['i_plasma']
        ip_scheduled = ip_scheduled.interp({"time": time_ref}, method="linear")
        ip_scheduled = ip_scheduled.assign_coords({"time": time_ref})
        ip_scheduled.attrs |= {"group": "pulse_schedule"}
        signal.append(ip_scheduled.to_dataset())

        # Load observation groups
        for group in groups:
            if verbose:
                print(f"Loading group {group} for shot {shot_id}...")
            try:
                data = retry_to_dask(shot_id, group, level=2).interp({"time": time_ref}, method="linear")
            except (IndexError, KeyError):
                continue
            data = filter_xr_dataset_channels(data, channels)

            other_times = set()
            for var in data.data_vars:
                if verbose:
                    print(f"Processing variable: {var}")
                time_dim = next((dim for dim in data[var].dims if dim.startswith('time')), 'time')
                if time_dim != "time":
                    other_times.add(time_dim)
                    data[var] = data[var].interp({time_dim: time_ref}, method="linear")
                data[var].attrs |= {"group": group}
                data[var] = data[var].transpose("time", ...)
            data = data.drop_vars(other_times)
            signal.append(data)

        signal_ds = xr.merge(signal)
        signal_ds = impute_to_zero(signal_ds)
        signal_ds["shot_id"] = ("time", shot_id * np.ones(len(time_ref), int))

        return signal_ds
        
    except Exception as e:
        print(f"Error processing shot {shot_id}: {e}")
        return None
    

def build_level_2_data_variable_time_parallel(shots: list[int], groups: list[str], channels: list[str], verbose: bool = False, max_workers: int = None) -> xr.Dataset:
    """
    Parallel version of build_level_2_data_variable_time using ThreadPoolExecutor.

    Data are interpolated to the time base of an arbitrary reference.
        -> Goal: have a variable time base for all variables.
    Selection is done on the channels listed in previously selected 'channels' (format: variable::channel).
    Missing values are imputed to zero.

    Args:
        shots (list[int]): List of shot IDs to retrieve data for.
        groups (list[str]): List of diagnostic groups to retrieve.
        channels (list[str]): List of diagnostic channels to retrieve.
        verbose (bool): If True, print additional information during processing.
        max_workers (int): Maximum number of threads to use (default is None, which uses the number of processors on the machine).

    Returns:
        xr.Dataset: An xarray Dataset containing the requested diagnostic data.
    """
    
    process_func = partial(
        single_shot_variable_time,
        groups=groups,
        channels=channels,
        verbose=verbose
    )       # Partial function with fixed arguments
    
    datasets = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"Using {executor._max_workers} workers for parallel processing.")
        future_to_shot = {executor.submit(process_func, shot_id): shot_id for shot_id in shots}

        for future in tqdm.tqdm(as_completed(future_to_shot), total=len(shots), desc="Processing shots (parallel)"):
            shot_id = future_to_shot[future]
            try:
                result = future.result()
                if result is not None:
                    datasets.append(result)
                    if verbose:
                        print(f"Successfully processed shot {shot_id}")
            except Exception as e:
                print(f"Shot {shot_id} generated an exception: {e}")
    if not datasets:
        raise ValueError("No shots were successfully processed")
    
    indices = sorted(range(len(datasets)), key=lambda i: datasets[i].shot_id.values[0])
    datasets = [datasets[i] for i in indices]
    final = xr.concat(datasets, "time")
    if 'license' in final.attrs:
        del final.attrs['license']
    
    print(f"Dataset ok - processed {len(datasets)} shots")
    return final



def load_data(shots: list[int], groups: list[str], channels: list[str], save_name: str, verbose: bool = False) -> None:
    """
    Load data from files or build it if not available.

    Args:
        shots (list[int]): List of shot IDs to retrieve data for.
        groups (list[str]): List of diagnostic groups to retrieve data from.
        channels (list[str]): List of diagnostic channels to retrieve data from.
        save_name (str): Name of the file to save the data to.
        verbose (bool): If True, print additional information during processing.

    Returns:
        None
    """

    path = config.DIR_RAW_DATA
    path.mkdir(exist_ok=True)
    filename_data = path / save_name

    try:
        with open(filename_data, "rb"):
            print("Files already exist!")
    except FileNotFoundError:
        dataset = build_level_2_data_variable_time_parallel(
            shots, 
            groups=groups, 
            channels=channels,
            verbose=verbose,
            max_workers=None # Use all available processors
        )
        print("Saving to netCDF...")
        dataset.to_netcdf(filename_data)
        print("netCDF ok")
    return None


if __name__ == "__main__":

    # Choose shot_ids and diagnostic groups
    path = config.DIR_OTHERS_DATA / "result_nan_lists_magnetics.json"
    if path.exists():
        with open(path, "r") as f:
            result_nan_lists_magnetics = json.load(f)
    else:
        raise FileNotFoundError(f"File {path} not found.")
    shots_list = [int(shot) for shot in result_nan_lists_magnetics["good_shot_ids"]]
    vars_list = result_nan_lists_magnetics["good_vars_ids"]
    # n_samples = 100       # Number of shots to load
    # np.random.shuffle(shots_list)
    # shots_list = shots_list[:n_samples]
    print("Type of shots: ", type(shots_list))
    print("Chosen shots: \n", shots_list)


    # Load the data
    save_name1 = "mast_magnetics_data_variable_time.nc"
    load_data(
        shots=shots_list, 
        groups=config.GROUPS,
        channels=vars_list,
        save_name=save_name1,
        verbose=False
        )
    print("Data loading completed.\n")
    

    # Check the loaded data
    ##path = config.DIR_RAW_DATA / save_name1
    ##if path.exists():
    ##    with xr.open_dataset(path) as train:
    ##        subset = train.isel(time=slice(0, 10000))   # Subset for faster loading
    ##        data = subset.load()
    ##else:
    ##    raise FileNotFoundError(f"File {path} not found.")
    ##print("\nDataset loaded successfully.")
    ##print("===============")
    ##print(data)
    ##print("===============")
    ##print(data.data_vars)
    ##print("===============")
    ##print(data["ip"].attrs)
    ##print("===============")