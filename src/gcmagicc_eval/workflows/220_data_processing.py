# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python (all-dev)
#     language: python
#     name: all-dev
# ---

# %% [markdown]
# # 220 - Data procecessing workbook

# %% [markdown]
# Malte, 6 Nov 2024 
# - trying to get ready for the Nicolai runs.. - still monthly. for ETH server crunching. 
#
#
#

# %% [markdown]
# ### Still to think about.. 
#
#     
# Notes and Recommendations:
# Error Handling and Logging: Incorporate error handling (try-except blocks) and logging to track the processing stages and capture any issues.
#
# Function Details: Fill in the details for functions like regrid_dataset_if_needed, combine_datasets, etc., based on their definitions and expected inputs.
#
# Parallel Processing: Consider parallelizing the loop if the dataset is large and the operations are time-consuming.
#
# Data Consistency Checks: Add checks to ensure data integrity, especially when stitching datasets and computing global means.
#
# Documentation: Document each step and function thoroughly for clarity and future reference.
#
# Testing: Initially, run the workflow with a subset of your data to ensure that all parts work as expected.
#
# Output Verification: After saving the preprocessed datasets, it's good practice to load some of them back in to verify that they were saved correctly.

# %% [markdown]
# ### setting up environment / imports 

# %%
# Standard library imports
import os
import re
from datetime import datetime
import ipynbname
import gc

from pathlib import Path
import fsspec                     # gcsfs, s3fs, local FS under one roof

# Third-party library imports for data manipulation and analysis
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import zarr
from netCDF4 import Dataset
import xesmf as xe
import fnmatch

# Time handling imports
import cftime
import nc_time_axis
import scipy.ndimage

# Visualization library imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Data intake and preprocessing imports
import intake
import xmip

# Utilities
from dask.delayed import delayed
#from dask.distributed import Client, wait
# Initialize Dask client
#try:
#    client.close()
#except NameError:
#    pass
# client = Client()
#from joblib import Parallel, delayed, parallel_backend

import difflib
from importlib import reload
import importlib.util
from tqdm.auto import tqdm

from xarray.coding.cftimeindex import CFTimeIndex
import logging

# Import from operations
import sys
SRC_DIR = Path("../src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cmip6cruncher.operations
reload(cmip6cruncher.operations)
from cmip6cruncher.operations import *
from cmip6cruncher.operations import open_any_cmip

# %%
# %load_ext line_profiler

# %%
# Setup central logging file.. # Get current date and time
now = datetime.now()
date_str = now.strftime("%d%b%Y")  # Format date as '10Jan2024'
time_str = now.strftime("%H-%M")   # Format time as 'HH-MM'

# Create the log file name
log_file_name = f"process_logs_{date_str}_{time_str}.log"

#Specify the subdirectory name
log_subdirectory = 'process_logs'

# Create the subdirectory if it does not exist
os.makedirs(log_subdirectory, exist_ok=True)

# Full path for the log file
log_file_path = os.path.join(log_subdirectory, log_file_name)

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# %%
# ------------------------------------------------------------------
# Dask client: use the existing one if 230 created it,
# otherwise start a small local cluster so 220 can run standalone.
# ------------------------------------------------------------------
# Standalone-safe runtime defaults for:
#   pixi run python 220_data_processing.py
# Override any of these via environment variables when needed.
dask_n_workers = int(os.getenv("CMIP220_DASK_N_WORKERS", "1"))
dask_threads_per_worker = int(os.getenv("CMIP220_DASK_THREADS_PER_WORKER", "8"))
dask_memory_limit = os.getenv("CMIP220_DASK_MEMORY_LIMIT", "0")  # "0" disables per-worker memory cap
regrid_num_jobs = int(os.getenv("CMIP220_REGRID_JOBS", "1"))
globalmean_num_jobs = int(os.getenv("CMIP220_GLOBALMEAN_JOBS", "1"))

# Avoid Dask worker pausing/terminating due to "unmanaged memory" in this
# single-process workflow. Keep defaults unless caller already set them.
os.environ.setdefault("DASK_DISTRIBUTED__WORKER__MEMORY__TARGET", "false")
os.environ.setdefault("DASK_DISTRIBUTED__WORKER__MEMORY__SPILL", "false")
os.environ.setdefault("DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE", "false")
os.environ.setdefault("DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE", "false")

from dask.distributed import Client, default_client
try:
    default_client()              # succeeds if 230 already made one
except ValueError:
    # nothing running – spin up a local cluster (adjust to taste)
    # client = Client(n_workers=8, threads_per_worker=2, memory_limit='16GB')
    # ERA5 processing, 19 Feb 2026
    # Use thread-based workers (no separate worker processes) to avoid
    # multiprocessing spawn bootstrap errors when this notebook-exported
    # script is executed directly with Python 3.12.
    client = Client(
        n_workers=dask_n_workers,
        threads_per_worker=dask_threads_per_worker,
        processes=False,
        memory_limit=dask_memory_limit,
    )
    print("Started local Dask cluster for standalone run:", client)


# %% [markdown]
# ## General settings: 

# %%
crunchtimedomain = 'monthly'

# %%
# set the target resolution. A 1 stands for a 1x1 degree grid.. a 2 for a 2x2 degree grid etc.. 
# target_resolution = 0.25
# target_resolution = 1
# ERA5 processing, 19 Feb 2026
target_resolution = 1

# %%
use_paralleloptions = False

# %%
# choose behavior: 
#  - 'overwrite' will always run the calculation and overwrite any existing .nc  
#  - 'skipcalculation' will omit any (source, scen, member) if there's already a DAT_*.nc
manage_existing_output_files = 'overwrite'  # or 'overwrite', 'skipcalculation'


# %%
##################################################################################
###################  options for nametag_thissearch         ######################
##################################################################################


# define the nametag for the files2crunch file that shall be crunched, which inludes all the filenames etc.. '
#nametag_thissearch = 'TasPrR3_second_15jan24_plusparent'
#nametag_thissearch = 'TasPrUasVasPslRsdsHussEvspsblLmonMrsoCltR3_23jan24_plusparent'
#nametag_thissearch = 'cmip6_All_TasPrEvspsblMrsoSfcwindHursUasVasPslRsdsHussCltRsdtRsutRlut_14may24_plusparent'
#nametag_thissearch = 'cmip6_NoNone_TasPrEvspsblMrsoSfcwindHursUasVasPslRsdsHussCltRsdtRsutRlut_21may24_plusparent'

#daily data test.. 
#nametag_thissearch = 'cmip6_AllInclMissingVars_TasPrEvspsblMrsoSfcwindHursUasVasPslRsdsHussCltRsdtRsutRlut_24may24_plusparent'
#nametag_thissearch = 'TasSiareanSiConcADaily_29jan24_updated_plusparent'
# nametag_thissearch = 'cmip6_Daily_NoNone_TasTasminTasmaxPrSfcwindHursUasVasPslHussClt_27may24_plusparent'



##################################################################################
###################   optios for filter_rows_files2crunch   ######################
##################################################################################

# specify the filter for which CMIP6 files you want to crunch
#filter_rows_files2crunch = {'source_id': 'HadGEM3-GC31-MM', 'experiment_id': ['historical', 'ssp585']}
#filter_rows_files2crunch = {'source_id': 'HadGEM3-GC31-MM', 'experiment_id': 'historical'}
#filter_rows_files2crunch = {'source_id': 'HadGEM3-GC31-LL', 'member_id': 'r1i1p1f3'}
# GOOD - filter_rows_files2crunch = {'source_id': 'HadGEM3-GC31-LL', 'member_id': 'r1i1p1f3'}
#filter_rows_files2crunch = {'source_id': 'HadGEM3-GC31-LL', 'experiment_id': 'ssp126', 'member_id': 'r1i1p1f3'}
#filter_rows_files2crunch = {'source_id': ['MRI-ESM2-0','MIROC6','IPSL-CM6A-LR'], 'member_id':  'r1i1p1f1'}
# GOOD - filter_rows_files2crunch = {'source_id': 'MRI-ESM2-0', 'member_id':  'r1i1p1f1'}
# FAULTY - filter_rows_files2crunch = {'source_id': 'MIROC6', 'member_id':  'r1i1p1f1'}
# FAULTY - filter_rows_files2crunch = {'source_id':'IPSL-CM6A-LR', 'member_id':  'r1i1p1f1'}
# GOOD - filter_rows_files2crunch = {'source_id':'UKESM1-0-LL', 'member_id':  'r1i1p1f2'}
# GOOD - filter_rows_files2crunch = {'source_id':'GFDL-ESM4', 'member_id':  'r1i1p1f1'}
# GOOD - filter_rows_files2crunch = {'source_id': 'CNRM-CM6-1', 'member_id': 'r1i1p1f2'}
# filter_rows_files2crunch = {'source_id': 'CanESM5', 'member_id': 'r1i1p1f1'}
#filter_rows_files2crunch = {'source_id': 'MRI-ESM2-0', 'member_id':  'r1i1p1f1', 'experiment_id': 'ssp585'}

# filter_rows_files2crunch = [1]
#filter_rows_files2crunch = list(range(1, 4))

#filter_rows_files2crunch = {'source_id': 'HadGEM3-GC31-MM', 'experiment_id': 'abrupt-4xCO2', 'member_id': 'r1i1p1f3'}
#filter_rows_files2crunch = 'all'
# filter_rows_files2crunch = {'source_id': 'GFDL-CM4', 'experiment_id':  'abrupt-4xCO2', 'member_id': 'r1i1p1f1'}
#filter_rows_files2crunch = {'source_id': 'IPSL-CM6A-LR', 'experiment_id': 'historical', 'member_id': 'r8i1p1f1'}
#filter_rows_files2crunch = {'experiment_id': 'ssp370', 'member_id': 'r1i1p1f1'}


# options for filter_rows_files2crunch: 
# - [2] or [1,3,5,10] for the lines in the file2crunch... 
# - 'all'
# - {'source_id', 'experiment_id', 'member_id'}

# %% tags=["parameters"]
# Parameters
#nametag_thissearch = 'cmip6_Daily_NoNone_TasTasminTasmaxPrSfcwindHursUasVasPslHussClt_27may24_plusparent'
#nametag_thissearch = 'cmip6_Monthly_NoNone_TasTasminTasmaxPrEvspsblMrsoSfcwindHursUasVasPslRsdsHussCltRsdtRsutRlut_10jul24.csv'
# nametag_thissearch = 'cmip6_Monthly_NoNone_TasTasminTasmaxPrEvspsblMrsoSfcwindHursUasVasPslRsdsHussCltRsdtRsutRlut_10jul24_plusparent'
#nametag_thissearch = 'cmip6_Monthly_NoNone_TasTasminTasmaxPrEvspsblMrsoSfcwindHursUasVasPslRsdsHussCltRsdtRsutRlutRtmt_21aug24_plusparent'
#nametag_thissearch = 'cmip6_Monthly_NoNone_TasTasminTasmaxPrSfcwindPslRsdtRsutRlutRtmt_27apr25'
nametag_thissearch = 'cmip6_Monthly_NoNone_TasTsTasminTasmaxPrSfcwindPslRsdtRsutRlutRtmt_28apr25'
nametag_thissearch = 'cmip6_Monthly_NoNone_TasTsTasminTasmaxPrSfcwindPslRsdtRsutRlutRtmt_FOGETH_22may25_1442'
nametag_thissearch = 'cmip6_Monthly_NoNone_TasTsTasminTasmaxPrEvspsblMrsoSfcwindHursPslRsdsHussCltRsdtRsutRlutRtmt_FOGETH_23may25_0028'
#nametag_thissearch = 'cmip6_Monthly_NoNone_TasTasminTasmaxPrEvspsblMrsoSfcwindHursUasVasPslRsdsHussCltRsdtRsutRlutRtmt_29oct24_test2'
#filter_rows_files2crunch = [20,29,30, 31, 32, 33, 34]
filter_rows_files2crunch = 'all'
# lon_convention = '360'  # '180' (-180..180) or '360' (0..360)
# ERA5 processing, 19 Feb 2026
lon_convention = '360'  # 0..360 longitude convention for ERA5 output
# filter_rows_files2crunch = [0]
# filter_rows_files2crunch ={'source_id': 'MIROC6', 'experiment_id': 'ssp126', 'member_id': 'r48i1p1f1'}
# filter_rows_files2crunch ={'source_id': 'ACCESS-CM2', 'experiment_id': 'historical', 'member_id': 'r1i1p1f1'}
# filter_rows_files2crunch ={'source_id': 'CanESM5', 'experiment_id': 'historical', 'member_id': 'r13i1p1f1'}
# DAT_CanESM5_historical_r13i1p1f1
# DAT_ACCESS-CM2_hist-GHG_r2i1p1f1_clt-day-evspsbl-hurs-huss-month-mrso-pr-psl-rlut-rsds-rsdt-rsut-rtmt-sfcWind-tas-tasmax-tasmin-ts-year_pr_map

# %%
# Define input folder with specifications 
specfoldern = '../data/DataAcquisition_SearchResults'

# Read the DataFrame from the CSV file
files2crunch_loaded_df = pd.read_csv(os.path.join(specfoldern,f'{nametag_thissearch}.csv'), index_col=[0, 1, 2])


# %%
files2crunch_loaded_df.columns


# %%
if False: 
    # Define the target values
    target_source_id = 'CNRM-CM6-1'
    target_experiment_id = '1pctCO2'
    target_member_id = 'r1i1p1f2'

    # Find the row index
    row_index = files2crunch_loaded_df.index.get_loc((target_source_id, target_experiment_id, target_member_id))

    print(f"The row number for source_id: {target_source_id}, member_id: {target_member_id}, experiment_id: {target_experiment_id} is {row_index}.")

# %%
#files2crunch_loaded_df.iloc[33]

# %%
# Define output folder

# Get today's date in the format "ddDecyyyy"
today_str = datetime.now().strftime("%d%b%Y")

# Define the output folder path
# output_folder = f'../data/out/out_{today_str}e_cmip6_p'
# output_folder = f'../data/out/out_{today_str}_highres_ERA5'
# output_folder = f'data/site_eth/out_{today_str}_151pm_ETHFOG_unvetted'
# output_folder = f'data/site_eth/out_ETHFOG_10June2025_8am_unvetted'
# output_folder = f'data/site_eth/out_ETHFOG_22Nov2025_4pm_unvetted'
# output_folder = f'data/site_eth/out_{today_str}ERA5_highres_unvetted'
# ERA5 processing, 19 Feb 2026
output_folder = 'data/site_eth/out_ERA5_19Feb2026_1degree_vetted'
if crunchtimedomain == 'daily':
    output_folder = f'../data/out/outDaily_{today_str}c'

# Check if the output folder exists, and create it if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")
else:
    print(f"Directory already exists: {output_folder}")

# %% [markdown]
# ## Potentially crunch observations instead... 

# %%
# specify whether you rather want to run the special case of crunching ERA5 observations: 

# crunch_observations_instead = False
# ERA5 processing, 19 Feb 2026
crunch_observations_instead = True

# obs_data_folder = '../data/processed/processedDaily_ERA5_XXMay24a'
# obs_data_folder = '../data/processed/processed_ERA5_29Aug24a'
# obs_data_folder = '../data/processed/processed_ERA5_01Sep2024'
# obs_data_folder = '../data/processed/ERA5_CMIP6_format_02Jun20250144_v3'
# obs_data_folder = 'data/site_eth/projects/cmipcruncher_firefly_data/processed/ERA5_CMIP6_format_02Jun20250144_v3'
# obs_data_folder = '../data/processed/processed_ERA5_15Oct2024'
# ERA5 processing, 19 Feb 2026
obs_data_folder = 'data/site_eth/projects/cmipcruncher_firefly/data/processed/ERA5_CMIP6_format_19Feb2026_v1'
obsflag = 'ERA5'

# obs_data_folder = '../data/processed/processedDaily_21CR_XXMay24b'
# obs_data_folder = '../data/processed/processed_21CR_20Aug24'
# obsflag = '20CR'



# %% [markdown]
# ### check the loaded file name tables. 

# %%
print(filter_rows_files2crunch)

# Depending on the setting for filter_rows_files2crunch, filter the rows (or not). 
files2crunch_df = filter_files2crunch_df(files2crunch_loaded_df, filter_rows_files2crunch)

# %%
files2crunch_df['tas'][0]


# %%
if manage_existing_output_files == 'skipcalculation':
    to_skip = []
    for (source_id, experiment_id, member_id) in files2crunch_df.index:
        prefix = f"DAT_{source_id}_{experiment_id}_{member_id}_"
        existing = [f for f in os.listdir(output_folder)
                    if fnmatch.fnmatch(f, prefix + '*.nc')]
        if existing:
            to_skip.append((source_id, experiment_id, member_id))
    if to_skip:
        print(f"Skipping {len(to_skip)} existing outputs for:")
        for t in to_skip:
            print("  ", t)
        files2crunch_df = files2crunch_df.drop(index=to_skip)


# %%

# %%
# collect the input arguments that are needed to compile the radiative forcing information. 

unique_experiment_ids = files2crunch_df.index.get_level_values('experiment_id').unique()
unique_experiment_ids

# %%
if crunch_observations_instead: # Example usage:
    # Assuming 'files2crunch_loaded_df' is your loaded DataFrame with CMIP6 data paths
    
    #data_dict = generate_era5_data_dict(files2crunch_loaded_df, era5_data_folder)
    #files2crunch_df = create_era5_data_frame(data_dict, era5_data_folder)    
    data_dict = generate_obs_data_dict(files2crunch_loaded_df, obs_data_folder,obsflag)
    files2crunch_df = create_obs_data_frame(data_dict, obs_data_folder,obsflag)
    # Step 1: Convert the Index to a list
    experiment_list = unique_experiment_ids.tolist()

    # Step 2: Append the new experiment IDs
    experiment_list.append('historical-ERA5')
    experiment_list.append('historical-20CR')

    # Step 3: Convert the list back to an Index
    unique_experiment_ids = pd.Index(experiment_list, name='experiment_id') 
    unique_experiment_ids

# %%
unique_experiment_ids

# %% [markdown]
# ## check whether there is anything to run. 
# in the case of experiments already ran before, it might be that your results data is already existent.. check the flag manage_existing_output_files, if you want to run your scenarios again, even though they exist in the output folder.. 

# %%
files2crunch_df

# %%
if files2crunch_df.empty:
    print("Nothing to do—all outputs already exist. Exiting notebook.")
    sys.exit(0)


# %%

# %% [markdown]
# ### get radiative forcing ready

# %%
def add_composite_scenario(df, new_scenario_name, scenarios_to_combine):
    """
    Adds a new composite scenario to the DataFrame by combining specified scenarios.
    Fills NaN values in the data of the first scenario with values from subsequent scenarios,
    but only for the variables present in the first scenario.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing radiative forcing data with a MultiIndex.
    new_scenario_name : str
        The name of the new composite scenario to be added.
    scenarios_to_combine : list of str
        The list of scenario names to combine in order.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with the new composite scenario added.
    """
    # Extract data for the first scenario
    combined_data = df.xs(scenarios_to_combine[0], level='Scenario', drop_level=False).copy()

    # Reset index to columns
    combined_data = combined_data.reset_index()

    # Define columns to set as index for merging (excluding ones we'll standardize)
    index_columns = [col for col in df.index.names if col not in ['Scenario', 'Model', 'Activity_Id', 'Mip_Era']]

    # Set index for combined_data
    combined_data.set_index(index_columns, inplace=True)

    # Loop through each additional scenario
    for scenario in scenarios_to_combine[1:]:
        scenario_data = df.xs(scenario, level='Scenario', drop_level=False).copy()
        scenario_data = scenario_data.reset_index()
        
        # Keep only variables present in the first scenario
        scenario_data = scenario_data[scenario_data['Variable'].isin(combined_data.index.get_level_values('Variable'))]

        # Set index for scenario_data
        scenario_data.set_index(index_columns, inplace=True)
        
        # Combine data using combine_first
        combined_data = combined_data.combine_first(scenario_data)

    # Reset index to modify 'Scenario', 'Model', etc.
    combined_data = combined_data.reset_index()

    # Set uniform 'Scenario', 'Model', 'Activity_Id', and 'Mip_Era' values
    combined_data['Scenario'] = new_scenario_name
    combined_data['Model'] = 'composite'
    combined_data['Activity_Id'] = 'composite_activity'
    combined_data['Mip_Era'] = 'composite_era'

    # Set index back to original structure
    combined_data = combined_data.set_index(df.index.names)

    # Remove any existing rows with the new_scenario_name to prevent duplicates
    df = df[~df.index.get_level_values('Scenario').isin([new_scenario_name])]

    # Append the combined data to the original DataFrame
    df = pd.concat([df, combined_data])

    return df



# %%
## RADIATIVE FORCING PREDICTORS
# load the radiative forcing file from which you want to complement the CMIP6 data
fn2load = 'rcmip-radiative-forcing-annual-means-v5-1-0.csv'
dir2load = '../data/raw/'

ExternalForcingFile_fullpath = os.path.join(dir2load,fn2load)

# Element 1: Load CSV file into DataFrame with MultiIndex
RadiativeForcing_RCMIP = pd.read_csv(ExternalForcingFile_fullpath, header=0, index_col=list(range(7)))

# Element 2: Complement DataFrame with CO2 scenarios
RadiativeForcing_RCMIP_complemented = add_co2_scenarios(RadiativeForcing_RCMIP)

# Element 3: components and pi_components for scenario variations. 
components, pi_components = define_radiative_forcing_components()

# Element 4: add other scenarios, like the combination of historical and ssp245 for the reanalysis data.. 

# Add 'historical-ERA5' scenario using the new function
RadiativeForcing_RCMIP_complemented = add_composite_scenario(
    RadiativeForcing_RCMIP_complemented,
    new_scenario_name='historical-ERA5',
    scenarios_to_combine=['historical', 'ssp245']
)

# Add 'historical-20CR' scenario using the new function
RadiativeForcing_RCMIP_complemented = add_composite_scenario(
    RadiativeForcing_RCMIP_complemented,
    new_scenario_name='historical-20CR',
    scenarios_to_combine=['historical', 'ssp245']
)

# print out all the scenarios that are available.. 
print(f'-----------------')
print(f'We have radiative forcing data for the following scenarios')
for scen in RadiativeForcing_RCMIP_complemented.index.get_level_values('Scenario').unique(): 
    print(scen)
    
    
# Define the scenario aliases when it comes to radiative forcing data. 
scenarioaliases = { 
        'ssp370-lowNTCF' :  ['ssp370-lowNTCF-aerchemmip'], 
        'hist': ['historical']} 


# %%
RadiativeForcing_RCMIP_complemented

# %%
# Assuming RadiativeForcing_RCMIP_complemented is your DataFrame
# filtered_df = RadiativeForcing_RCMIP_complemented.xs('historical-ERA5', level='Scenario', drop_level=False)
filtered_df = RadiativeForcing_RCMIP_complemented.xs('historical-ERA5', level='Scenario', drop_level=False)

# Print the filtered DataFrame
filtered_df

# %%
if False: ### to delete... just to test.. 


    # Select three scenarios for plotting and subset the data accordingly
    scenarios_to_plot = ["historical", "ssp245", "historical-ERA5"]
    #scenarios_to_plot = ["historical-ERA5"]
    selected_data = RadiativeForcing_RCMIP_complemented[
        RadiativeForcing_RCMIP_complemented.index.get_level_values("Scenario").isin(scenarios_to_plot)
    ]

    # Identify the unique variables to create a separate plot for each
    variables = selected_data.index.get_level_values("Variable").unique()

    # Define line styles for each scenario
    line_styles = {"historical": "-.", "ssp245": "--", "historical-ERA5": ":"}

    # Loop through each variable and plot data for selected scenarios
    for variable in variables:
        plt.figure(figsize=(10, 6))

        # Filter data for the specific variable
        variable_data = selected_data[selected_data.index.get_level_values("Variable") == variable]

         # Plot each scenario
        for scenario in scenarios_to_plot:
            scenario_data = variable_data[variable_data.index.get_level_values("Scenario") == scenario]

            # Convert column headers (years) to integers and transpose the data
            scenario_data.columns = scenario_data.columns.astype(int)
            scenario_data = scenario_data.loc[:, 1990:2040].T  # Filter years from 1990 to 2040

            # Drop extra levels from the index
            scenario_data.columns = scenario_data.columns.droplevel([0, 1, 2, 3, 4, 5])

            # Plot
            plt.plot(
                scenario_data.index,
                scenario_data.values,
                label=f"{scenario}",
                linestyle=line_styles[scenario]
            )

        # Add labels, legend, and title
        plt.xlabel("Year")
        plt.ylabel(variable_data.index.get_level_values("Unit").unique()[0])
        plt.title(f"{variable} Across Selected Scenarios")
        plt.legend()
        plt.show()


# %%
# unique_experiment_ids

# %%
## now create full radiative forcing dataset.. 

radiativeforcing_handles = create_radiativeforcing_handles(RadiativeForcing_RCMIP_complemented, unique_experiment_ids, components, pi_components, ExternalForcingFile_fullpath,scenarioaliases)
#radiativeforcing_handles

# %% [markdown]
# # Preprocessing Workflow:
#

# %%

# ############ STEP 1 ###########################################
# ############ INITIAL REGRIDDING ###############################

if use_paralleloptions: 
    regridded_data_handles = process_files_parallel(
        files2crunch_df,
        target_resolution,
        num_jobs=regrid_num_jobs,
        lon_convention=lon_convention,
    )
else: 
    regridded_data_handles = process_regridding_files(files2crunch_df, target_resolution, lon_convention=lon_convention)


# %%
# # with refacroting.. 
# ############ STEP 1 ###########################################
# ############ INITIAL REGRIDDING ###############################

# if use_paralleloptions:
#     regridded_data_handles = process_files_parallel(files2crunch_df, target_resolution, temporary_dir, num_jobs=20)
# else:
#     regridded_data_handles = process_regridding_files(files2crunch_df, target_resolution, temporary_dir)



# %%
############ STEP 2 ###########################################
############ COMBINE FILES: ENERGY FLUX, RTMT ETC ###################

#regridded_data_handles = process_combine_files(files2crunch_df, target_resolution, regridded_data_handles)
regridded_data_handles = process_combine_files(files2crunch_df, \
                        target_resolution, regridded_data_handles, use_parallel=use_paralleloptions,\
                                              crunch_observations_instead=crunch_observations_instead,\
                                              lon_convention=lon_convention)

#refactor attempt

# # STEP 2: Combine Files
# regridded_data_handles = process_combine_files(files2crunch_df, target_resolution, regridded_data_handles, \
#                                                use_parallel=use_paralleloptions, \
#                                                crunch_observations_instead=crunch_observations_instead, \
#                                                temporary_dir=temporary_dir)



# %%
############ STEP 3 ###########################################
############ STITCHING  #######################################

# now.. loop through the handles of all the regridded data and check whether any parents can be brought together with their childs. 

regridded_data_handles = stitch_parent_child_datasets(regridded_data_handles)


# %%
#use_paralleloptions = True

# %%
############ STEP 4 ###########################################
############ GLOBAL-MEAN + SMOOTHED ###########################


# Define the list of variables to process for global means
vars4globalmean = ['tas', 'ts', 'pr', 'rtmt', 'rsdt', 'rlut', 'rsut', 'rsnt', 'mtnlwrf', 'mtnswrf', 'mtdwswrf']
vars4globalmean = ['tas', 'ts', 'pr', 'rtmt', 'rsdt', 'rlut', 'rsut', 'rsnt', 'mtnlwrf', 'mtnswrf', 'mtdwswrf', 'tasmax', 'tasmin', 'clt', 'evspsbl', 'mrso', 'sfcwind', 'psl', 'huss', 'hurs']

# TasTsTasminTasmaxPrEvspsblMrsoSfcwindHursPslRsdsHussCltRsdtRsutRlutRtmt
# Modify your function to use the scattered data
#globalmean_data_handles, globalmean_smoothed_data_handles = \
#    process_global_means_parallel(client, scattered_data, vars4globalmean, period_years=21, use_parallel=use_paralleloptions)

globalmean_data_handles, globalmean_smoothed_data_handles = \
process_global_means_parallel(
    regridded_data_handles,
    vars4globalmean,
    period_years=21,
    use_parallel=use_paralleloptions,
    n_jobs=globalmean_num_jobs,
)

# %%
# ────────────────────────────────────────────────────────────────
# STEP 5 – CONSOLIDATE + WRITE,  **then free the memory**
# ────────────────────────────────────────────────────────────────


for (source_id, experiment_id, member_id), row in files2crunch_df.iterrows():
    print("--------------------------------------------------------")
    print("Now Processing", source_id, experiment_id, member_id)

    filter_for_modelscenmember = (source_id, experiment_id, member_id)

    # ---------- build the in-memory Dataset(s) ----------
    dataset, dataset_parent_child = consolidate_datasets_new(
        regridded_data_handles,
        globalmean_data_handles,
        globalmean_smoothed_data_handles,
        radiativeforcing_handles,
        filter_for_modelscenmember,
    )

    prefixword = "DAYDAT" if crunchtimedomain == "daily" else "DAT"
    dataset_prefix = f"{prefixword}_{source_id}_{experiment_id}_{member_id}"

    # ---------- write the “raw” file ----------
    if dataset.data_vars:
        outfile = write_compressed_netcdf(output_folder, dataset, dataset_prefix)
    else:
        print(f"⚠️  {dataset_prefix}.nc has no data variables – skipped")

    # ---------- write the parent+child file (if any) ----------
    if dataset_parent_child and dataset_parent_child.data_vars:
        pfx = f"{prefixword}PLUSPARENT_{source_id}_{experiment_id}_{member_id}"
        outfile = write_compressed_netcdf(output_folder, dataset_parent_child, pfx)

    # ---------- close the just-written Dataset objects ----------
    try:
        dataset.close()
    except Exception:
        pass
    try:
        dataset_parent_child.close()
    except Exception:
        pass

    # ---------- FREE re-gridded fields no longer needed ----------
    keys_to_drop = [
        k for k in regridded_data_handles.keys()
        if k[:3] == (source_id, experiment_id, member_id)
    ]
    for k in keys_to_drop:
        try:
            regridded_data_handles[k].close()   # drop HDF5 / Zarr handles
        except Exception:
            pass
        del regridded_data_handles[k]

    # optional: also drop the tiny global-mean arrays
    # for k in list(globalmean_data_handles):
    #     if k[:3] == (source_id, experiment_id, member_id):
    #         del globalmean_data_handles[k]; del globalmean_smoothed_data_handles[k]

    gc.collect()     # return the memory before next loop iteration

print("--------------------------------------------------------")
print("------------------- FINISHED ---------------------------")


# %%

# %%
