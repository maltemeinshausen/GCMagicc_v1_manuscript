import torch
import healpy as hp
import os
import numpy as np
import h5py
from torch.utils.data import Dataset

import torch.nn as nn
import math

# class HEALPixDataset5(Dataset):
#     # def __init__(self, file_path, variable_names, optional_variables=None, maxsample=None):
#     #     self.file_path = file_path
#     #     self.variable_names = variable_names
#     #     self.optional_variables = optional_variables if optional_variables is not None else []
#     #     self.data = {}
        
#     #     with h5py.File(file_path, 'r') as h5f:
#     #         first_var = self.variable_names[0] if self.variable_names else None
#     #         self.sample_indices = None
            
#     #         if first_var and first_var in h5f:
#     #             data_length = h5f[first_var].shape[0]
#     #             if maxsample is not None and maxsample < data_length:
#     #                 self.sample_indices = np.sort(np.random.choice(data_length, maxsample, replace=False))
#     #             else:
#     #                 self.sample_indices = np.arange(data_length)
            
#     #         for var in self.variable_names + self.optional_variables:
#     #             if var in h5f:
#     #                 self.data[var] = h5f[var]  # Store reference, avoid full load
#     #             else:
#     #                 self.data[var] = None  # Optional variable missing

#     #         self.num_samples = len(self.sample_indices)
            
#     def __init__(self, file_path, variable_names, optional_variables=None, maxsample=None):
#         """
#         Initializes the dataset by loading specified variables from a single HDF5 file into memory.

#         Args:
#             file_path (str): Path to the HDF5 file containing the dataset.
#             variable_names (list of str): List of required variable names to load from the file.
#             optional_variables (list of str, optional): List of optional variable names to load if they exist.
#         """
#         self.file_path = file_path
#         self.variable_names = variable_names
#         self.optional_variables = optional_variables if optional_variables is not None else []
            
#         self.data = {}
#         with h5py.File(file_path, 'r') as h5f:
#             first_var = self.variable_names[0] if self.variable_names else None
#             sample_indices = None
            
#             if first_var and first_var in h5f:
#                 data_length = h5f[first_var].shape[0]
#                 if maxsample is not None and maxsample < data_length:
#                     sample_indices = np.sort(np.random.choice(data_length, maxsample, replace=False))
#                 else:
#                     sample_indices = np.arange(data_length)
            
#             for var in self.variable_names:
#                 if var in h5f:
#                     self.data[var] = h5f[var][sample_indices] if sample_indices is not None else h5f[var]
#                 else:
#                     raise ValueError(f"Variable '{var}' not found in file {file_path}.")

#             for var in self.optional_variables:
#                 if var in h5f:
#                     self.data[var] = h5f[var][sample_indices] if sample_indices is not None else h5f[var]
#                 else:
#                     print(f"Optional variable '{var}' not found in file {file_path}. It will be set to None.")
#                     self.data[var] = None

#             self.num_samples = self.data[self.variable_names[0]].shape[0]

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
        
#         sample = {}
#         for var in self.variable_names:
#             sample[var] = torch.tensor(self.data[var][idx], dtype=torch.float32)
#             #sample[var] = torch.from_numpy(self.data[var][idx])
#         for var in self.optional_variables:
#             if self.data[var] is not None:
#                 #sample[var] = torch.from_numpy(self.data[var][idx])
#                 sample[var] = torch.tensor(self.data[var][idx], dtype=torch.float32)
#             else:
#                 sample[var] = None
#         return sample


class HEALPixDataset4sample(Dataset):
    def __init__(self, file_path, variable_names, optional_variables=None, 
                 # <span style="color:red">usefraction: float = 1.0</span>):
                 usefraction: float = 0.5):
        """
        Initializes the dataset by loading specified variables from a single HDF5 file into memory.

        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            variable_names (list of str): List of required variable names to load from the file.
            optional_variables (list of str, optional): List of optional variable names to load if they exist.
            <span style="color:red">usefraction (float): Fraction of rows to load (0.0–1.0). Same subset used for all variables.</span>
        """
        self.file_path = file_path
        self.variable_names = variable_names
        self.optional_variables = optional_variables if optional_variables is not None else []

        # Load data into memory
        self.data = {}
        with h5py.File(file_path, 'r') as h5f:
            
            N = h5f[self.variable_names[0]].shape[0]
            m = int(N * usefraction)
            rng = np.random.default_rng()
            idx = np.sort(rng.choice(N, size=m, replace=False)) if m < N else np.arange(N)

            for var in self.variable_names:
                if var in h5f:
                    self.data[var] = h5f[var][idx]
                else:
                    raise ValueError(f"Variable '{var}' not found in file {file_path}.")

            for var in self.optional_variables:
                if var in h5f:
                    self.data[var] = h5f[var][idx]
                else:
                    print(f"Optional variable '{var}' not found in file {file_path}. It will be set to None.")
                    self.data[var] = None

        self.num_samples = m

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {}
        for var in self.variable_names:
            sample[var] = torch.from_numpy(self.data[var][idx])
        for var in self.optional_variables:
            if self.data[var] is not None:
                sample[var] = torch.from_numpy(self.data[var][idx])
            else:
                sample[var] = None
        return sample
    
class HEALPixDataset4(Dataset):
    def __init__(self, file_path, variable_names, optional_variables=None):
        """
        Initializes the dataset by loading specified variables from a single HDF5 file into memory.

        Args:
            file_path (str): Path to the HDF5 file containing the dataset.
            variable_names (list of str): List of required variable names to load from the file.
            optional_variables (list of str, optional): List of optional variable names to load if they exist.
        """
        self.file_path = file_path
        self.variable_names = variable_names
        self.optional_variables = optional_variables if optional_variables is not None else []

        # Load data into memory
        self.data = {}
        with h5py.File(file_path, 'r') as h5f:
            for var in self.variable_names:
                if var in h5f:
                    self.data[var] = h5f[var][:]  # Load all data for this variable
                else:
                    raise ValueError(f"Variable '{var}' not found in file {file_path}.")

            for var in self.optional_variables:
                if var in h5f:
                    self.data[var] = h5f[var][:]  # Load data if it exists
                else:
                    print(f"Optional variable '{var}' not found in file {file_path}. It will be set to None.")
                    self.data[var] = None

        # Determine the number of samples from the first variable
        self.num_samples = self.data[self.variable_names[0]].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        
        sample = {}
        for var in self.variable_names:
            sample[var] = torch.from_numpy(self.data[var][idx])
        for var in self.optional_variables:
            if self.data[var] is not None:
                sample[var] = torch.from_numpy(self.data[var][idx])
            else:
                sample[var] = None
        return sample


class DownsampleWithNoise(nn.Module):
    def __init__(self, noise_level=0.0):
        super(DownsampleWithNoise, self).__init__()
        self.noise_level = noise_level

    def forward(self, y_high):
        
        # Infer nside_hi from y_high
        batch_size, n_features, npix_hi = y_high.shape
        nside_hi = hp.npix2nside(npix_hi)
        
        # Compute nside_lo as nside_hi divided by 2^k
        nside_lo = nside_hi // 2
        if nside_lo < 1:
            raise ValueError("Downsampling factor 2 results in invalid nside_lo < 1.")
        
        # Perform downsampling using healpix_avg_pooling
        y_low = healpix_avg_pooling(y_high, nside_hi, nside_lo)
        
        # Add Gaussian noise
        if self.noise_level>0.0:
            noise = self.noise_level * torch.randn_like(y_low, device=y_low.device, dtype=y_low.dtype)
            y_low = y_low + noise
        
        return y_low
    

class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()

    def forward(self, y_high):
        
        # Infer nside_hi from y_high
        _, _, npix_hi = y_high.shape
        nside_hi = hp.npix2nside(npix_hi)
        
        nside_lo = nside_hi // 2
        if nside_lo < 1:
            raise ValueError("Downsampling factor 2 results in invalid nside_lo < 1.")
        
        y_low = healpix_avg_pooling(y_high, nside_hi, nside_lo)
        
        return y_low
    
    
    
def healpix_avg_pooling(high_res_data, nside_hi, nside_lo, ordering='RING'):
    """
    Downsamples HEALPix data from higher resolution (nside_hi) to lower resolution (nside_lo)
    using average pooling. Supports data with arbitrary leading dimensions.
    
    Args:
        high_res_data (torch.Tensor): Input data of shape (..., npix_hi).
        nside_hi (int): High-resolution nside parameter.
        nside_lo (int): Low-resolution nside parameter.
        ordering (str): HEALPix ordering scheme ('RING' or 'NESTED').
    
    Returns:
        torch.Tensor: Downsampled data of shape (..., npix_lo).
    """
    # Validate ordering
    if ordering not in ['RING', 'NESTED']:
        raise ValueError("ordering must be 'RING' or 'NESTED'")
    
    npix_hi = hp.nside2npix(nside_hi)
    npix_lo = hp.nside2npix(nside_lo)
    
    # Reshape high_res_data to (..., npix_hi)
    input_shape = high_res_data.shape
    if input_shape[-1] != npix_hi:
        raise ValueError(f"The last dimension of high_res_data must be {npix_hi}, got {input_shape[-1]}")
    
    # Flatten leading dimensions
    high_res_data_flat = high_res_data.reshape(-1, npix_hi)
    
    # Compute the number of high-res pixels per low-res pixel
    ratio = (nside_hi // nside_lo) ** 2
    
    # Get parent pixels mapping from high-res to low-res
    if ordering == 'RING':
        # Convert high-res pixel indices from RING to NESTED
        pix_indices_hi = np.arange(npix_hi)
        pix_indices_hi_nest = hp.ring2nest(nside_hi, pix_indices_hi)
        # Compute corresponding low-res pixel indices in NESTED ordering
        pix_indices_lo_nest = pix_indices_hi_nest // ratio
        # Convert low-res pixel indices back to RING ordering
        pix_indices_lo = hp.nest2ring(nside_lo, pix_indices_lo_nest)
    else:
        pix_indices_hi = np.arange(npix_hi)
        pix_indices_lo = pix_indices_hi // ratio
    
    # Create a mapping from low-res pixel to high-res pixels
    mapping = torch.zeros((npix_lo, npix_hi), device=high_res_data.device)
    mapping[pix_indices_lo, torch.arange(npix_hi)] = 1.0
    
    # Perform average pooling
    pooled_data = torch.matmul(high_res_data_flat, mapping.T) / ratio
    
    # Reshape back to original leading dimensions with npix_lo
    output_shape = input_shape[:-1] + (npix_lo,)
    low_res_data = pooled_data.reshape(output_shape)
    
    return low_res_data


def healpix_to_regular_grid(healpix_data, nlat, ordering='RING'):
    """
    Convert HEALPix data to a regular grid of equal latitude and longitude.

    Args:
        healpix_data (torch.Tensor): HEALPix data of shape (..., npix)
            where npix = 12 * nside^2.
        nlat (int): Number of latitude points in the regular grid.
            The number of longitude points will be nlon = 2 * nlat.
        ordering (str): 'RING' or 'NESTED' ordering of the HEALPix data.

    Returns:
        regular_grid_data (torch.Tensor): Data on a regular grid of shape (..., nlat, nlon)
    """
    # Check ordering
    if ordering not in ['RING', 'NESTED']:
        raise ValueError("ordering must be 'RING' or 'NESTED'")

    # Get the number of pixels in the HEALPix data
    npix = healpix_data.shape[-1]

    # Determine nside from npix
    nside = hp.npix2nside(npix)

    # Generate latitude and longitude arrays
    nlon = 2 * nlat  # Number of longitude points
    lat = np.linspace(-89.5, 89.5, nlat, endpoint=True)
    lon = np.linspace(-179.5, 179.5, nlon, endpoint=True)

    # Create meshgrid of longitude and latitude
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Convert to theta (colatitude in radians) and phi (longitude in radians)
    theta = np.radians(90.0 - lat_grid.flatten())  # colatitude
    phi = np.radians(lon_grid.flatten())  # longitude

    # Prepare healpix_data as numpy array
    healpix_data_np = healpix_data.detach().cpu().numpy()  # Shape: (..., npix)
    input_shape = healpix_data_np.shape
    leading_shape = input_shape[:-1]  # Shape of leading dimensions
    npix = input_shape[-1]

    # Reshape healpix_data to (-1, npix)
    healpix_data_flat = healpix_data_np.reshape(-1, npix)  # Shape: (batch_size, npix)

    # Number of grid points
    n_points = nlat * nlon

    # Initialize regular grid data
    regular_grid_data_flat = np.zeros((healpix_data_flat.shape[0], n_points), dtype=healpix_data_np.dtype)

    # Loop over all maps in the batch
    for idx in range(healpix_data_flat.shape[0]):
        # Get HEALPix map for this index
        healpix_map = healpix_data_flat[idx]

        # Interpolate values at the grid points
        values = hp.get_interp_val(healpix_map, theta, phi, nest=(ordering == 'NESTED'))

        # Store the interpolated values
        regular_grid_data_flat[idx] = values

    # Reshape back to original leading dimensions with grid dimensions
    regular_grid_shape = leading_shape + (nlat, nlon)
    regular_grid_data = regular_grid_data_flat.reshape(regular_grid_shape)

    # Convert back to PyTorch tensor
    regular_grid_data_torch = torch.from_numpy(regular_grid_data)

    return regular_grid_data_torch


def regular_grid_to_healpix(regular_grid_data, nside, ordering='RING'):
    """
    Convert data from a regular latitude-longitude grid to HEALPix format.

    Args:
        regular_grid_data (torch.Tensor): Data on a regular grid of shape (..., nlat, nlon)
            where nlon = 2 * nlat.
        nside (int): Desired HEALPix resolution parameter.
        ordering (str): 'RING' or 'NESTED' ordering for the output HEALPix data.

    Returns:
        healpix_data (torch.Tensor): HEALPix data of shape (..., npix),
            where npix = 12 * nside^2.
    """
    # Check ordering
    if ordering not in ['RING', 'NESTED']:
        raise ValueError("ordering must be 'RING' or 'NESTED'")

    # Get the grid dimensions
    nlat = regular_grid_data.shape[-2]
    nlon = regular_grid_data.shape[-1]

    # Generate latitude and longitude arrays
    lat = np.linspace(-89.5, 89.5, nlat, endpoint=True)
    #lon = np.linspace(-179.5, 179.5, nlon, endpoint=True)
    ####### 180
    lon = np.linspace(0.5, 359.5, nlon, endpoint=True)

    # Prepare regular_grid_data as numpy array
    regular_grid_data_np = regular_grid_data.detach().cpu().numpy()  # Shape: (..., nlat, nlon)
    input_shape = regular_grid_data_np.shape
    leading_shape = input_shape[:-2]  # Shape of leading dimensions

    # Reshape regular_grid_data to (-1, nlat, nlon)
    regular_grid_data_flat = regular_grid_data_np.reshape(-1, nlat, nlon)  # Shape: (batch_size, nlat, nlon)

    # Number of HEALPix pixels
    npix = hp.nside2npix(nside)

    # Initialize healpix_data
    healpix_data_flat = np.zeros((regular_grid_data_flat.shape[0], npix), dtype=regular_grid_data_np.dtype)

    # Get the pixel centers in theta and phi
    pix_indices = np.arange(npix)
    theta_pix, phi_pix = hp.pix2ang(nside, pix_indices, nest=(ordering == 'NESTED'))
    phi_pix = (phi_pix + np.pi) % (2*np.pi) ####### 180 - np.pi

    #phi_pix = (phi_pix % (2*np.pi))
    #phi_pix[phi_pix == 2*np.pi] = 0.0

    # Convert colatitude to latitude in radians
    lat_pix = 0.5 * np.pi - theta_pix  # Latitude in radians

    # Prepare interpolator
    from scipy.interpolate import RegularGridInterpolator

    # Loop over all maps in the batch
    for idx in range(regular_grid_data_flat.shape[0]):
        # Extract the data for this map
        data_map = regular_grid_data_flat[idx]  # Shape: (nlat, nlon)

        # Define the interpolator function for this map
        interpolator = RegularGridInterpolator(
            (np.radians(lat), np.radians(lon)),
            data_map,
            method='linear',
            bounds_error=False,
            fill_value=100.0
        )

        lat_min, lat_max = lat[0], lat[-1]
        lon_min, lon_max = lon[0], lon[-1]
        
        lat_pix = np.clip(lat_pix,  lat_min * np.pi/180 + 0.0001, lat_max * np.pi/180 - 0.0001)
        phi_pix = np.clip(phi_pix,  lon_min * np.pi/180 + 0.0001, lon_max * np.pi/180 - 0.0001)

        # Interpolate the data
        healpix_values = interpolator((lat_pix, phi_pix))

        # Handle NaN values (e.g., at the poles)
        nan_indices = np.isnan(healpix_values)
        if np.any(nan_indices):
            healpix_values[nan_indices] = -100.0  # Or use another appropriate value

        # Store the interpolated values
        healpix_data_flat[idx] = healpix_values

    # Reshape back to original leading dimensions with npix
    healpix_shape = leading_shape + (npix,)
    healpix_data = healpix_data_flat.reshape(healpix_shape)

    # Convert back to PyTorch tensor
    healpix_data_torch = torch.from_numpy(healpix_data)

    return healpix_data_torch

def manual_interp(x, xp, fp, left=None, right=None):
    """
    Manually interpolate x given xp (sorted) and fp (function values at xp),
    replicating the behavior of torch.interp.

    Args:
        x (Tensor): The x-coordinates at which to evaluate the interpolated values.
        xp (Tensor): The x-coordinates of the data points, must be sorted in ascending order.
        fp (Tensor): The y-coordinates of the data points.
        left (float, optional): Value to return for x < xp[0]. If None, defaults to fp[0].
        right (float, optional): Value to return for x > xp[-1]. If None, defaults to fp[-1].

    Returns:
        Tensor: Interpolated values, same shape as x.
    """

    # By default, match NumPy/torch.interp behavior for out-of-bounds values.
    if left is None:
        left = fp[0].item()  # If xp is not empty, defaults to fp[0].
    if right is None:
        right = fp[-1].item()  # If xp is not empty, defaults to fp[-1].

    # Find indices where x would be inserted to keep xp sorted.
    # searchsorted returns an index in [0, len(xp)].
    idx = torch.searchsorted(xp, x, right=False)

    # We'll clamp idx to be within [1, len(xp)-1] for valid interpolation:
    #   idx == 0  -> out of range (left side)
    #   idx == len(xp) -> out of range (right side)
    idx_left = (idx - 1).clamp(min=0, max=len(xp) - 2)
    idx_right = idx_left + 1

    # Gather the x-coordinates for the interpolation boundaries
    xp_left = xp[idx_left]
    xp_right = xp[idx_right]

    # Gather the function values at those boundaries
    fp_left = fp[idx_left]
    fp_right = fp[idx_right]

    # Compute the weights for linear interpolation
    denom = (xp_right - xp_left)
    # To avoid division-by-zero (in degenerate cases where xp_right == xp_left),
    # we can mask or clamp denom. For typical sorted xp, denom > 0.
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    weight = (x - xp_left) / denom
    # Linear interpolation
    out = fp_left + weight * (fp_right - fp_left)

    # Handle out-of-bounds (left side: idx=0, right side: idx=len(xp))
    out = torch.where(idx == 0, torch.tensor(left, dtype=out.dtype, device=out.device), out)
    out = torch.where(idx == len(xp), torch.tensor(right, dtype=out.dtype, device=out.device), out)

    return out


def healpix_to_regular_grid(healpix_data, nlat=180, ordering='RING'):
    """
    Convert HEALPix data to a regular grid of equal latitude and longitude.

    Args:
        healpix_data (torch.Tensor): HEALPix data of shape (batch_size, n_features, npix)
            where npix = 12 * nside^2.
        nlat (int): Number of latitude points in the regular grid.
            The number of longitude points will be nlon = 2 * nlat.
        ordering (str): 'RING' or 'NESTED' ordering of the HEALPix data.

    Returns:
        regular_grid_data (torch.Tensor): Data on a regular grid of shape (batch_size, n_features, nlat, nlon)
    """
    # Check ordering
    if ordering not in ['RING', 'NESTED']:
        raise ValueError("ordering must be 'RING' or 'NESTED'")

    # Get the number of pixels in the HEALPix data
    npix = healpix_data.shape[-1]

    # Determine nside from npix
    nside = hp.npix2nside(npix)

    # Generate latitude and longitude arrays
    nlon = 2 * nlat  # Number of longitude points
    lat = np.linspace(-89.5, 89.5, nlat, endpoint=True)
    lon = np.linspace(-179.5, 179.5, nlon, endpoint=True)

    # Create meshgrid of longitude and latitude
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Convert to theta (colatitude in radians) and phi (longitude in radians)
    theta = np.radians(90.0 - lat_grid.flatten())  # colatitude
    phi = np.radians(lon_grid.flatten())  # longitude

    # Prepare healpix_data as numpy array
    healpix_data_np = healpix_data.detach().cpu().numpy()  # Shape: (batch_size, n_features, npix)
    batch_size, n_features, npix = healpix_data_np.shape

    # Number of grid points
    n_points = nlat * nlon

    # Initialize regular grid data
    regular_grid_data_flat = np.zeros((batch_size, n_features, n_points), dtype=healpix_data_np.dtype)

    # Loop over all maps in the batch
    for idx in range(batch_size):
        for feat in range(n_features):
            # Get HEALPix map for this index and feature
            healpix_map = healpix_data_np[idx, feat, :]
            # Interpolate values at the grid points
            values = hp.get_interp_val(healpix_map, theta, phi, nest=(ordering == 'NESTED'))
            # Store the interpolated values
            regular_grid_data_flat[idx, feat, :] = values

    # Reshape back to (batch_size, n_features, nlat, nlon)
    regular_grid_data = regular_grid_data_flat.reshape(batch_size, n_features, nlat, nlon)

    # Convert back to PyTorch tensor
    regular_grid_data_torch = torch.from_numpy(regular_grid_data)

    return regular_grid_data_torch

from scipy.spatial import cKDTree

def get_neighbor_indices(nside, num_neighbors=32):
    npix = hp.nside2npix(nside)
    # Compute the unit vector (x,y,z) for each pixel.
    vecs = np.vstack(hp.pix2vec(nside, np.arange(npix))).T  # shape (npix, 3)
    # Build a KDTree for the pixel centers.
    tree = cKDTree(vecs)
    # Query for the (num_neighbors+1) nearest neighbors (the first neighbor is the pixel itself).
    dists, indices = tree.query(vecs, k=num_neighbors+1)
    return torch.tensor(indices, dtype=torch.long)
