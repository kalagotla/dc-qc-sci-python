"""
helpers.py - Helper functions for loading and processing CFD data

This module provides utility functions for working with lptlib data:
- Reading grid files
- Listing flow files
- Extracting mid-plane velocity data

Usage:
    from modules.helpers import read_grid, list_flow_files, prepare_midplane_u
    
    grid = read_grid("path/to/grid.sp.x")
    flow_files = list_flow_files("path/to/sol-0000010.q")
    blocks = prepare_midplane_u(grid, flow)
"""

import os
import glob
import numpy as np


def read_grid(grid_path: str):
    """
    Read a grid file using lptlib's GridIO.
    
    Parameters
    ----------
    grid_path : str
        Path to the grid file (.sp.x format)
        
    Returns
    -------
    GridIO
        Grid object with loaded grid data
        
    Raises
    ------
    ImportError
        If lptlib is not available
    FileNotFoundError
        If grid_path does not exist
    """
    try:
        from lptlib import GridIO
    except ImportError:
        raise ImportError("lptlib is required for read_grid(). Install with: pip install lptlib")
    
    grid = GridIO(grid_path)
    grid.read_grid(data_type="f4")
    return grid


def list_flow_files(example_flow_path: str):
    """
    List all flow files in the same directory as the example file.
    
    Parameters
    ----------
    example_flow_path : str
        Path to an example flow file (e.g., "sol-0000010.q")
        
    Returns
    -------
    list of str
        Sorted list of flow file paths with the same extension
        
    Examples
    --------
    >>> files = list_flow_files("data/cylinder/sol-0000010.q")
    >>> print(files[0])
    data/cylinder/sol-0000010.q
    """
    base_dir = os.path.dirname(example_flow_path)
    ext = os.path.splitext(example_flow_path)[1]
    files = sorted(glob.glob(os.path.join(base_dir, f"*{ext}")))
    return files


def prepare_midplane_u(grid, flow):
    """
    Extract mid-plane u-velocity data from flow field.
    
    Returns lists of (X, Y, U) tuples for each block at a mid-span plane.
    - X, Y are 2D coordinate arrays
    - U is the computed streamwise velocity u = (rho*u)/rho
    
    Parameters
    ----------
    grid : GridIO
        Grid object from read_grid()
    flow : FlowIO
        Flow object with read_flow() already called
        
    Returns
    -------
    list of tuple
        List of (X, Y, U) tuples, one per block, where:
        - X : ndarray, 2D array of x-coordinates
        - Y : ndarray, 2D array of y-coordinates  
        - U : ndarray, 2D array of u-velocity values
        
    Examples
    --------
    >>> grid = read_grid("grid.sp.x")
    >>> flow = FlowIO("sol-0000010.q")
    >>> flow.read_flow(data_type="f4")
    >>> blocks = prepare_midplane_u(grid, flow)
    >>> X, Y, U = blocks[0]
    """
    blocks = []
    for b in range(flow.nb):
        ni, nj, nk = flow.ni[b], flow.nj[b], flow.nk[b]
        k_idx = nk // 2  # take a mid-plane in the spanwise direction
        X = grid.grd[:ni, :nj, k_idx, 0, b]
        Y = grid.grd[:ni, :nj, k_idx, 1, b]
        rho = flow.q[:ni, :nj, k_idx, 0, b]
        rhou = flow.q[:ni, :nj, k_idx, 1, b]
        with np.errstate(divide="ignore", invalid="ignore"):
            U = np.where(rho != 0, rhou / rho, 0.0)
        blocks.append((X, Y, U))
    return blocks
