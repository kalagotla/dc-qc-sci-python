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
from pathlib import Path
import io
import contextlib

import numpy as np
from huggingface_hub import snapshot_download
from tqdm import tqdm


def download_cylinder_les_data(base_dir: str = "data") -> str:
    """
    Download the LES cylinder dataset used for the POD / animation demo.

    This fetches the dataset from Hugging Face
    (repo_id=\"kalagotla/cylinder_les\", repo_type=\"dataset\") into the given
    base directory. The final layout will be:

        base_dir / \"cylinder_les\" / *.x
        base_dir / \"cylinder_les\" / *.q

    Parameters
    ----------
    base_dir : str, optional
        Directory in which to place the downloaded dataset. Defaults to \"data\".

    Returns
    -------
    str
        Path to the dataset directory (e.g. \"data/cylinder_les\").
    """
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    snapshot_download(
        repo_id="kalagotla/cylinder_les",
        repo_type="dataset",
        local_dir=str(base_path) + "/cylinder",
        local_dir_use_symlinks=False,
    )

    dataset_dir = base_path / "cylinder_les"
    return str(dataset_dir)


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


def stack_midplane_u_over_time(grid, example_flow_path: str, *, block_idx: int = 0, data_type: str = "f4"):
    """
    Build a single NumPy array U with shape (t, x, y) by stacking the mid-plane
    u-velocity from multiple flow snapshots in the same directory.

    This uses:
    - list_flow_files(example_flow_path) to find snapshots (sorted by filename)
    - prepare_midplane_u(grid, flow) to compute per-snapshot mid-plane u

    Parameters
    ----------
    grid
        Grid object (GridIO) with loaded grid data.
    example_flow_path : str
        Path to an example flow snapshot (e.g., ".../sol-0000010.q"). All files
        with the same extension in the same directory will be included.
    block_idx : int, optional (keyword-only)
        Which block to extract from prepare_midplane_u(...). Default is 0.
    data_type : str, optional (keyword-only)
        Data type passed to FlowIO.read_flow(). Default is "f4".

    Returns
    -------
    np.ndarray
        Stacked u-velocity array with shape (t, x, y).

    Raises
    ------
    ImportError
        If lptlib is not available.
    FileNotFoundError
        If no flow files are found.
    ValueError
        If the per-snapshot U shape changes across timesteps.
    IndexError
        If block_idx is not valid for the returned blocks list.
    """
    try:
        from lptlib.io.plot3dio import FlowIO
    except ImportError:
        raise ImportError(
            "lptlib is required for stack_midplane_u_over_time(). "
            "Install with: pip install lptlib"
        )

    flow_files = list_flow_files(example_flow_path)
    if len(flow_files) == 0:
        raise FileNotFoundError(
            f"No flow files found next to: {example_flow_path}\n"
            f"Expected something like: {os.path.join(os.path.dirname(example_flow_path), 'sol-*.q')}"
        )

    U_list = []
    target_shape = None

    for i, flow_path in enumerate(
        tqdm(flow_files, desc="Reading flow data", unit="file")
    ):
        flow = FlowIO(flow_path)

        # Suppress verbose lptlib output like "flow data reading is successful"
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
            flow.read_flow(data_type=data_type)

        blocks = prepare_midplane_u(grid, flow)
        _, _, U_i = blocks[block_idx]

        if i == 0:
            target_shape = U_i.shape
        elif U_i.shape != target_shape:
            raise ValueError(
                "Mid-plane U shape mismatch across timesteps. "
                f"Expected {target_shape}, got {U_i.shape} for file: {flow_path}"
            )

        U_list.append(U_i)

    return np.stack(U_list, axis=0)
