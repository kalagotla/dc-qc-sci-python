import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sys

# Ensure project src is importable when running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from lptlib.io.plot3dio import GridIO, FlowIO


def read_grid(grid_path):
    grid = GridIO(grid_path)
    grid.read_grid(data_type='f4')
    return grid


def list_flow_files(example_flow_path):
    base_dir = os.path.dirname(example_flow_path)
    ext = os.path.splitext(example_flow_path)[1]
    files = sorted(glob.glob(os.path.join(base_dir, f'*{ext}')))
    return files


def compute_u_limits(flow_files):
    u_min = np.inf
    u_max = -np.inf
    for f in flow_files:
        flow = FlowIO(f)
        flow.read_flow(data_type='f4')
        # q shape: (ni, nj, nk, 5, nb); indices: 0=rho, 1=rho*u
        rho = flow.q[..., 0, :]
        rhou = flow.q[..., 1, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.where(rho != 0, rhou / rho, 0.0)
        u_min = min(u_min, np.nanmin(u))
        u_max = max(u_max, np.nanmax(u))
    return float(u_min), float(u_max)


def prepare_block_views(grid, flow):
    blocks = []
    for b in range(flow.nb):
        ni, nj, nk = flow.ni[b], flow.nj[b], flow.nk[b]
        # Take mid-plane in k if nk > 1
        k_idx = nk // 2
        X = grid.grd[:ni, :nj, k_idx, 0, b]
        Y = grid.grd[:ni, :nj, k_idx, 1, b]
        rho = flow.q[:ni, :nj, k_idx, 0, b]
        rhou = flow.q[:ni, :nj, k_idx, 1, b]
        with np.errstate(divide='ignore', invalid='ignore'):
            U = np.where(rho != 0, rhou / rho, 0.0)
        blocks.append((X, Y, U))
    return blocks


def animate_u_contours(grid_path='cylinder.sp.x', example_flow_path='sol-0000010.q',
                       interval_ms=80, levels=40, save_path=None):
    # Resolve paths: if absolute or exists relative to cwd, use as-is; otherwise resolve relative to module
    if not os.path.isabs(grid_path) and not os.path.exists(grid_path):
        grid_path = os.path.join(os.path.dirname(__file__), grid_path)
    if not os.path.isabs(example_flow_path) and not os.path.exists(example_flow_path):
        example_flow_path = os.path.join(os.path.dirname(__file__), example_flow_path)

    grid = read_grid(grid_path)
    flow_files = list_flow_files(example_flow_path)
    if len(flow_files) == 0:
        raise FileNotFoundError('No flow files found for animation.')

    umin, umax = compute_u_limits(flow_files)

    # Initialize first frame
    flow0 = FlowIO(flow_files[0])
    flow0.read_flow(data_type='f4')
    blocks0 = prepare_block_views(grid, flow0)

    fig, ax = plt.subplots(figsize=(8, 3))
    contour_sets = []

    # Plot initial contours per block
    for (X, Y, U) in blocks0:
        cs = ax.contourf(X, Y, U, levels=levels, vmin=umin, vmax=umax, cmap='RdBu_r')
        contour_sets.append(cs)

    cbar = fig.colorbar(contour_sets[-1], ax=ax)
    cbar.set_label('u-velocity')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 2)
    ax.set_title(os.path.basename(flow_files[0]))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()

    def init():
        # Return initial collections
        all_collections = []
        for cs in contour_sets:
            all_collections.extend(cs.collections)
        return all_collections

    def update(idx):
        nonlocal contour_sets
        # Remove previous contour collections
        for cs in contour_sets:
            for coll in cs.collections:
                coll.remove()
        contour_sets = []

        # Load and process new frame
        flow = FlowIO(flow_files[idx])
        flow.read_flow(data_type='f4')
        blocks = prepare_block_views(grid, flow)
        all_collections = []
        for (X, Y, U) in blocks:
            cs = ax.contourf(X, Y, U, levels=levels, vmin=umin, vmax=umax, cmap='RdBu_r')
            contour_sets.append(cs)
            all_collections.extend(cs.collections)
        
        # Update title
        ax.set_title(os.path.basename(flow_files[idx]))
        
        # Return all collections so matplotlib knows what to redraw
        return all_collections

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=len(flow_files), interval=interval_ms, blit=False, repeat=True)

    if save_path is not None:
        save_path = os.path.join(os.path.dirname(__file__), save_path)
        anim.save(save_path, dpi=150, writer='ffmpeg')

    plt.show()
    return anim


if __name__ == '__main__':
    # Adjust parameters as needed; set save_path to save mp4
    animate_u_contours(save_path=None)


