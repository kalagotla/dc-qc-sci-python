import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button, Slider
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

    # Create figure with space for controls
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    
    # Initialize first frame
    flow0 = FlowIO(flow_files[0])
    flow0.read_flow(data_type='f4')
    blocks0 = prepare_block_views(grid, flow0)

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
    
    # Create control panel
    ax_slider = plt.subplot2grid((4, 1), (3, 0))
    slider = Slider(ax_slider, 'Timestep', 0, len(flow_files)-1, 
                    valinit=0, valfmt='%d', valstep=1)
    
    # Create buttons
    ax_prev = plt.axes([0.1, 0.02, 0.1, 0.04])
    ax_play = plt.axes([0.25, 0.02, 0.1, 0.04])
    ax_pause = plt.axes([0.4, 0.02, 0.1, 0.04])
    ax_next = plt.axes([0.55, 0.02, 0.1, 0.04])
    
    btn_prev = Button(ax_prev, '◀ Prev')
    btn_play = Button(ax_play, '▶ Play')
    btn_pause = Button(ax_pause, '⏸ Pause')
    btn_next = Button(ax_next, 'Next ▶')
    
    # Animation state
    is_playing = [False]
    current_frame = [0]
    slider_updating = [False]  # Flag to prevent recursive updates
    
    def update_frame(idx):
        """Update the plot with data from frame idx"""
        nonlocal contour_sets
        
        # Clear existing contours by removing collections
        for cs in contour_sets:
            for coll in cs.collections:
                try:
                    coll.remove()
                except:
                    pass
        contour_sets = []

        # Load and process new frame
        flow = FlowIO(flow_files[idx])
        flow.read_flow(data_type='f4')
        blocks = prepare_block_views(grid, flow)
        
        # Redraw contours
        for (X, Y, U) in blocks:
            cs = ax.contourf(X, Y, U, levels=levels, vmin=umin, vmax=umax, cmap='RdBu_r')
            contour_sets.append(cs)
        
        # Update title - ensure it's actually set
        title_text = f'Frame {idx+1}/{len(flow_files)}: {os.path.basename(flow_files[idx])}'
        ax.set_title(title_text)
        ax.title.set_text(title_text)  # Also set directly
        
        # Ensure axis properties are maintained
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Update colorbar to reflect new data
        if len(contour_sets) > 0:
            try:
                cbar.mappable = contour_sets[-1]
                cbar.update_normal(contour_sets[-1])
            except:
                try:
                    cbar.update_bruteforce(contour_sets[-1])
                except:
                    pass
        
        # Update slider without triggering callback
        slider_updating[0] = True
        try:
            slider.set_val(idx)
        except:
            pass
        slider_updating[0] = False
        
        # Force redraw - use flush_events to ensure update
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except:
            try:
                fig.canvas.draw_idle()
                plt.pause(0.01)
            except:
                plt.pause(0.01)
    
    def animate_frame(frame_idx):
        """Animation callback function - called by FuncAnimation"""
        if is_playing[0]:
            # Use frame_idx modulo to cycle through frames
            frame_to_show = frame_idx % len(flow_files)
            current_frame[0] = frame_to_show
            update_frame(frame_to_show)
        return []
    
    # Create animation with fixed number of frames - it will repeat
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(flow_files),
                                   interval=interval_ms, repeat=True, blit=False)
    
    # Start paused - stop the animation immediately
    try:
        if hasattr(anim, 'event_source') and anim.event_source:
            anim.event_source.stop()
    except:
        pass
    is_playing[0] = False
    
    # Now define handlers that can reference anim
    def on_slider_change(val):
        """Handle slider changes"""
        if not slider_updating[0]:
            idx = int(val)
            if not is_playing[0]:
                current_frame[0] = idx
                update_frame(idx)
    
    def on_prev(event):
        """Go to previous frame"""
        is_playing[0] = False
        if hasattr(anim, 'event_source') and anim.event_source:
            anim.event_source.stop()
        current_frame[0] = max(0, current_frame[0] - 1)
        update_frame(current_frame[0])
    
    def on_next(event):
        """Go to next frame"""
        is_playing[0] = False
        if hasattr(anim, 'event_source') and anim.event_source:
            anim.event_source.stop()
        current_frame[0] = min(len(flow_files) - 1, current_frame[0] + 1)
        update_frame(current_frame[0])
    
    def on_play(event):
        """Start animation"""
        is_playing[0] = True
        if hasattr(anim, 'event_source') and anim.event_source:
            anim.event_source.start()
        else:
            # Try alternative start method
            try:
                anim._start()
            except:
                pass
    
    def on_pause(event):
        """Pause animation"""
        is_playing[0] = False
        if hasattr(anim, 'event_source') and anim.event_source:
            anim.event_source.stop()
    
    # Connect callbacks
    slider.on_changed(on_slider_change)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_play.on_clicked(on_play)
    btn_pause.on_clicked(on_pause)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = os.path.join(os.path.dirname(__file__), save_path)
        anim.save(save_path, dpi=150, writer='ffmpeg')

    plt.show()
    return anim


if __name__ == '__main__':
    # Adjust parameters as needed; set save_path to save mp4
    animate_u_contours(save_path=None)


