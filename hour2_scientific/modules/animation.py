import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
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
    
    # Create slider for frame navigation
    ax_slider = fig.add_axes([0.1, 0.08, 0.8, 0.03])
    slider = Slider(ax_slider, 'Timestep', 0, len(flow_files)-1, 
                    valinit=0, valfmt='%d', valstep=1)
    
    # Create status text area to show controls
    status_text = fig.text(0.1, 0.12, 
                          'Controls: Space=Play/Pause, Left/Right=Step, Slider=Jump to frame',
                          fontsize=9, verticalalignment='bottom')
    
    # Animation state
    is_playing = [False]
    current_frame = [0]
    slider_updating = [False]  # Flag to prevent recursive updates
    
    def update_frame(idx):
        """Update the plot with data from frame idx"""
        nonlocal contour_sets
        
        # Clear existing contours by removing all collections from axes
        # This works for both old and new matplotlib versions
        while len(ax.collections) > 0:
            ax.collections[0].remove()
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
    
    # Frame counter for animation sequence
    anim_frame_count = [0]
    
    def animate_frame(frame_idx):
        """Animation callback function"""
        if is_playing[0]:
            # Move to next frame
            current_frame[0] = (current_frame[0] + 1) % len(flow_files)
            update_frame(current_frame[0])
            # Update status text
            try:
                status_text.set_text('Controls: Space=Play/Pause, Left/Right=Step, Slider=Jump | Status: PLAYING')
            except:
                pass
        return []
    
    # Create animation with many frames so it can run continuously
    anim = animation.FuncAnimation(fig, animate_frame, frames=range(10000),
                                   interval=interval_ms, repeat=True, blit=False, cache_frame_data=False)
    
    # Start paused
    is_playing[0] = False
    if hasattr(anim, 'event_source') and anim.event_source:
        anim.event_source.stop()
    
    # Now define handlers that can reference anim
    def on_slider_change(val):
        """Handle slider changes"""
        if not slider_updating[0]:
            idx = int(val)
            if not is_playing[0]:
                current_frame[0] = idx
                update_frame(idx)
    
    def on_key_press(event):
        """Handle keyboard events"""
        if event.key == ' ':  # Space bar for play/pause
            if is_playing[0]:
                # Pause
                is_playing[0] = False
                try:
                    if hasattr(anim, 'event_source') and anim.event_source:
                        anim.event_source.stop()
                except:
                    pass
                status_text.set_text('Controls: Space=Play/Pause, Left/Right=Step, Slider=Jump | Status: PAUSED')
            else:
                # Play
                is_playing[0] = True
                try:
                    if hasattr(anim, 'event_source') and anim.event_source:
                        if not anim.event_source.isAlive():
                            anim.event_source.start()
                except:
                    try:
                        anim._start()
                    except:
                        pass
                status_text.set_text('Controls: Space=Play/Pause, Left/Right=Step, Slider=Jump | Status: PLAYING')
            fig.canvas.draw_idle()
            
        elif event.key in ['left', 'a']:  # Left arrow or 'a' for previous
            is_playing[0] = False
            try:
                if hasattr(anim, 'event_source') and anim.event_source:
                    anim.event_source.stop()
            except:
                pass
            current_frame[0] = max(0, current_frame[0] - 1)
            update_frame(current_frame[0])
            status_text.set_text('Controls: Space=Play/Pause, Left/Right=Step, Slider=Jump | Status: PAUSED')
            fig.canvas.draw_idle()
            
        elif event.key in ['right', 'd']:  # Right arrow or 'd' for next
            is_playing[0] = False
            try:
                if hasattr(anim, 'event_source') and anim.event_source:
                    anim.event_source.stop()
            except:
                pass
            current_frame[0] = min(len(flow_files) - 1, current_frame[0] + 1)
            update_frame(current_frame[0])
            status_text.set_text('Controls: Space=Play/Pause, Left/Right=Step, Slider=Jump | Status: PAUSED')
            fig.canvas.draw_idle()
    
    # Update slider callback to also update status
    def on_slider_change_with_status(val):
        """Handle slider changes with status update"""
        on_slider_change(val)
        if not is_playing[0]:
            status_text.set_text('Controls: Space=Play/Pause, Left/Right=Step, Slider=Jump | Status: PAUSED')
    
    # Connect callbacks
    slider.on_changed(on_slider_change_with_status)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Set initial status
    status_text.set_text('Controls: Space=Play/Pause, Left/Right=Step, Slider=Jump | Status: PAUSED')
    
    # Ensure figure can receive keyboard events
    fig.canvas.set_window_title('Animation - Click on figure to enable keyboard controls')
    
    # Don't use tight_layout with manually positioned widgets
    # plt.tight_layout()  # Commented out to avoid warnings with manual positioning
    
    if save_path is not None:
        save_path = os.path.join(os.path.dirname(__file__), save_path)
        anim.save(save_path, dpi=150, writer='ffmpeg')

    plt.show()
    return anim


if __name__ == '__main__':
    # Adjust parameters as needed; set save_path to save mp4
    animate_u_contours(save_path=None)


