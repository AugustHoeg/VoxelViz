import os
import argparse
import time
import zarr
import numpy as np
import dask.array as da
from PIL import Image
from nicegui import ui
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

# Configuration
ome_path = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Oak_A/output_ome.zarr"

def parse_arguments():
    parser = argparse.ArgumentParser(description="OME-Zarr Viewer")
    parser.add_argument('--ome_path', type=str, default=ome_path, help='Path to the OME-Zarr root directory')
    args = parser.parse_args()
    return args

class OMEZarrProgressiveViewer:
    def __init__(self, zarr_root_path):
        self.root_path = zarr_root_path
        self.available_groups = self.find_image_groups(zarr_root_path)
        self.current_group = self.available_groups[0] if self.available_groups else "0"

        # resolution levels: 0 is highest, higher numbers are lower resolution
        self.HIGH_RES = 0
        self.DEBOUNCE_TIME = 0.3  # Wait 0.3s before starting to sharpen

        self.sliders = []
        self.views = []
        self.slice_labels = []

        # Initial Data Load
        self.load_active_group(self.current_group)

        ui.page_title('Progressive Sharpening Viewer')

        with ui.column().classes('w-full items-center'):
            with ui.card().classes('w-full max-w-4xl p-4 items-center bg-slate-50'):
                ui.markdown(f"### OME-Zarr: `{zarr_root_path.split('/')[-1]}`")
                with ui.row().classes('items-center gap-4'):
                    ui.label("Volume:")
                    ui.select(options=self.available_groups, value=self.current_group,
                              on_change=self.handle_group_change).classes('w-40')

            with ui.row().classes('w-full justify-center gap-4 mt-4'):
                labels = ["XY", "XZ", "YZ"]
                for i in range(3):
                    with ui.column().classes('items-center bg-slate-100 p-2 rounded shadow-sm'):
                        ui.label(labels[i]).classes('font-bold text-gray-700')
                        self.views.append(ui.interactive_image().style('width: 400px; height: 400px;'))
                        sl = ui.slider(min=0, max=self.shapes[0][i] - 1, value=self.indices[i],
                                       on_change=lambda e, axis=i: self.on_slider_move(axis, e.value))
                        self.sliders.append(sl)
                        lbl = ui.label().classes('text-xs font-mono text-gray-600 mt-[-10px]')
                        self.slice_labels.append(lbl)

        ui.timer(0.05, self.update_loop)
        self.update_label_text()
        self.force_refresh()

    def load_active_group(self, group_name):
        full_path = os.path.join(self.root_path, group_name)
        loc = parse_url(full_path)
        reader = Reader(loc)
        nodes = list(reader())
        self.pyramid = [da.squeeze(p) for p in nodes[0].data]
        self.pyramid = [p[0] if p.ndim > 3 else p for p in self.pyramid]
        self.shapes = [p.shape for p in self.pyramid]

        # Set LOW_RES dynamically based on available levels
        self.LOW_RES = len(self.pyramid) - 1

        self.indices = [s // 2 for s in self.shapes[0]]
        self.last_rendered_indices = [-1, -1, -1]
        self.rendered_levels = [self.LOW_RES, self.LOW_RES, self.LOW_RES]
        self.last_interaction = time.time()

    def update_loop(self):
        """Step-by-step resolution sharpening logic."""
        now = time.time()
        # Still interacting?
        is_moving = (now - self.last_interaction) < self.DEBOUNCE_TIME

        for i in range(3):
            # 1. Check if the index changed (User is sliding)
            if self.indices[i] != self.last_rendered_indices[i]:
                # Immediately show the lowest resolution for speed
                self.render_slice(i, self.indices[i], self.LOW_RES)
                self.last_rendered_indices[i] = self.indices[i]
                self.rendered_levels[i] = self.LOW_RES

            # 2. If stopped, and we haven't reached HIGH_RES (0) yet, sharpen by 1 level
            elif not is_moving and self.rendered_levels[i] > self.HIGH_RES:
                next_level = self.rendered_levels[i] - 1
                self.render_slice(i, self.indices[i], next_level)
                self.rendered_levels[i] = next_level

    def render_slice(self, axis, level0_index, level):
        """Fetches and displays slice from the requested level."""
        scale_factor = self.shapes[level][axis] / self.shapes[0][axis]
        scaled_index = int(level0_index * scale_factor)
        scaled_index = min(scaled_index, self.shapes[level][axis] - 1)

        data = self.pyramid[level]
        if axis == 0:
            arr = data[scaled_index, :, :]
        elif axis == 1:
            arr = data[:, scaled_index, :]
        else:
            arr = data[:, :, scaled_index]

        # Extract, normalize, and push to UI
        arr = arr.compute()
        arr = arr.astype(np.float32)
        arr = (np.clip(arr, 0, 65535) / 65535 * 255).astype(np.uint8)
        self.views[axis].source = Image.fromarray(arr)

    # --- Utility Methods ---
    def find_image_groups(self, path):
        try:
            root = zarr.open_group(path, mode='r')
            return sorted([k for k in root.group_keys()])
        except:
            return ["0"]

    def handle_group_change(self, e):
        self.load_active_group(e.value)
        for i, slider in enumerate(self.sliders):
            slider.max = self.shapes[0][i] - 1
            slider.value = self.indices[i]
            slider.update()
        self.update_label_text()
        self.force_refresh()

    def on_slider_move(self, axis, val):
        self.indices[axis] = int(val)
        self.last_interaction = time.time()
        self.slice_labels[axis].set_text(f"Slice: {self.indices[axis]:03d} / {self.shapes[0][axis]}")

    def update_label_text(self):
        for i in range(3):
            self.slice_labels[i].set_text(f"Slice: {self.indices[i]:03d} / {self.shapes[0][i]}")

    def force_refresh(self):
        self.last_rendered_indices = [-1, -1, -1]
        self.rendered_levels = [self.LOW_RES] * 3


if __name__ in {"__main__", "__mp_main__"}:

    args = parse_arguments()

    viewer = OMEZarrProgressiveViewer(args.ome_path)
    ui.run()