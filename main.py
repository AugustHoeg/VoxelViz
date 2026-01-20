import os
import io
import time
import zarr
import numpy as np
import dask.array as da
from PIL import Image
from nicegui import ui
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

# --- Configuration ---
# Path to your .zarr root
OME_PATH = "../Vedrana_master_project/3D_datasets/datasets/VoDaSuRe/Oak_A/output_ome.zarr"


class OMEZarrOrthoViewer:
    def __init__(self, zarr_root_path):
        self.root_path = zarr_root_path

        # 1. State & Configuration
        self.HIGH_RES = 0
        self.DEBOUNCE_TIME = 0.2  # Wait 0.2s after sliding before sharpening
        self.last_interaction = time.time()

        # UI references
        self.views = []
        self.sliders = []
        self.slice_labels = []

        # 2. Discover available volumes (groups)
        self.available_groups = self.find_image_groups(zarr_root_path)
        self.current_group = self.available_groups[0] if self.available_groups else "0"

        # 3. Initial Data Load (Sets self.indices to center)
        self.load_active_group(self.current_group)

        # 4. Build UI
        ui.page_title('OME-Zarr Progressive Ortho-Viewer')

        with ui.column().classes('w-full items-center p-4'):
            # Header Card
            with ui.card().classes('w-full max-w-5xl p-6 bg-slate-50 shadow-md'):
                ui.markdown(f"## OME-Zarr Explorer: `{os.path.basename(zarr_root_path)}`").classes('m-0')

                with ui.row().classes('items-center gap-6 mt-2'):
                    with ui.column():
                        ui.label("Image Group:").classes('text-xs font-bold uppercase text-gray-500')
                        ui.select(options=self.available_groups, value=self.current_group,
                                  on_change=self.handle_group_change).classes('w-48 bg-white')

                    ui.separator().props('vertical')

                    with ui.column():
                        ui.label("Status:").classes('text-xs font-bold uppercase text-gray-500')
                        self.status_label = ui.label("Ready").classes('text-sm text-green-600 font-medium')

            # 3-Plane Viewport
            with ui.row().classes('w-full justify-center gap-4 mt-6'):
                labels = ["XY (Axial)", "XZ (Coronal)", "YZ (Sagittal)"]
                for i in range(3):
                    with ui.column().classes('items-center bg-slate-100 p-3 rounded-lg border border-slate-200'):
                        ui.label(labels[i]).classes('font-bold text-slate-700 mb-2')

                        # Interactive Image View
                        view = ui.interactive_image().style('width: 400px; height: 400px; background: black;')
                        self.views.append(view)

                        # Slider
                        sl = ui.slider(min=0, max=self.shapes[0][i] - 1, value=self.indices[i],
                                       on_change=lambda e, axis=i: self.on_slider_move(axis, e.value)) \
                            .props('label-always')
                        self.sliders.append(sl)

                        # Slice Counter Label
                        lbl = ui.label().classes('text-xs font-mono text-gray-500 mt-[-8px]')
                        self.slice_labels.append(lbl)

        # 5. Start Logic
        ui.timer(0.05, self.update_loop)
        self.update_label_text()
        self.force_refresh()

    def find_image_groups(self, path):
        """Scans the Zarr root for sub-groups representing different volumes."""
        try:
            root = zarr.open_group(path, mode='r')
            keys = sorted([k for k in root.group_keys()])
            return keys if keys else ["0"]
        except Exception:
            return ["0"]

    def load_active_group(self, group_name):
        """Loads the multi-scale pyramid and centers the indices."""
        full_path = os.path.join(self.root_path, group_name)
        loc = parse_url(full_path)
        reader = Reader(loc)
        nodes = list(reader())

        # Extract dask arrays for all levels
        self.pyramid = [da.squeeze(p) for p in nodes[0].data]
        self.pyramid = [p[0] if p.ndim > 3 else p for p in self.pyramid]
        self.shapes = [p.shape for p in self.pyramid]

        # Set dynamic lowest resolution level
        self.LOW_RES = len(self.pyramid) - 1

        # RESET TO MIDDLE of Level 0
        self.indices = [s // 2 for s in self.shapes[0]]

        # Initialize rendering state
        self.last_rendered_indices = [-1, -1, -1]
        self.rendered_levels = [self.LOW_RES] * 3
        self.last_interaction = time.time()

    def on_slider_move(self, axis, val):
        """Handle user input: update index and interaction timestamp."""
        self.indices[axis] = int(val)
        self.last_interaction = time.time()
        # Update text immediately for responsiveness
        self.slice_labels[axis].set_text(f"Slice: {self.indices[axis]:03d} / {self.shapes[0][axis]}")

    def update_loop(self):
        """Core logic for progressive sharpening through resolution levels."""
        now = time.time()
        is_moving = (now - self.last_interaction) < self.DEBOUNCE_TIME

        for i in range(3):
            # Step 1: User is sliding -> Show coarsest level immediately
            if self.indices[i] != self.last_rendered_indices[i]:
                self.render_slice(i, self.indices[i], self.LOW_RES)
                self.last_rendered_indices[i] = self.indices[i]
                self.rendered_levels[i] = self.LOW_RES

            # Step 2: User stopped -> Sharpen progressively level by level
            elif not is_moving and self.rendered_levels[i] > self.HIGH_RES:
                next_level = self.rendered_levels[i] - 1
                self.render_slice(i, self.indices[i], next_level)
                self.rendered_levels[i] = next_level

    def render_slice(self, axis, level0_index, level):
        """High-speed slice extraction with bit-shifting and PIL handover."""
        # 1. Coordinate Mapping
        scale_factor = self.shapes[level][axis] / self.shapes[0][axis]
        scaled_index = int(level0_index * scale_factor)
        scaled_index = min(scaled_index, self.shapes[level][axis] - 1)

        # 2. Extract Data (Dask)
        data = self.pyramid[level]
        if axis == 0:
            arr = data[scaled_index, :, :]
        elif axis == 1:
            arr = data[:, scaled_index, :]
        else:
            arr = data[:, :, scaled_index]

        # 3. FAST CONVERSION
        # We compute the dask array and immediately bit-shift to 8-bit.
        # This is much faster than floating point division.
        arr_8bit = (arr.compute() >> 8).astype(np.uint8)

        # 4. Update the UI
        # By passing the PIL image object directly, NiceGUI handles the
        # internal byte conversion correctly without the pathlib error.
        self.views[axis].source = Image.fromarray(arr_8bit)

    def handle_group_change(self, e):
        self.status_label.set_text("Loading...")

        # 1. Update backend data
        self.load_active_group(e.value)

        # 2. Batch update UI to prevent "flicker"
        with ui.column():  # Small trick to group updates
            for i, slider in enumerate(self.sliders):
                new_max = self.shapes[0][i] - 1
                slider.props(f'max={new_max}')
                slider.set_value(self.indices[i])  # set_value is slightly faster than .value =

        self.update_label_text()
        self.force_refresh()
        self.status_label.set_text("Ready")

    def update_label_text(self):
        """Sync all three coordinate labels."""
        for i in range(3):
            self.slice_labels[i].set_text(f"Slice: {self.indices[i]:03d} / {self.shapes[0][i]}")

    def force_refresh(self):
        """Invalidates cache to trigger a fresh render in the next timer cycle."""
        self.last_rendered_indices = [-1, -1, -1]
        self.rendered_levels = [self.LOW_RES] * 3


# --- Execution ---
if __name__ in {"__main__", "__mp_main__"}:
    viewer = OMEZarrOrthoViewer(OME_PATH)
    ui.run(title="VoxelViz OME-Zarr", port=8080)