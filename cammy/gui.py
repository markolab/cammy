import dearpygui.dearpygui as dpg
import cv2
import multiprocessing
import numpy as np
import queue
from cammy.util import (
    intensity_to_rgba,
    mpl_to_cv2_colormap,
)

slider_defaults_min = {
    "default_value": 1800,
    "min_value": 0,
    "max_value": 5000,
}

slider_defaults_max = {
    "default_value": 2200,
    "min_value": 0,
    "max_value": 5000,
}
colormap_default = "gray"
gui_ncols = 2  # number of cols before we start new row
# for labeling videos
font = cv2.FONT_HERSHEY_SIMPLEX
white = (255, 255, 255)
txt_pos = (25, 25)




# simple data writer, should be general enough to take 1d/2d/etc. data
class FrameDisplay(multiprocessing.Process):
    def __init__(self, queue, cameras, downsample=1, display_params={}, display_colormap=None):
        multiprocessing.Process.__init__(self)
        
        self.queue = queue
        self.id = id
        self.display_params = {}
        
        for _id, _cam in cameras.items():
            self.display_params[_id] = {}
            self.display_params[_id]["width"] = _cam._width // downsample
            self.display_params[_id]["height"] = _cam._height // downsample
        
        for _id, _cam in cameras.items():
            use_config = {}
            for k, v in display_params.items():
                if k in _id:
                    use_config = v

            self.display_params[_id] = {**self.display_params[_id], **use_config}


        if display_colormap is None:
            self.display_colormap = mpl_to_cv2_colormap(colormap_default)
        else:
            self.display_colormap = mpl_to_cv2_colormap(display_colormap)


    def run(self):
        dpg.create_context()
        with dpg.texture_registry(show=False):
            for _id, _params in self.display_params.items():
                blank_data = np.zeros(
                    (_params["height"], _params["width"], 4),
                    dtype="float32",
                )
                dpg.add_raw_texture(
                    _params["width"],
                    _params["height"],
                    blank_data,
                    tag=f"texture_{_id}",
                    format=dpg.mvFormat_Float_rgba,
                )

        miss_status = {}
        for _id, _params in self.display_params.items():
            with dpg.window(
                label=f"Camera {_id}", tag=f"Camera {_id}"
            ):
                dpg.add_image(f"texture_{_id}")
                with dpg.group(horizontal=True):
                    dpg.add_slider_float(
                        tag=f"texture_{_id}_min",
                        width=_params["width"] / 3,
                        **{**slider_defaults_min, **_params["slider_defaults_min"]},
                    )
                    dpg.add_slider_float(
                        tag=f"texture_{_id}_max",
                        width=_params["width"] / 3,
                        **{**slider_defaults_max, **_params["slider_defaults_max"]},
                    )
                miss_status[_id] = dpg.add_text(f"0 missed frames / 0 total")


        gui_x_offset = 0
        gui_y_offset = 0
        gui_x_max = 0
        gui_y_max = 0
        row_pos = 0
        for _id, _params in self.display_params.items():
            cur_key = f"Camera {_id}"
            dpg.set_item_pos(cur_key, (gui_x_offset, gui_y_offset))

            width = _params["width"] + 25
            height = _params["height"] + 100

            gui_x_max = int(np.maximum(gui_x_offset + width, gui_x_max))
            gui_y_max = int(np.maximum(gui_y_offset + height, gui_y_max))
            
            row_pos += 1
            if row_pos == gui_ncols:
                row_pos = 0
                gui_x_offset = 0
                gui_y_offset += height
            else:
                gui_x_offset += width


        dpg.create_viewport(title="Live preview", width=gui_x_max, height=gui_y_max)

        # dpg.set_viewport_vsync(False)
        # dpg.show_metrics()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        last_framegrab = np.nan
        while dpg.is_dearpygui_running():
            for _id, _params in self.display_params.items():
                dat = None			   
                try:
                    dat = self.queue[_id].get_nowait()
                except (queue.Empty, KeyboardInterrupt, EOFError):
                    continue
                if dat is None:
                    continue
                else:
                    frame, tstamps = dat
                grab_time = tstamps["system_timestamp"]
                cam_fps = 1 / (((grab_time - last_framegrab) / 1e9) + 1e-12)
                last_framegrab = grab_time
                
                if dat is None:
                    continue
                disp_min = dpg.get_value(f"texture_{_id}_min")
                disp_max = dpg.get_value(f"texture_{_id}_max")
                height, width = frame.shape
                disp_img = cv2.resize(
                    frame, (_params["width"], _params["height"])
                )
                plt_val = intensity_to_rgba(
                    disp_img, minval=disp_min, maxval=disp_max, colormap=self.display_colormap
                ).astype("float32")
                cv2.putText(
                    plt_val, str(tstamps["total_frames"]), txt_pos, font, 1, (1, 1, 1, 1)
                )
                dpg.set_value(f"texture_{_id}", plt_val)
                miss_frames = float(tstamps["missed_frames"])
                total_frames = float(tstamps["total_frames"])
                percent_missed = (miss_frames / total_frames) * 100
                dpg.set_value(
                    miss_status[_id],
                    f"{miss_frames} missed / {total_frames} total ({percent_missed:.1f}% missed)\n{cam_fps:.1f} FPS",
                )

                dpg.render_dearpygui_frame()

        dpg.destroy_context()
    
