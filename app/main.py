import argparse
from typing import Tuple, List, Union
from datetime import datetime
import time
import numpy as np
from epics import PV
from p4p.client.thread import Context,  Disconnected
import copy
from collections import defaultdict
from functools import partial

from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.server.server import Server
from bokeh import palettes
from bokeh.models import Div, Label, Spacer, ColumnDataSource, TableColumn, StringFormatter, DataTable, Button, Select
from bokeh.models.formatters import DatetimeTickFormatter
from lume_epics.client.controller import Controller
from bokeh.models.widgets import HTMLTemplateFormatter
from bokeh.themes import built_in_themes

from lume_epics.client.widgets.tables import ValueTable 
from lume_epics.client.widgets.controls import build_sliders, EntryTable
from lume_epics.client.widgets.plots import Striptool, ImagePlot
from lume_epics.client.monitors import PVTimeSeries, PVImage

from lume_model.variables import ImageVariable
from lume_model.utils import variables_from_yaml
from lume_model.variables import ScalarVariable
import sys

parser = argparse.ArgumentParser(description='Parse bokeh args.')
parser.add_argument('prefix', type=str, help='Prefix for process variables.')
parser.add_argument('protocol', type=str, help="Protocol for accessing pvs.", choices=["pva", "ca"])

args = parser.parse_args()
prefix = args.prefix
protocol = args.protocol
scale_mode = "scale_both"


DEFAULT_IMAGE_DATA = {
    "image": [np.zeros((50, 50))],
    "x": [50],
    "y": [50],
    "dw": [0.01],
    "dh": [0.01],
}

DEFAULT_SCALAR_VALUE = 0


class CustomController:
    """
    Controller class used to access process variables. Controllers are used for 
    interfacing with both Channel Access and pvAccess process variables. The 
    controller object is initialized using a single protocol has methods for
    both getting and setting values on the process variables.

    Attributes:
        protocol (str): Protocol for getting values from variables ("pva" for pvAccess, "ca" for
            Channel Access)

        context (Context): P4P threaded context instance for use with pvAccess.

        set_ca (bool): Update Channel Access variable on put.

        set_pva (bool): Update pvAccess variable on put.

        pv_registry (dict): Registry mapping pvname to dict of value and pv monitor

    Example:
        ```
        # create PVAcess controller
        controller = Controller("pva")

        value = controller.get_value("scalar_input")
        image_value = controller.get_image("image_input")

        controller.close()

        ```

    """

    def __init__(self, protocol: str, prefix, track_inputs: bool = False, input_pvs: list = None):
        """
        Initializes controller. Stores protocol and creates context attribute if 
        using pvAccess.

        Args: 
            protocol (str): Protocol for getting values from variables ("pva" for pvAccess, "ca" for
            Channel Access)

        """
        self.protocol = protocol
        self.last_update = ""
        self.pv_registry = defaultdict()
        self.track_inputs = track_inputs
        self.input_pvs = [f"{prefix}:{variable}" for variable in input_pvs]
        self.prefix = prefix

        # initalize context for pva
        self.context = None
        if self.protocol == "pva":
            self.context = Context("pva")


    def ca_value_callback(self, pvname, value, *args, **kwargs):
        """Callback executed by Channel Access monitor.

        Args:
            pvname (str): Process variable name

            value (Union[np.ndarray, float]): Value to assign to process variable.
        """
        self.pv_registry[pvname]["value"] = value

        if self.track_inputs:
            if pvname in self.input_pvs:
                self.last_update = datetime.now().strftime('%m/%d/%Y, %H:%M:%S')


    def ca_connection_callback(self, *, pvname, conn, pv):
        """Callback used for monitoring connection and setting values to None on disconnect.
        """
        # if disconnected, set value to None
        if not conn:
            self.pv_registry[pvname]["value"] = None


    def pva_value_callback(self, pvname, value):
        """Callback executed by pvAccess monitor.

        Args:
            pvname (str): Process variable name

            value (Union[np.ndarray, float]): Value to assign to process variable.
        """
        if isinstance(value, Disconnected):
            self.pv_registry[pvname]["value"] = None
        else:
            self.pv_registry[pvname]["value"] = value

        if self.track_inputs:
            if pvname in self.input_pvs:
                self.last_update = datetime.now().strftime('%m/%d/%Y, %H:%M:%S')



    def setup_pv_monitor(self, pvname):
        """Set up process variable monitor.

        Args:
            pvname (str): Process variable name

        """
        if pvname in self.pv_registry:
            return

        if self.protocol == "ca":
            # add to registry (must exist for connection callback)
            self.pv_registry[pvname] = {"pv": None, "value": None}

            # create the pv
            pv_obj = PV(pvname, callback=self.ca_value_callback, connection_callback=self.ca_connection_callback)

            # update registry
            self.pv_registry[pvname]["pv"] = pv_obj


        elif self.protocol == "pva":
            cb = partial(self.pva_value_callback, pvname)
            # populate registry s.t. initially disconnected will populate
            self.pv_registry[pvname] = {"pv": None, "value": None}

            # create the monitor obj
            mon_obj = self.context.monitor(pvname, cb, notify_disconnect=True)
            
            # update registry with the monitor
            self.pv_registry[pvname]["pv"] = mon_obj


    def get(self, pvname: str) -> np.ndarray:
        """
        Accesses and returns the value of a process variable.

        Args:
            pvname (str): Process variable name

        """
        self.setup_pv_monitor(pvname)
        pv = self.pv_registry.get(pvname, None)
        if pv:
            #return pv.get("value", None)
            return pv["value"]
        return None


    def get_value(self, pvname):
        """Gets scalar value of a process variable.

        Args:
            pvname (str): Image process variable name.

        """
        value = self.get(pvname)

        if value is None:
            value = DEFAULT_SCALAR_VALUE

        return value


    def get_image(self, pvname) -> dict:
        """Gets image data via controller protocol.

        Args:
            pvname (str): Image process variable name

        """
        image = None
        if self.protocol == "ca":
            image_flat = self.get(f"{pvname}:ArrayData_RBV")
            nx = self.get(f"{pvname}:ArraySizeX_RBV")
            ny = self.get(f"{pvname}:ArraySizeY_RBV")
            x = self.get(f"{pvname}:MinX_RBV")
            y = self.get(f"{pvname}:MinY_RBV")
            x_max = self.get(f"{pvname}:MaxX_RBV")
            y_max = self.get(f"{pvname}:MaxY_RBV")

            if all([image_def is not None for image_def in [image_flat, nx, ny, x, y, x_max, y_max]]):
                dw = x_max - x
                dh = y_max - y

                image = image_flat.reshape(int(nx), int(ny))

        elif self.protocol == "pva":
            # context returns numpy array with WRITEABLE=False
            # copy to manipulate array below

            image = self.get(pvname)

            if image is not None:
                attrib = image.attrib
                x = attrib["x_min"]
                y = attrib["y_min"]
                dw = attrib["x_max"] - attrib["x_min"]
                dh = attrib["y_max"] - attrib["y_min"]
                image = copy.copy(image)

        if image is not None:
            return {
                "image": [image],
                "x": [x],
                "y": [y],
                "dw": [dw],
                "dh": [dh],
            }

        else:
            return DEFAULT_IMAGE_DATA


    def put(self, pvname, value: Union[np.ndarray, float], timeout=1.0) -> None:
        """Assign the value of a process variable.

        Args:
            pvname (str): Name of the process variable

            value (Union[np.ndarray, float]): Value to assing to process variable.

            timeout (float): Operation timeout in seconds

        """
        self.setup_pv_monitor(pvname)

        # allow no puts before a value has been collected
        registered = self.get(pvname)

        # if the value is registered
        if registered is not None:
            if self.protocol == "ca":
                self.pv_registry[pvname]["pv"].put(value, timeout=timeout)

            elif self.protocol == "pva":
                self.context.put(pvname, value, throw=False, timeout=timeout)

        else:
            logger.debug(f"No initial value set for {pvname}.")

    def close(self):
        if self.protocol == "pva":
            self.context.close()



class PVTimeSeriesTimestamped(PVTimeSeries):
    def poll(self) -> Tuple[np.ndarray]:
        """
        Collects image data via appropriate protocol and returns time and data.

        """
        t = datetime.now()

        v = self.controller.get_value(self.pvname)

        self.time = np.append(self.time, t)
        self.data = np.append(self.data, v)

        return self.time, self.data


class CorrectedImagePlot(ImagePlot):
    """
    Object for viewing and updating an image plot.

    Attributes:
        live_variable (str): Current variable to be displayed

        source (ColumnDataSource): Data source for the viewer.

        pv_monitors (PVImage): Monitors for the process variables.

        plot (Figure): Bokeh figure object for rendering.

        img_obj (GlyphRenderer): Bokeh glyph renderer for displaying image.

    Example:

        ```
        prefix = "test"

        # controller initialized to use Channel Access
        controller = Controller("ca")

        value_table = ImagePlot(
                [output_variables["image_variable"]], 
                controller, 
                prefix
            )

        ```
    """

    def __init__(
        self, variables: List[ImageVariable], controller: Controller, prefix: str
    ) -> None:
        """
        Initialize monitors, current process variable, and data source.

        Args:
            variables (List[ImageVariable]): List of image variables to include in plot

            controller (Controller): Controller object for getting pv values

            prefix (str): Prefix used for server

        """
        self.pv_monitors = {}

        for variable in variables:
            self.pv_monitors[variable.name] = PVImage(prefix, variable, controller)

        self.live_variable = list(self.pv_monitors.keys())[0]
        image_data = DEFAULT_IMAGE_DATA
        image_data["image"][0] = np.flipud(image_data["image"][0].T)

        self.source = ColumnDataSource(image_data)


    def update(self, live_variable: str = None) -> None:
        """
        Callback which updates the plot to reflect updated process variable values or 
        new process variable.

        Args:
            live_variable (str): Variable to display
        """
        # update internal pv trackinng
        if live_variable:
            self.live_variable = live_variable

        # update axis and labels
        axis_labels = self.pv_monitors[self.live_variable].axis_labels
        axis_units = self.pv_monitors[self.live_variable].axis_units

        x_axis_label = axis_labels[0]
        y_axis_label = axis_labels[1]

        if axis_units:
            x_axis_label += " (" + axis_units[0] + ")"
            y_axis_label += " (" + axis_units[1] + ")"

        self.plot.xaxis.axis_label = x_axis_label
        self.plot.yaxis.axis_label = y_axis_label

        # get image data
        image_data = self.pv_monitors[self.live_variable].poll()
        image_data["image"][0] = np.flipud(image_data["image"][0].T)

        self.source.data.update(image_data)



class FixedImagePlot(ImagePlot):

    def __init__(
        self, variables: List[ImageVariable], controller: Controller, prefix: str
    ) -> None:
        """
        Initialize monitors, current process variable, and data source.

        Args:
            variables (List[ImageVariable]): List of image variables to include in plot

            controller (Controller): Controller object for getting pv values

            prefix (str): Prefix used for server

        """
        self.pv_monitors = {}

        for variable in variables:
            self.pv_monitors[variable.name] = PVImage(prefix, variable, controller)

        self.live_variable = list(self.pv_monitors.keys())[0]

        image_data = DEFAULT_IMAGE_DATA

        image_data["image"][0] = np.flipud(image_data["image"][0].T)

        self.source = ColumnDataSource(image_data)


    def build_plot(
        self, palette, color_mapper=None
    ) -> None:
        """
        Creates the plot object.

        Args:
            palette (Optional[tuple]): Bokeh color palette to use for plot.

            color_mapper (Optional[ColorMapper]): Bokeh color mapper for rendering 
                plot.

        """
        # create plot
        self.plot = figure(
            tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")],
                sizing_mode=scale_mode, x_range=(-0.01,0.01), y_range=(-0.01,0.01)
        )

        if color_mapper:
            self.plot.image(
                name="image_plot",
                image="image",
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                source=self.source,
                color_mapper=color_mapper,
            )
        elif palette:
            self.plot.image(
                name="image_plot",
                image="image",
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                source=self.source,
                palette=palette,
            )

        axis_labels = self.pv_monitors[self.live_variable].axis_labels
        axis_units = self.pv_monitors[self.live_variable].axis_units

        x_axis_label = axis_labels[0]
        y_axis_label = axis_labels[1]

        if axis_units:
            x_axis_label += " (" + axis_units[0] + ")"
            y_axis_label += " (" + axis_units[1] + ")"

        self.plot.xaxis.axis_label = x_axis_label
        self.plot.yaxis.axis_label = y_axis_label

    def update(self, live_variable: str = None) -> None:
        """
        Callback which updates the plot to reflect updated process variable values or 
        new process variable.

        Args:
            live_variable (str): Variable to display
        """
        # update internal pv trackinng
        if live_variable:
            self.live_variable = live_variable

        # update axis and labels
        axis_labels = self.pv_monitors[self.live_variable].axis_labels
        axis_units = self.pv_monitors[self.live_variable].axis_units

        x_axis_label = axis_labels[0]
        y_axis_label = axis_labels[1]

        if axis_units:
            x_axis_label += " (" + axis_units[0] + ")"
            y_axis_label += " (" + axis_units[1] + ")"

        self.plot.xaxis.axis_label = x_axis_label
        self.plot.yaxis.axis_label = y_axis_label

        # get image data
        image_data = self.pv_monitors[self.live_variable].poll()

        image_data["image"][0] = np.flipud(image_data["image"][0].T)

        self.source.data.update(image_data)


# Override striptool update. Label was being replaced
class CustomStriptool(Striptool):
    """Custom class to override the default Striptool update where label was replaced 
    depending on the live process variable.

    """


    def __init__(
        self, variables: List[ScalarVariable], controller: Controller, prefix: str, limit: int = None, aspect_ratio: float = 1.05
    ) -> None:
        """
        Set up monitors, current process variable, and data source.

        Args:
            variables (List[ScalarVariable]): List of variables to display with striptool

            controller (Controller): Controller object for getting process variable values

            prefix (str): Prefix used for server.

            limit (int): Maximimum steps for striptool to render

            aspect_ratio (float): Ratio of width to height

        """
        self.pv_monitors = {}

        for variable in variables:
            self.pv_monitors[variable.name] = PVTimeSeriesTimestamped(prefix, variable, controller)

        self.live_variable = list(self.pv_monitors.keys())[0]

        ts = []
        ys = []
        self.source = ColumnDataSource(dict(x=ts, y=ys))
        self.reset_button = Button(label="Reset")
        self.reset_button.on_click(self._reset_values)
        self._aspect_ratio = aspect_ratio
        self._limit = limit
        self.selection =  Select(
            title="Variable to plot:",
            value=self.live_variable,
            options=list(self.pv_monitors.keys()),
        )
        self.selection.on_change("value", self.update_selection)
        self.build_plot()


    def build_plot(self) -> None:
        """
        Creates the plot object.
        """
        self.plot = figure(sizing_mode=scale_mode, aspect_ratio=self._aspect_ratio, x_axis_type='datetime')
        self.plot.line(x="x", y="y", line_width=2, source=self.source)
        self.plot.yaxis.axis_label = self.live_variable

        # as its scales, the plot uses all definedformats
        self.plot.xaxis.formatter = DatetimeTickFormatter(
            minutes= "%H:%M:%S", 
            minsec="%H:%M:%S", 
            seconds="%H:%M:%S", 
            microseconds="%H:%M:%S", 
            milliseconds="%H:%M:%S"
        )

        self.plot.xaxis.major_label_orientation = "vertical"

        # add units to label
        if self.pv_monitors[self.live_variable].units:
            self.plot.yaxis.axis_label += (
                f" ({self.pv_monitors[self.live_variable].units})"
            )

    def update(self) -> None:
        """
        Callback to update the plot to reflect updated process variable values or to 
        display a new process variable.


        """

        ts, ys = self.pv_monitors[self.live_variable].poll()
        if self._limit is not None and len(ts) > self._limit:
            ts = ts[-self._limit:]
            ys = ys[-self._limit:]

        self.source.data = dict(x=ts, y=ys)



# Override datatable update. Use sig digits
class CustomValueTable(ValueTable):
    """
    Overrides datatable to use significant digits and update font size

    """

    def update(self) -> None:
        """
        Callback function to update data source to reflect updated values.
        """
        for variable in self.pv_monitors:
            v = self.pv_monitors[variable].poll()
            self.output_values[variable] = "{0:.4g}".format(v)

        x_vals = [self.labels[var] for var in self.output_values.keys()]
        y_vals = list(self.output_values.values())
        self.source.data = dict(x=x_vals, y=y_vals)

    def create_table(self) -> None:
        """
        Creates the bokeh table and populates variable data.
        """
        x_vals = [self.labels[var] for var in self.output_values.keys()]
        y_vals = list(self.output_values.values())
        
        table_data = dict(x=x_vals, y=y_vals)
        self.source = ColumnDataSource(table_data)
        columns = [
            TableColumn(
                field="x", title="Variable"
            ),
            TableColumn(field="y", title="Value"),
        ]

        self.table = DataTable(
            source=self.source, columns=columns, index_position=None, autosize_mode = 'fit_columns'
        )


# check out variables
variable_params = [
    "distgen:r_dist:sigma_xy:value",
    "distgen:t_dist:length:value",
    "SOL1:solenoid_field_scale",
    "CQ01:b1_gradient",
    "SQ01:b1_gradient",
    "L0A_phase:dtheta0_deg",
]

# track constants
constants = [
    "distgen:total_charge:value",
    "L0A_scale:voltage"
]



striptool_limit = 10 *  4 * 60 # 10 mins x callbacks / second * seconds/min
controller = CustomController(protocol, prefix, track_inputs=True, input_pvs=variable_params)

pal = palettes.viridis(256)

with open("files/model_config.yml", "r") as f:
    input_variables, output_variables = variables_from_yaml(f)

# track callbacks
callbacks = []



# set up input value table
input_value_table = CustomValueTable([input_variables[var] for var in variable_params], controller, prefix)
callbacks.append(input_value_table.update)

# build input striptools, updating label sizes
striptools = []
for variable in variable_params:
    striptool = CustomStriptool([input_variables[variable]], controller, prefix, limit=striptool_limit)
    striptool.plot.xaxis.axis_label_text_font_size = '7pt'
    striptool.plot.yaxis.axis_label_text_font_size = '7pt'
    striptool.plot.xaxis.major_label_text_font_size = "7pt"
    striptool.plot.yaxis.major_label_text_font_size = "6pt"
    striptool.plot.sizing_mode = scale_mode
    callbacks.append(striptool.update)
    striptools.append(striptool.plot)

# set up the input striptool grid
input_grid =  gridplot(striptools,  ncols=6, sizing_mode="scale_height", merge_tools = True, toolbar_location=None)

# filter variables
image="x:y"

scalar_outputs = [
    "end_n_particle",
    "end_norm_emit_x",
    "end_norm_emit_y",
    "end_sigma_x",
    "end_sigma_y",
    "end_higher_order_energy_spread",
]

# create labels to be used with output striptool
output_labels = {
    "end_n_particle": "n particles",
    "end_norm_emit_x": "norm emit x",
    "end_norm_emit_y": "norm emit y",
    "end_sigma_x": "σₓ",
    "end_sigma_y": "σᵧ",
    "end_higher_order_energy_spread": "higher order energy spread",
}

# set up output value table
output_value_table = CustomValueTable([output_variables[var] for var in scalar_outputs], controller, prefix)
callbacks.append(output_value_table.update)

# build output striptools and update settings 
striptools = []
for variable in scalar_outputs:
    striptool = CustomStriptool([output_variables[variable]], controller, prefix, limit=striptool_limit)
    callbacks.append(striptool.update)
    striptool.plot.yaxis.axis_label = output_labels[variable] + f" ({striptool.pv_monitors[variable].units})"
    striptool.plot.xaxis.axis_label_text_font_size = '7pt'
    striptool.plot.yaxis.axis_label_text_font_size = '7pt'
    striptool.plot.xaxis.major_label_text_font_size = "7pt"
    striptool.plot.yaxis.major_label_text_font_size = "6pt"
    striptools.append(striptool.plot)


image = CorrectedImagePlot([output_variables["x:y"]], controller, prefix)
image.build_plot(pal)
image.plot.xaxis.major_label_text_font_size = "6pt"
image.plot.yaxis.major_label_text_font_size = "6pt"
image.plot.sizing_mode = scale_mode
image.plot.toolbar_location=None
callbacks.append(image.update)

# fixed image
fixed_image = FixedImagePlot([output_variables["x:y"]], controller, prefix)
fixed_image.build_plot(pal)
fixed_image.plot.xaxis.major_label_text_font_size = "6pt"
fixed_image.plot.yaxis.major_label_text_font_size = "6pt"
fixed_image.plot.sizing_mode = scale_mode
fixed_image.plot.toolbar_location=None
callbacks.append(fixed_image.update)


output_grid = gridplot(striptools,  ncols=6, sizing_mode="scale_height", merge_tools=True, toolbar_location=None)


title_div = Div(text=f"<b>LCLS-CU-INJ: Last input update {controller.last_update}</b>", style={'font-size': '150%', 'color': '#3881e8', 'text-align': 'center', 'width':'100%'}, sizing_mode=scale_mode,)
input_div_label = Div(text="<b>INPUTS</b>", style={'font-size': '150%', 'color': '#3881e8'}, name="input_title")
output_div_label = Div(text="<b>OUTPUTS</b>", style={'font-size': '150%', 'color': '#3881e8'})

def update_title():
    global controller
    title_div.text = f"<b>LCLS-CU-INJ: Last input update {controller.last_update}</b>"

input_value_table.table.height=175
input_value_table.table.width=400
output_value_table.table.height=175
output_value_table.table.width=400

callbacks.append(update_title)

image.plot.aspect_ratio = 1.2
fixed_image.plot.aspect_ratio = 1.2

curdoc().theme="dark_minimal"
curdoc().add_root(
    column(
        row(title_div, sizing_mode="scale_width"),
        row(
            column(
                input_div_label,
                input_value_table.table,
                output_div_label,
                output_value_table.table,
            ),
            column(
                #row(sys_fig, sizing_mode="scale_height"),
                row(image.plot, fixed_image.plot, sizing_mode="scale_width"),
                Div(text="<b>YAG02</b>", style={'font-size': '150%', 'color': '#3881e8', 'width': '100%', 'text-align': 'center'}),
                Div(text="<img src='app/static/cu_inj_layout.png'  class='sys-fig'/>", style={'text-align': 'center'}), sizing_mode=scale_mode,
            )
        ),
        input_div_label,
        input_grid,
        output_div_label,
        output_grid,
        sizing_mode=scale_mode,
    )
)

for callback in callbacks:
    curdoc().add_periodic_callback(callback, 250)
