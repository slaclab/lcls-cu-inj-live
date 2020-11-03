import argparse
from typing import Tuple, List
from datetime import datetime
import time
import numpy as np
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
from lume_epics.client.monitors import PVTimeSeries


from bokeh.util.compiler import TypeScript
from lume_model.utils import variables_from_yaml
from lume_model.variables import ScalarVariable
import sys

parser = argparse.ArgumentParser(description='Parse bokeh args.')
parser.add_argument('prefix', type=str, help='Prefix for process variables.')
parser.add_argument('protocol', type=str, help="Protocol for accessing pvs.", choices=["pva", "ca"])

args = parser.parse_args()
prefix = args.prefix
protocol = args.protocol



def time_to_microseconds(t):
    t = t.time()
    dmin = datetime.min
    dummy_tdelta = (datetime.combine(dmin, t) - dmin)
    return dummy_tdelta.total_seconds()*1000

class PVTimeSeriesTimestamped(PVTimeSeries):
    def poll(self) -> Tuple[np.ndarray]:
        """
        Collects image data via appropriate protocol and returns time and data.

        """
        t = datetime.now()
    #    t = time_to_microseconds(t)
        v = self.controller.get_value(self.pvname)

        self.time = np.append(self.time, t)
        self.data = np.append(self.data, v)

        return self.time, self.data


class FixedImagePlot(ImagePlot):
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
                sizing_mode="scale_both", x_range=(-8e-4,8e-4), y_range=(-8e-4,8e-4)
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
        self.plot = figure(sizing_mode="scale_both", aspect_ratio=self._aspect_ratio, x_axis_type='datetime')
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


striptool_limit = 10 *  4 * 60 # 10 mins x callbacks / second * seconds/min
controller = Controller(protocol)

pal = palettes.viridis(256)

with open("files/model_config.yml", "r") as f:
    input_variables, output_variables = variables_from_yaml(f)

# track callbacks
callbacks = []

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


# set up input value table
input_value_table = CustomValueTable([input_variables[var] for var in variable_params], controller, prefix)
callbacks.append(input_value_table.update)

# build input striptools, updating label sizes
striptools = []
for variable in variable_params:
    striptool = CustomStriptool([input_variables[variable]], controller, prefix, limit=striptool_limit)
    striptool.plot.xaxis.axis_label_text_font_size = '7pt'
    striptool.plot.yaxis.axis_label_text_font_size = '7pt'
    striptool.plot.xaxis.major_label_text_font_size = "8pt"
    striptool.plot.yaxis.major_label_text_font_size = "6pt"
    callbacks.append(striptool.update)
    striptools.append(striptool.plot)

# set up the input striptool grid
input_grid =  gridplot(striptools,  ncols=6, sizing_mode="scale_both", merge_tools = True, toolbar_location=None)

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
    "end_n_particle": "n_particles",
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
    striptool.plot.xaxis.major_label_text_font_size = "8pt"
    striptool.plot.yaxis.major_label_text_font_size = "6pt"
    striptools.append(striptool.plot)

image = ImagePlot([output_variables["x:y"]], controller, prefix)
image.build_plot(pal)
image.plot.xaxis.major_label_text_font_size = "6pt"
image.plot.yaxis.major_label_text_font_size = "6pt"
image.plot.sizing_mode = "scale_both"
image.plot.toolbar_location=None
callbacks.append(image.update)

# fixed image
fixed_image = FixedImagePlot([output_variables["x:y"]], controller, prefix)
fixed_image.build_plot(pal)
fixed_image.plot.xaxis.major_label_text_font_size = "6pt"
fixed_image.plot.yaxis.major_label_text_font_size = "6pt"
fixed_image.plot.sizing_mode = "scale_both"
fixed_image.plot.toolbar_location=None
callbacks.append(fixed_image.update)


output_grid = gridplot(striptools,  ncols=6, sizing_mode="scale_both", merge_tools=True, toolbar_location=None)

title_div = Div(text=f"<b class='centered-text'>LCLS-CU-INJ</b> {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}", style={'font-size': '150%', 'color': '#3881e8', 'text-align': 'center'})
input_div_label = Div(text="<b>INPUTS</b>", style={'font-size': '150%', 'color': '#3881e8'}, name="input_title")
output_div_label = Div(text="<b>OUTPUTS</b>", style={'font-size': '150%', 'color': '#3881e8'})

def update_title():
    title_div.text = f"<b class='centered-text'>LCLS-CU-INJ {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}</b>"

callbacks.append(update_title)

curdoc().theme="dark_minimal"
curdoc().add_root(
    column(
        row(title_div, sizing_mode="scale_width"),
        row(
            column(input_div_label, input_value_table.table, sizing_mode="scale_width"), 
            column(output_div_label, output_value_table.table, sizing_mode="scale_width"), 
            column(image.plot, sizing_mode="scale_both"),
            column(fixed_image.plot, sizing_mode="scale_both"),
            sizing_mode="scale_both", 
        ),
        input_div_label,
        input_grid,
        output_div_label,
        output_grid,
        sizing_mode="stretch_both",
    )
)

for callback in callbacks:
    curdoc().add_periodic_callback(callback, 250)
