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


pal = palettes.viridis(256)

with open("files/model_config.yml", "r") as f:
    input_variables, output_variables = variables_from_yaml(f)

input_variables = {x: input_variables[x] for x in variable_params if x in input_variables}
striptool_limit = 10 *  4 * 60 # 10 mins x callbacks / second * seconds/min
controller = Controller(protocol, input_variables, output_variables, prefix)

# track callbacks
callbacks = []


# set up input value table
input_value_table = ValueTable(input_variables.values(), controller, sig_figs=4)
callbacks.append(input_value_table.update)

# build input striptools, updating label sizes
striptools = []
for variable in variable_params:
    striptool = Striptool([input_variables[variable]], controller, limit=striptool_limit)
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
output_value_table = ValueTable([output_variables[var] for var in scalar_outputs], controller, labels =output_labels)
callbacks.append(output_value_table.update)

# build output striptools and update settings 
striptools = []
for variable in scalar_outputs:
    striptool = Striptool([output_variables[variable]], controller, limit=striptool_limit)
    callbacks.append(striptool.update)
    striptool.plot.yaxis.axis_label = output_labels[variable] + f" ({striptool.pv_monitors[variable].units})"
    striptool.plot.xaxis.axis_label_text_font_size = '7pt'
    striptool.plot.yaxis.axis_label_text_font_size = '7pt'
    striptool.plot.xaxis.major_label_text_font_size = "7pt"
    striptool.plot.yaxis.major_label_text_font_size = "6pt"
    striptools.append(striptool.plot)


image = ImagePlot([output_variables["x:y"]], controller, palette=pal)
image.plot.xaxis.major_label_text_font_size = "6pt"
image.plot.yaxis.major_label_text_font_size = "6pt"
image.plot.sizing_mode = scale_mode
image.plot.toolbar_location=None
callbacks.append(image.update)

# fixed image
fixed_image = ImagePlot([output_variables["x:y"]], controller, x_range=(-8e-4,8e-4), y_range=(-8e-4,8e-4), palette=pal)
fixed_image.plot.xaxis.major_label_text_font_size = "6pt"
fixed_image.plot.yaxis.major_label_text_font_size = "6pt"
fixed_image.plot.sizing_mode = scale_mode
fixed_image.plot.toolbar_location=None
callbacks.append(fixed_image.update)


output_grid = gridplot(striptools,  ncols=6, sizing_mode="scale_height", merge_tools=True, toolbar_location=None)


title_div = Div(text=f"<b>LCLS-CU-INJ: Last input update {controller.last_input_update}</b>", style={'font-size': '150%', 'color': '#3881e8', 'text-align': 'center', 'width':'100%'}, sizing_mode=scale_mode,)
input_div_label = Div(text="<b>INPUTS</b>", style={'font-size': '150%', 'color': '#3881e8'}, name="input_title")
output_div_label = Div(text="<b>OUTPUTS</b>", style={'font-size': '150%', 'color': '#3881e8'})

def update_title():
    global controller
    title_div.text = f"<b>LCLS-CU-INJ: Last input update {controller.last_input_update}</b>"

input_value_table.table.height=165
input_value_table.table.width=400
output_value_table.table.height=165
output_value_table.table.width=400

callbacks.append(update_title)

image.plot.aspect_ratio = 1.25
fixed_image.plot.aspect_ratio = 1.25

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
                Div(text="<img src='app/static/cu_inj_layout.png' class='sys-fig'/>", style={'text-align': 'center', 'width': '100%'}, sizing_mode="scale_width"), 
                sizing_mode=scale_mode,
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
