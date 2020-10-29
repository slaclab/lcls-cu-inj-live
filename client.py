from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.server.server import Server
from bokeh import palettes
from bokeh.models import Div, Label
from lume_epics.client.controller import Controller

from lume_epics.client.widgets.tables import ValueTable 
from lume_epics.client.widgets.controls import build_sliders, EntryTable
from lume_epics.client.widgets.plots import Striptool, ImagePlot

from bokeh.util.compiler import TypeScript
from lume_model.utils import variables_from_yaml


# Override striptool update. Label was being replaced
class CustomStriptool(Striptool):

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


protocol = "pva"
prefix = "test"
striptool_limit =50
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
    "end_mean_z",
]

# track constants
constants = [
    "distgen:total_charge:value",
    "L0A_scale:voltage"
]



value_table = ValueTable(input_variables.values(), controller, prefix)

callbacks.append(value_table.update)
value_table.table.width_policy = "min"
value_table.table.height_policy = "fit"

# build input striptools
striptools = []
for variable in variable_params:
    striptool = CustomStriptool([input_variables[variable]], controller, prefix, limit=striptool_limit)
   # striptool.plot.yaxis.axis_label = input_labels[variable] + f" ({striptool.pv_monitors[variable].units})"
    striptool.plot.xaxis.axis_label_text_font_size = '7pt'
    striptool.plot.yaxis.axis_label_text_font_size = '7pt'
    striptool.plot.xaxis.major_label_text_font_size = "6pt"
    striptool.plot.yaxis.major_label_text_font_size = "6pt"
    callbacks.append(striptool.update)
    striptools.append(striptool.plot)


striptool_grid =  gridplot(striptools,  ncols=4, sizing_mode="scale_both")
input_row = row(column(value_table.table, sizing_mode = "stretch_height", width=300), column(striptool_grid, sizing_mode="scale_both"), sizing_mode="scale_both")

image="x:y"

# Render outputs
scalar_outputs = [
    "end_n_particle",
#    "end_mean_gamma",
#    "end_sigma_gamma",
#    "end_mean_x",
#    "end_mean_y",
    "end_norm_emit_x",
    "end_norm_emit_y",
#    "end_norm_emit_z",
    "end_sigma_x",
    "end_sigma_y",
#    "end_sigma_z",
#    "end_mean_px",
#    "end_mean_py",
#    "end_mean_pz",
#    "end_sigma_px",
#    "end_sigma_py",
#    "end_sigma_pz",
    "end_higher_order_energy_spread",
#    "end_cov_x__px",
#    "end_cov_y__py",
#    "end_cov_z__pz",
#    "out_ymax",
#    "out_xmax",
#    "out_ymin",
#    "out_xmin",
]

output_labels = {
    "end_n_particle": "n_particles",
    "end_norm_emit_x": "norm emit x",
    "end_norm_emit_y": "norm emit y",
    "end_sigma_x": "σₓ",
    "end_sigma_y": "σᵧ",
    "end_higher_order_energy_spread": "higher order energy spread",
}


output_row = row()

output_value_table = ValueTable([output_variables[var] for var in scalar_outputs], controller, prefix)
output_value_table.table.width_policy = "min"
output_value_table.table.height_policy = "fit"
callbacks.append(output_value_table.update)

# build output striptools
striptools = []
for variable in scalar_outputs:
    striptool = CustomStriptool([output_variables[variable]], controller, prefix, limit=striptool_limit)
    callbacks.append(striptool.update)
    striptool.plot.yaxis.axis_label = output_labels[variable] + f" ({striptool.pv_monitors[variable].units})"
    striptool.plot.xaxis.axis_label_text_font_size = '7pt'
    striptool.plot.yaxis.axis_label_text_font_size = '7pt'
    striptool.plot.xaxis.major_label_text_font_size = "6pt"
    striptool.plot.yaxis.major_label_text_font_size = "6pt"
    striptools.append(striptool.plot)

image = ImagePlot([output_variables["x:y"]], controller, prefix)
image.build_plot(pal)
image.plot.xaxis.major_label_text_font_size = "6pt"
image.plot.yaxis.major_label_text_font_size = "6pt"
callbacks.append(image.update)

grid1 = gridplot(striptools,  ncols=3, sizing_mode="scale_both")

#image_row = row(column(image.plot, sizing_mode="fixed", width=300, height=300), grid1, sizing_mode="scale_both")

output_row = row(column(output_value_table.table, sizing_mode="stretch_height", width=300),
                column(image.plot, sizing_mode="stretch_height", width=350),
                column(grid1, sizing_mode="scale_both"), sizing_mode="scale_both")
title_div = Div(text="<b>LCLS-CU-INJ</b>", style={'font-size': '150%', 'color': 'blue'})
output_div = Div(text="<b>MODEL OUTPUT</b>", style={'font-size': '150%', 'color': 'blue'})

curdoc().add_root(
    column(row(title_div), input_row,  row(output_div),output_row,sizing_mode="fixed", height=300, width=1200)
)

for callback in callbacks:
    curdoc().add_periodic_callback(callback, 250)
