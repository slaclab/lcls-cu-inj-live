from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.server.server import Server
from bokeh import palettes
from bokeh.models import Div, Label, Spacer, ColumnDataSource, TableColumn, StringFormatter, DataTable
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



# Override datatable update. Use sig digits
class CustomValueTable(ValueTable):

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
                field="x", title="Variable", formatter=StringFormatter(font_style="bold")
            ),
            TableColumn(field="y", title="Value"),
        ]

        self.table = DataTable(
            source=self.source, columns=columns, sizing_mode="stretch_both", index_position=None
        )

protocol = "pva"
prefix = "test"
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

input_value_table = CustomValueTable([input_variables[var] for var in variable_params], controller, prefix)

callbacks.append(input_value_table.update)
input_value_table.table.sizing_mode="scale_both"

# build input striptools
striptools = []
for variable in variable_params:
    striptool = CustomStriptool([input_variables[variable]], controller, prefix, limit=striptool_limit)
    striptool.plot.xaxis.axis_label_text_font_size = '7pt'
    striptool.plot.yaxis.axis_label_text_font_size = '7pt'
    striptool.plot.xaxis.major_label_text_font_size = "6pt"
    striptool.plot.yaxis.major_label_text_font_size = "6pt"
    callbacks.append(striptool.update)
    striptools.append(striptool.plot)


input_grid =  gridplot(striptools,  ncols=6, sizing_mode="scale_both", merge_tools = True, toolbar_location=None)

image="x:y"

# Render outputs
scalar_outputs = [
    "end_n_particle",
    "end_norm_emit_x",
    "end_norm_emit_y",
    "end_sigma_x",
    "end_sigma_y",
    "end_higher_order_energy_spread",
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

output_value_table = CustomValueTable([output_variables[var] for var in scalar_outputs], controller, prefix)
output_value_table.table.sizing_mode = "scale_both"
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
image.plot.sizing_mode = "scale_both"
image.plot.toolbar_location=None
callbacks.append(image.update)

output_grid = gridplot(striptools,  ncols=6, sizing_mode="scale_both", merge_tools=True, toolbar_location=None)

title_div = Div(text="<b>LCLS-CU-INJ</b>", style={'font-size': '150%', 'color': 'blue', 'text-align': 'center'})
input_div_label = Div(text="<b>INPUTS</b>", style={'font-size': '150%', 'color': 'blue'})
output_div_label = Div(text="<b>OUTPUTS</b>", style={'font-size': '150%', 'color': 'blue'})

curdoc().add_root(
    column(
        row(
            column(input_div_label, input_value_table.table, sizing_mode="scale_both"), column(output_div_label, image.plot, sizing_mode="scale_both"), column(Spacer(height=30), output_value_table.table, sizing_mode="scale_width"), sizing_mode="scale_both"
        ),
        input_div_label,
        input_grid,
        output_div_label,
        output_grid,
        height = 675, width=1200, sizing_mode="scale_both"
    )

)

for callback in callbacks:
    curdoc().add_periodic_callback(callback, 250)
