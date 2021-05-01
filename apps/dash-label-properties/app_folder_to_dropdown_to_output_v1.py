import dash
import dash_table
from dash_table.Format import Format
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import os
import plotly.express as px

import numpy as np
from skimage import io, filters, measure, color, img_as_ubyte
import PIL
import pandas as pd
import matplotlib as mpl
from skimage.transform import resize

from util_baseline_comp import baseline_edges, baseline_of_the_row
import traceback

# Set up the app
external_stylesheets = [dbc.themes.LITERA, "assets/object_properties_style.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server



# Define Modal
with open("../assets/modal.md", "r") as f:
    howto_md = f.read()

modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)

# Buttons
button_gh = dbc.Button(
    "by RJ",
    id="howto-open",
    outline=True,
    color="secondary",
    # Turn off lowercase transformation for class .button in stylesheet
    style={"textTransform": "none"},
)

button_howto = dbc.Button(
    "Code",
    outline=True,
    color="primary",
    href="https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-label-properties",
    id="gh-link",
    style={"text-transform": "none"},
)

# Define Header Layout
header = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.A(
                            html.Img(
                                # src=app.get_asset_url("dash-logo-new.png"),
                                src=app.get_asset_url("assets/fsr_logo_white.png"),
                                height="70px",
                            ),
                            href="https://www.fraunhofer.sg/",
                        )
                    ),
                    dbc.Col(
                        html.A(
                            html.Img(
                                # src=app.get_asset_url("dash-logo-new.png"),
                                src=app.get_asset_url("assets/fhikts_logo_white.png"),
                                height="43px",
                            ),
                            href="https://www.ikts.fraunhofer.de/",
                        )
                    ),
                    dbc.Col(dbc.NavbarBrand("Droplet Properties App")),
                    modal_overlay,
                ],
                align="center",
            ),
            dbc.Row(
                dbc.Col(
                    [
                        dbc.NavbarToggler(id="navbar-toggler"),
                        dbc.Collapse(
                            dbc.Nav(
                                [dbc.NavItem(button_howto), dbc.NavItem(button_gh)],
                                className="ml-auto",
                                navbar=True,
                            ),
                            id="navbar-collapse",
                            navbar=True,
                        ),
                    ]
                ),
                align="center",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
)

df = pd.DataFrame()

# Define Cards
# value="assets/sample_droplets.csv"
file_card = dbc.Card([
        dbc.CardHeader(html.H2("Choose file")),
        dbc.CardBody([
            dbc.Row([html.Div("Please enter the path to the .csv file into the field below and press Submit to view the data it contains.",
                     style={'text-align': 'center'}),
                ]),
            dbc.Col(
                    html.Div([
                        dcc.Input(id='input_path_state', type="text", value=None, inputMode='latin', required=True),
                        html.Button(id='submit_button', n_clicks=0, children='Submit'),
                    ], style={'text-align': 'center'}),
            ),
            dbc.Col(
                html.Div(id='show_submit_clicks', style={'text-align': 'center'}),
            ),
            dbc.Col(
                html.Div(id='show_path_state', style={'text-align': 'center'}),
            ),
            dbc.Col(
                html.Div(children=[
                    dcc.Dropdown(id='dropdown_row_names'),
                ], style={'text-align': 'center'}),
            )
        ])
])

table_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Data Table")),
        dbc.CardBody(
            dbc.Row(
                dbc.Col(id="table", children={}
                )
            )
        ),
    ]
)

app.layout = html.Div(
    [
        header,
        dbc.Container(
            [
                dbc.Row([dbc.Col(file_card, md=12)]),
                dbc.Row([dbc.Col(table_card, md=12)]),
            ],
            fluid=True,
        ),
    ]
)


@app.callback(
    Output("modal", "is_open"),
    [Input("howto-open", "n_clicks"), Input("howto-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# we use a callback to toggle the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# from dash.exceptions import PreventUpdate

def cout_data_table(table):
    columns = [{'name': str(i), 'id': str(i)} for i in table.columns]
    data_tabl = dash_table.DataTable(
        id='table-line',
        columns=columns,
        data=table.to_dict('records'),
        tooltip_header={
            col: "Select columns with the checkbox to include them in the hover info of the image."
            for col in table.columns
        },
        style_header={
            "textDecoration": "underline",
            "textDecorationStyle": "dotted",
        },
        style_table={"overflowY": "scroll"},
        fixed_rows={"headers": False, "data": 0},
        style_cell={"width": "85px"},
    )
    return data_tabl

@app.callback(
    [Output(component_id='table', component_property='children')],
    [Input(component_id='dropdown_row_names', component_property='value')],
    State(component_id='dropdown_row_names', component_property='options'))
def get_row_names(chosen_file_path_from_dropdown, options):
    if chosen_file_path_from_dropdown is None:
        raise PreventUpdate
    else:
        fp = chosen_file_path_from_dropdown
        df = pd.read_csv(fp, header=None, delimiter=";", decimal=',')
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        d = {'df.info': [s]}
        df = pd.DataFrame(data=d)
        # my_type = 'float64'
        # dtypes = df.dtypes.to_dict()
        # check = False
        # for col_nam, typ in dtypes.items():
        #     if (typ != my_type):
        #         check = True
        # if check == True:
        #     fn = os.path.basename(chosen_file_path_from_dropdown)
        #     d = {'ERROR': [fn, 'does not only contain float64 values (...is not a laserprofiler file?)']}
        #     nope_df = pd.DataFrame(data=d)
        #     nope = cout_data_table(nope_df)
        #     return [nope]
        # else:
        #     dt = cout_data_table(df)
        #     return [dt]
        dt = cout_data_table(df)
        return [dt]



@app.callback(
    Output(component_id='dropdown_row_names', component_property='options'),
    [Input(component_id='input_path_state', component_property='value')])
def get_row_names(inserted_path):
    if inserted_path is None:
        raise PreventUpdate
    available_files = []
    for dirpath, dirnames, filenames in os.walk(inserted_path):
        for filenam in filenames:
            if filenam.endswith(".csv"):
                try:
                    fp = os.path.join(dirpath, filenam)
                    # pd.read_csv(fp, header=None, delimiter=";", decimal=',')
                    available_files.append((filenam, fp))
                except:
                    continue
            else:
                continue
    lst = [{'label': i[0], 'value': i[1]} for i in available_files]
    print(lst)
    return lst


@app.callback(
    [Output('show_submit_clicks', 'children'),
     Output('show_path_state', 'children')],
    [Input(component_id='submit_button', component_property='n_clicks')],
    [State(component_id='input_path_state', component_property='value')]
)
def update_input_path(num_clicks_submit_btn, inserted_path):
    if inserted_path is None:
        raise PreventUpdate
    elif os.path.isdir(inserted_path):
        show_submit_clicks_children = '[the button has been clicked {} times]'.format(num_clicks_submit_btn)
        show_path_state_children = 'Filepath: "{}"'.format(inserted_path)
    else:
        show_submit_clicks_children = '[the button has been clicked {} times]'.format(num_clicks_submit_btn)
        show_path_state_children = 'Error: Not a folder - inserted path was "{}"'.format(inserted_path)
    return [show_submit_clicks_children, show_path_state_children]


if __name__ == "__main__":
    app.run_server(debug=True)
