import dash
import dash_table
from dash_table.Format import Format
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px

import numpy as np
from skimage import io, filters, measure, color, img_as_ubyte
import PIL
import pandas as pd
import matplotlib as mpl
from skimage.transform import resize

from util_baseline_comp import baseline_edges, baseline_of_the_row
import traceback

#

# Set up the app
# external_stylesheets = [dbc.themes.BOOTSTRAP, "assets/object_properties_style.css"]
external_stylesheets = [dbc.themes.LITERA, "assets/object_properties_style.css"]
# themes at:
# https://bootswatch.com/solar/
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


def read_and_process_file(wf):
    # wf = pd.read_csv(fn, header=None, delimiter=";", decimal=',')
    # print(wf.head())
    # Clean of -99 errors
    cut_off_value = -5.2  # Keyence LJ-V7020, Measurement range, Z-axis (height): ±2.6 mm (F.S.=5.2 mm)
    start_min = np.nanmin(wf.values)
    counter = 0
    while start_min < cut_off_value:
        wf.replace(start_min, np.nan, inplace=True)
        wf.fillna(method='ffill', inplace=True)
        start_min = np.nanmin(wf.values)
        counter += 1
    wf.replace([np.inf, -np.inf], np.nan)
    wf.fillna(method='ffill', inplace=True)
    # Baseline
    wf.iloc[:, 0] = baseline_edges(wf.iloc[:, 0])
    wf.iloc[:, -1] = baseline_edges(wf.iloc[:, -1])
    bl = wf.copy(deep=True).apply(lambda row: baseline_of_the_row(row),
                                  axis=1)  # axis=1 --> row-wise /=0 --> column-wise
    bl.fillna(method='ffill', inplace=True)  # filling of missing values (nan)
    flat = wf - bl
    flat.clip(0.0, None, inplace=True)
    # as array
    arr = flat.to_numpy().reshape(flat.shape)
    # resizing
    # from skimage.transform import resize
    downscale_factor = 0.005 / 0.010
    x_shape = arr.shape[1]
    y_shape = int(arr.shape[0] * downscale_factor)
    arr = resize(arr, (y_shape, x_shape), anti_aliasing=True)
    # rotate
    if arr.shape[0] > arr.shape[1]:
        arr = np.rot90(arr)  # switches x and y --> y and x
        rot = 1
    else:
        rot = 0
    # Grayscale image
    arr = pd.DataFrame(arr)
    arr -= np.nanmin(arr.values)
    zmax = np.nanmax(arr.values)
    arr *= (255.0 / zmax)
    img = arr.to_numpy(dtype='uint8').reshape(arr.shape)  # <-- IMAGE

    # Segmentation
    radius = 8  # 5 to 15
    from skimage.morphology import disk
    selem = disk(radius)
    import cv2 as cv
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    from skimage.filters import rank
    local_thresh = rank.otsu(blurred, selem)
    from skimage.morphology import closing, square
    bw = closing(img > local_thresh, square(3))  # TODO < --- check
    # bw = closing(img > 0, square(3))
    # --remove artifacts connected to image border
    from skimage.segmentation import clear_border
    cleared = clear_border(bw)
    # --filling small holes in obj. mask
    from scipy import ndimage as ndi
    cleared = ndi.binary_fill_holes(cleared, structure=np.ones((5, 5))).astype(int)
    # --label image regions
    from skimage.measure import label
    label_array = label(cleared)  # colored objects map


    # Compute and store properties of the labeled image
    # prop_names = [
    #     "label",
    #     "area",
    #     "perimeter",
    #     "eccentricity",
    #     "euler_number",
    #     "mean_intensity",
    # ]
    prop_names = [
        'label',
        "eccentricity",
        'area',
        # 'filled_area',
        'minor_axis_length',
        'major_axis_length',
        'equivalent_diameter',
        'orientation',
        'weighted_local_centroid',
        'centroid',
    ]
    prop_table = measure.regionprops_table(
        label_array, intensity_image=img, properties=prop_names
    )

    table = pd.DataFrame(prop_table)
    # Filter:
    # print(table.info())
    # print(table.head())
    # TODO: Computig actual properties in µm, mm, etc.
    # centroid-0, centroid-1

    table["area"] = table["area"] * 0.01 * 0.01
    table['area'] = table['area'].round(3)
    not_table = table[table["area"] <= 0.1000]
    table = table[table["area"] > 0.1000]
    # SORTING After applying any filters for droplets

    table = table.sort_values(["centroid-1"], ascending=(True,))
    # print(table["centroid-1"])
    # # Compute DBSCAN
    #
    # from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, AgglomerativeClustering
    # # ms = MeanShift(bandwidth=None, bin_seeding=True, n_jobs=-1)
    # # ms = KMeans(n_clusters=3, random_state=0)
    # ms = AgglomerativeClustering()
    # X = table["centroid-1"].values
    # print(X)
    # ms.fit(np.reshape(X, (-1, 1)))
    # lbl = ms.labels_
    # # cluster_centers = ms.cluster_centers_
    # labels_uni = np.unique(lbl)
    # n_clusters_ = len(labels_uni)
    # print("number of estimated clusters : %d" % n_clusters_)
    # print(lbl)
    # table['clusters'] = lbl
    # table = table.sort_values(["clusters"], ascending=(True,))

    table['eccentricity'] = table['eccentricity'].round(2)
    table['new_label'] = np.arange(1, table.index.values.shape[0] + 1)
    table.reset_index(drop=True, inplace=True)
    label_array[np.isin(label_array, not_table['label'].values)] = 0
    for row_name, row_data in table.iterrows():
        label_array[label_array == int(row_data['label'])] = int(row_data['new_label'])
    current_labels = np.unique(label_array)[np.nonzero(np.unique(label_array))]
    table['label'] = table['new_label']

    # https://stackoverflow.com/questions/55692129/how-to-use-skimage-measure-regionprops-to-query-labels
    # def mp(entry, map_dict):
    #     # return mapper_dict[entry] if entry in mapper_dict else entry
    #     return map_dict.get(entry, entry)
    #
    #
    # mapper_dict = dict(zip(table['label'].values, table['new_label'].values))
    # label_array = np.vectorize(label_array, mapper_dict)
    #
    #
    # table['label'] = table['new_label']

    # Format the Table columns
    columns = [
        {"name": label_name, "id": label_name, "selectable": True}
        if precision is None
        else {
            "name": label_name,
            "id": label_name,
            "type": "numeric",
            "format": Format(precision=precision),
            "selectable": True,
        }
        for label_name, precision in zip(prop_names, (None, None, 4, 4, None, 3))
    ]
    # Select the columns that are selected when the app starts
    initial_columns = ["label", "area"]

    img = img_as_ubyte(color.gray2rgb(img))
    img = PIL.Image.fromarray(img)
    # Pad label-array with zero to get contours of regions on the edge
    label_array = np.pad(label_array, (1,), "constant", constant_values=(0,))
    return table, current_labels, columns, initial_columns, img, label_array


def image_with_contour(table, label_array, img, active_labels, data_table, active_columns, color_column):
    """
    Returns a greyscale image that is segmented and superimposed with contour traces of
    the segmented regions, color coded by values from a data table.

    Parameters
    ----------
    img : PIL Image object.
    active_labels : list
        the currently visible labels in the datatable
    data_table : pandas.DataFrame
        the currently visible entries of the datatable
    active_columns: list
        the currently selected columns of the datatable
    color_column: str
        name of the datatable column that is used to define the colorscale of the overlay
    """

    # First we get the values from the selected datatable column and use them to define a colormap
    values = np.array(table[color_column].values)
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = mpl.cm.get_cmap("plasma")

    # Now we convert our background image to a greyscale bytestring that is very small and can be transferred very
    # efficiently over the network. We do not want any hover-information for this image, so we disable it
    fig = px.imshow(img, binary_string=True, binary_backend="jpg",)
    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    # For each region that is visible in the datatable, we compute and draw the filled contour, color it based on
    # the color_column value of this region, and add it to the figure
    # here is an small tutorial of this: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html#sphx-glr-auto-examples-segmentation-plot-regionprops-py
    for rid, row in data_table.iterrows():
        label = row.label
        value = row[color_column]
        contour = measure.find_contours(label_array == label, 0.5)[0]
        # We need to move the contour left and up by one, because
        # we padded the label array
        y, x = contour.T - 1
        # We add the values of the selected datatable columns to the hover information of the current region
        hoverinfo = (
            "<br>".join(
                [
                    # All numbers are passed as floats. If there are no decimals, cast to int for visibility
                    f"{prop_name}: {f'{int(prop_val):d}' if prop_val.is_integer() else f'{prop_val:.3f}'}"
                    if np.issubdtype(type(prop_val), "float")
                    else f"{prop_name}: {prop_val}"
                    for prop_name, prop_val in row[active_columns].iteritems()
                ]
            )
            # remove the trace name. See e.g. https://plotly.com/python/reference/#scatter-hovertemplate
            + " <extra></extra>"
        )
        fig.add_scatter(
            x=x,
            y=y,
            name=label,
            opacity=0.8,
            mode="lines",
            line=dict(color=mpl.colors.rgb2hex(cmap(norm(value))),),
            fill="toself",
            customdata=[label] * len(x),
            showlegend=False,
            hovertemplate=hoverinfo,
            hoveron="points+fills",
        )

    # Finally, because we color our contour traces one by one, we need to manually add a colorscale to explain the
    # mapping of our color_column values to the colormap. This also gets added to the figure
    fig.add_scatter(
        # We only care about the colorscale here, so the x and y values can be empty
        x=[None],
        y=[None],
        mode="markers",
        showlegend=False,
        marker=dict(
            colorscale=[mpl.colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 50)],
            showscale=True,
            # The cmin and cmax values here are arbitrary, we just set them to put our value ticks in the right place
            cmin=-5,
            cmax=5,
            colorbar=dict(
                tickvals=[-5, 5],
                ticktext=[f"{np.min(values[values!=0]):.2f}", f"{np.max(values):.2f}",],
                # We want our colorbar to scale with the image when it is resized, so we set them to
                # be a fraction of the total image container
                lenmode="fraction",
                len=0.6,
                thicknessmode="fraction",
                thickness=0.05,
                outlinewidth=1,
                # And finally we give the colorbar a title so the user may know what value the colormap is based on
                title=dict(text=f"<b>{color_column.capitalize()}</b>"),
            ),
        ),
        hoverinfo="none",
    )

    # Remove axis ticks and labels and have the image fill the container
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), template="simple_white")
    fig.update_xaxes(visible=False, range=[0, img.width]).update_yaxes(
        visible=False, range=[img.height, 0]
    )

    return fig


# Define Modal
with open("assets/modal.md", "r") as f:
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
                                src=app.get_asset_url("fsr_logo_white.png"),
                                height="70px",
                            ),
                            href="https://plotly.com/dash/",
                        )
                    ),
                    dbc.Col(
                        html.A(
                            html.Img(
                                # src=app.get_asset_url("dash-logo-new.png"),
                                src=app.get_asset_url("fhikts_logo_white.png"),
                                height="43px",
                            ),
                            href="https://plotly.com/dash/",
                        )
                    ),
                    dbc.Col(dbc.NavbarBrand("Droplet Properties App")),
                    # modal_overlay,
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



# Define Cards
# value="assets/sample_droplets.csv"
file_card = dbc.Card([
        dbc.CardHeader(html.H2("Choose file")),
        dbc.CardBody(
            dbc.Col([html.Div("Please enter the path to the .csv file into the field below and press Submit to view the data it contains.",
                     style={'text-align': 'center'}),
                    html.Div([
                        dcc.Input(id='input_state', type="text", value=None, inputMode='latin', required=True),
                        html.Button(id='submit_button', n_clicks=0, children='Submit'),
                        html.Div(id='output_state'),
                    ], style={'text-align': 'center'}),
                ])
        )
])
# assets/sample_droplets.csv
image_card = dbc.Card(
    [
        dbc.CardHeader(html.H2("Explore object properties")),
        dbc.CardBody(
            dbc.Row(
                dbc.Col(
                    dcc.Graph(id="graph", figure={}),
                    ),
                )
            ),
        dbc.CardFooter(
            dbc.Row(
                [
                    dbc.Col(
                        "Use the dropdown menu to select which variable to base the colorscale on:"
                    ),
                    dbc.Col(dcc.Dropdown(
                        id="color-drop-menu",
                        options=[],
                        value=None,
    )),
                    # dbc.Toast(
                    #     [
                    #         html.P(
                    #             "In order to use all functions of this app, please select a variable "
                    #             "to compute the colorscale on.",
                    #             className="mb-0",
                    #         )
                    #     ],
                    #     id="auto-toast",
                    #     header="No colorscale value selected",
                    #     icon="danger",
                    #     style={
                    #         "position": "fixed",
                    #         "top": 66,
                    #         "left": 10,
                    #         "width": 350,
                    #     },
                    # ),
                ],
                align="center",
            ),
        ),
    ]
)
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
                dbc.Row([dbc.Col(image_card, md=12)]),
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


def col_drop(table):
    # Color selector dropdown
    color_drop = dcc.Dropdown(
        id="color-drop-menu",
        options=[{"label": col_name.capitalize(), "value": col_name}
                 for col_name in table.columns],
        value="label",
    )
    return color_drop


def cout_data_table(columns, table, initial_columns):
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


def cout_data_figure(img, props_table, data_table, label_array, active_columns, color_column="area"):
    """
        Returns a greyscale image that is segmented and superimposed with contour traces of
        the segmented regions, color coded by values from a data table.

        Parameters
        ----------
        img : PIL Image object.
        active_labels : list
            the currently visible labels in the datatable
        data_table : pandas.DataFrame
            the currently visible entries of the datatable
        active_columns: list
            the currently selected columns of the datatable
        color_column: str
            name of the datatable column that is used to define the colorscale of the overlay
        """

    # First we get the values from the selected datatable column and use them to define a colormap
    values = np.array(props_table[color_column].values)
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = mpl.cm.get_cmap("plasma")

    # Now we convert our background image to a greyscale bytestring that is very small and can be transferred very
    # efficiently over the network. We do not want any hover-information for this image, so we disable it
    fig = px.imshow(img, binary_string=True, binary_backend="jpg", )
    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    # For each region that is visible in the datatable, we compute and draw the filled contour, color it based on
    # the color_column value of this region, and add it to the figure
    # here is an small tutorial of this: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html#sphx-glr-auto-examples-segmentation-plot-regionprops-py
    for rid, row in data_table.iterrows():
        label = row.label
        value = row[color_column]
        contour = measure.find_contours(label_array == label, 0.5)[0]
        # We need to move the contour left and up by one, because
        # we padded the label array
        y, x = contour.T - 1
        # We add the values of the selected datatable columns to the hover information of the current region
        hoverinfo = (
                "<br>".join(
                    [
                        # All numbers are passed as floats. If there are no decimals, cast to int for visibility
                        f"{prop_name}: {f'{int(prop_val):d}' if prop_val.is_integer() else f'{prop_val:.3f}'}"
                        if np.issubdtype(type(prop_val), "float")
                        else f"{prop_name}: {prop_val}"
                        for prop_name, prop_val in row[active_columns].iteritems()
                    ]
                )
                # remove the trace name. See e.g. https://plotly.com/python/reference/#scatter-hovertemplate
                + " <extra></extra>"
        )
        fig.add_scatter(
            x=x,
            y=y,
            name=label,
            opacity=0.8,
            mode="lines",
            line=dict(color=mpl.colors.rgb2hex(cmap(norm(value))), ),
            fill="toself",
            customdata=[label] * len(x),
            showlegend=False,
            hovertemplate=hoverinfo,
            hoveron="points+fills",
        )

    # Finally, because we color our contour traces one by one, we need to manually add a colorscale to explain the
    # mapping of our color_column values to the colormap. This also gets added to the figure
    fig.add_scatter(
        # We only care about the colorscale here, so the x and y values can be empty
        x=[None],
        y=[None],
        mode="markers",
        showlegend=False,
        marker=dict(
            colorscale=[mpl.colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 50)],
            showscale=True,
            # The cmin and cmax values here are arbitrary, we just set them to put our value ticks in the right place
            cmin=-5,
            cmax=5,
            colorbar=dict(
                tickvals=[-5, 5],
                ticktext=[f"{np.min(values[values != 0]):.2f}", f"{np.max(values):.2f}", ],
                # We want our colorbar to scale with the image when it is resized, so we set them to
                # be a fraction of the total image container
                lenmode="fraction",
                len=0.6,
                thicknessmode="fraction",
                thickness=0.05,
                outlinewidth=1,
                # And finally we give the colorbar a title so the user may know what value the colormap is based on
                title=dict(text=f"<b>{color_column.capitalize()}</b>"),
            ),
        ),
        hoverinfo="none",
    )

    # Remove axis ticks and labels and have the image fill the container
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), template="plotly_dark")
    fig.update_xaxes(visible=False, range=[0, img.width]).update_yaxes(
        visible=False, range=[img.height, 0]
    )

    return fig


def fig_short(img, props_table):
    fig = px.imshow(img, binary_string=True, binary_backend="jpg", )
    fig.update_traces(hoverinfo="skip", hovertemplate=None)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0, pad=0), template="plotly_dark")
    fig.update_xaxes(visible=False, range=[0, img.width]).update_yaxes(
        visible=False, range=[img.height, 0])
    return fig

from dash.exceptions import PreventUpdate
@app.callback(
    [Output('output_state', 'children'),
     Output("table", 'children'),
     Output("graph", "figure"),
     Output("color-drop-menu", "options"),
     Output("color-drop-menu", "value")],
    [Input(component_id='submit_button', component_property='n_clicks'),
     Input("color-drop-menu", "options"),
     Input("color-drop-menu", "value")],
    [State(component_id='input_state', component_property='value')]
)
def update_output(num_clicks, drop_menu_state_options, drop_menu_state_selected, inserted_path):
    if inserted_path is None:
        raise PreventUpdate
    if inserted_path == "":
        raise PreventUpdate
    else:
        try:
            wf = pd.read_csv(str(inserted_path), header=None, delimiter=";", decimal=',')
            print(' ... LOADED')
            # print(wf.columns[0])
            props_table, current_labels, columns, initial_columns, img, label_array =\
                read_and_process_file(wf)
            data_table = cout_data_table(None, props_table, None)
            drop_menu_options = [{"label": col_name.capitalize(), "value": col_name}
                                 for col_name in props_table.columns]
            if len(drop_menu_state_options) == 0:
                val_active = 'label'
            else:
                # val_active = drop_menu_options[0].values()
                val_active = drop_menu_state_selected
            fig = cout_data_figure(img, props_table, props_table, label_array, active_columns=['label', 'area'], color_column=val_active)
            drop_menu_value = val_active

            print(' ... OUTPUT')
            button_text = 'Filepath: "{}"'\
                          ' (the button has been clicked {} times)'.format(inserted_path, num_clicks)
            return [button_text, data_table, fig, drop_menu_options, drop_menu_value]
        except Exception as e:
            tb = traceback.format_exc()
            print(' ... ERROR')
            print(e, tb)
            button_text = 'No file at provided path: '\
                          '"{}"'\
                          ' (the button has been clicked {} times)'.format(inserted_path, num_clicks)
            return [button_text, None, None, [], None]





if __name__ == "__main__":
    app.run_server(debug=True)
