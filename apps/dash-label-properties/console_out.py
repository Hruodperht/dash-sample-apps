import dash_core_components as dcc
import dash_html_components as html
import dash
import sys

f = open('out.txt', 'w')
f.close()


app = dash.Dash()

app.layout = html.Div([
    dcc.Interval(id='interval1', interval=1 * 1000,
n_intervals=0),
    dcc.Interval(id='interval2', interval=5 * 1000,
n_intervals=0),
    html.H1(id='div-out', children=''),
    html.Iframe(id='console-out',srcDoc='',style={'width':
'100%','height':400})
])

@app.callback(dash.dependencies.Output('div-out',
'children'),
    [dash.dependencies.Input('interval1', 'n_intervals')])
def update_interval(n):
    orig_stdout = sys.stdout
    f = open('out.txt', 'a')
    sys.stdout = f
    print('Intervals Passed: ' + str(n))
    sys.stdout = orig_stdout
    f.close()
    return 'Intervals Passed: ' + str(n)

@app.callback(dash.dependencies.Output('console-out',
'srcDoc'),
    [dash.dependencies.Input('interval2', 'n_intervals')])
def update_output(n):
    file = open('out.txt', 'r')
    data=''
    lines = file.readlines()
    if lines.__len__()<=20:
        last_lines=lines
    else:
        last_lines = lines[-20:]
    for line in last_lines:
        data=data+line + '<BR>'
    file.close()
    return data

app.run_server(debug=False, port=8050)