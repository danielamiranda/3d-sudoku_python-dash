import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_auth
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import json
import base64
import plotly.graph_objs as go
import random
from random import randint
import pulp as plp
from pulp import *

####----------------3D SUDOKU MODEL--------------#####
m_n = randint(0, 2000)

## Basic Model
model = plp.LpProblem(name="MIP Model")

set_N = [1, 2, 3, 4]
set_I = [1, 2, 3, 4]
set_J = [1, 2, 3, 4]
set_K = [1, 2, 3, 4]
first = [1, 2]
second = [3, 4]

# X is Binary
x_vars = plp.LpVariable.dicts('X',
                              [(n, i, j, k) for n in set_N
                               for i in set_I
                               for j in set_J
                               for k in set_K],
                              0, 1, plp.LpBinary)
model += m_n, "Arbitrary Objective Function"
# i
for n in set_N:
    for j in set_J:
        for k in set_K:
            model += plp.lpSum(x_vars[(n, i, j, k)] for i in set_I) == 1
# j
for n in set_N:
    for i in set_I:
        for k in set_K:
            model += plp.lpSum(x_vars[(n, i, j, k)] for j in set_J) == 1
# k
for n in set_N:
    for i in set_I:
        for j in set_J:
            model += plp.lpSum(x_vars[(n, i, j, k)] for k in set_K) == 1
# n
for k in set_K:
    for i in set_I:
        for j in set_J:
            model += plp.lpSum(x_vars[(n, i, j, k)] for n in set_N) == 1
# i=1,2- j=1,2
for k in set_K:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for i in first for j in first) == 1
# i=1,2- j=3,4
for k in set_K:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for i in first for j in second) == 1
# i=3,4- j=1,2
for k in set_K:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for i in second for j in first) == 1
# i=3,4- j=3,4
for k in set_K:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for i in second for j in second) == 1
# i=1,2- k=1,2
for j in set_J:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for i in first for k in first) == 1
# i=1,2- k=3,4
for j in set_J:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for i in first for k in second) == 1
# i=3,4- k=1,2
for j in set_J:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for i in second for k in first) == 1
# i=3,4- k=3,4
for j in set_J:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for i in second for k in second) == 1
# j=1,2- k=1,2
for i in set_I:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for j in first for k in first) == 1
# j=1,2- k=3,4
for i in set_I:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for j in first for k in second) == 1
# j=3,4- k=1,2
for i in set_I:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for j in second for k in first) == 1
# j=3,4- k=3,4
for i in set_I:
    for n in set_N:
        model += plp.lpSum(x_vars[(n, i, j, k)] for j in second for k in second) == 1

## Solve basic model
model.solve()
opt_df = pd.DataFrame.from_dict(x_vars, orient="index",
                                columns=["variable_object"])

opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["column_n", "column_i", "column_j", "column_k"])
opt_df.reset_index(inplace=True)
# PuLP
opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)
opt_df.drop(columns=["variable_object"], inplace=True)
opt_df.to_csv("./optimization_solution.csv")

## Uniqueness

prime = plp.LpProblem("uniqueness", LpMinimize)

# Decision variables
set_N = [1, 2, 3, 4]
set_I = [1, 2, 3, 4]
set_J = [1, 2, 3, 4]
set_K = [1, 2, 3, 4]
first = [1, 2]
second = [3, 4]

# X is Binary
xp_vars = plp.LpVariable.dicts('Xp',
                               [(n, i, j, k) for n in set_N
                                for i in set_I
                                for j in set_J
                                for k in set_K],
                               0, 1, plp.LpBinary)

##Solution array
df_sol = opt_df[opt_df['solution_value'] == 1.0]

##Constraints
# i
for n in set_N:
    for j in set_J:
        for k in set_K:
            prime += plp.lpSum(xp_vars[(n, i, j, k)] for i in set_I) == 1
# j
for n in set_N:
    for i in set_I:
        for k in set_K:
            prime += plp.lpSum(xp_vars[(n, i, j, k)] for j in set_J) == 1
# k
for n in set_N:
    for i in set_I:
        for j in set_J:
            prime += plp.lpSum(xp_vars[(n, i, j, k)] for k in set_K) == 1
# n
for k in set_K:
    for i in set_I:
        for j in set_J:
            prime += plp.lpSum(xp_vars[(n, i, j, k)] for n in set_N) == 1
# i=1,2- j=1,2
for k in set_K:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for i in first for j in first) == 1
# i=1,2- j=3,4
for k in set_K:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for i in first for j in second) == 1
# i=3,4- j=1,2
for k in set_K:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for i in second for j in first) == 1
# i=3,4- j=3,4
for k in set_K:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for i in second for j in second) == 1
# i=1,2- k=1,2
for j in set_J:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for i in first for k in first) == 1
# i=1,2- k=3,4
for j in set_J:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for i in first for k in second) == 1
# i=3,4- k=1,2
for j in set_J:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for i in second for k in first) == 1
# i=3,4- k=3,4
for j in set_J:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for i in second for k in second) == 1
# j=1,2- k=1,2
for i in set_I:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for j in first for k in first) == 1
# j=1,2- k=3,4
for i in set_I:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for j in first for k in second) == 1
# j=3,4- k=1,2
for i in set_I:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for j in second for k in first) == 1
# j=3,4- k=3,4
for i in set_I:
    for n in set_N:
        prime += plp.lpSum(xp_vars[(n, i, j, k)] for j in second for k in second) == 1

indices = list(df_sol.index)
random.shuffle(indices)


def give_color(c):
    if c == 0:
        return 'grey'
    elif c == 1:
        return 'blue'
    elif c == 2:
        return 'green'
    elif c == 3:
        return 'red'
    else:
        return 'yellow'


def check_game(df_solved):
    diff = abs(df_solved['solution_value'] - opt_df['solution_value'])
    if diff.sum() > 0:
        return "You Lose!!"
    else:
        return "You Win!!"


def random_x(df, i):
    list_var = [df['column_n'][indices[i]], df['column_i'][indices[i]], df['column_j'][indices[i]],
                df['column_k'][indices[i]]]
    df = df.drop(indices[i])
    return df, list_var


def remove_clues(q, prime1, df1, indices):
    for i in range(q):
        get = random_x(df1, i)
        df1 = get[0]
        indices.append(get[1])
    for idx, row in df1.iterrows():
        prime1 += xp_vars[(row['column_n'], row['column_i'], row['column_j'], row['column_k'])] == 1
    return df1


def give_puzzle(u, level, df_):
    list_indices_rem = []
    prime__ = prime.copy()
    if level == 4:
        q = u - 4

    elif level == 3:
        q = u - 8  # ,u-11)
    elif level == 2:
        q = u - 18
    else:
        q = u - 30
    df_f = remove_clues(q, prime__, df_, list_indices_rem)
    puzzle = [0 for i in range(64)]
    for i in range(len(df_f)):
        puzzle[list(df_f.index)[i] % 64] = list(df_f['column_n'])[i]
    return puzzle


def find_uniqueness(df_):
    df_u = df_.copy()
    for i in range(63):
        df = df_u
        list_indices_rem = []
        prime_ = prime.copy()
        remove_clues(i, prime_, df, list_indices_rem)
        prime_ += plp.lpSum(xp_vars[(row[0], row[1], row[2], row[3])] for row in list_indices_rem)
        prime_.solve()
        prime_df = pd.DataFrame.from_dict(xp_vars, orient="index",
                                          columns=["variable_object"])

        prime_df.index = pd.MultiIndex.from_tuples(prime_df.index,
                                                   names=["column_n", "column_i", "column_j", "column_k"])
        prime_df.reset_index(inplace=True)
        prime_df["solution_value"] = prime_df["variable_object"].apply(lambda item: item.varValue)
        prime_df.drop(columns=["variable_object"], inplace=True)
        prime_df.to_csv("./optimization_solution.csv")
        diff = abs(prime_df['solution_value'] - opt_df['solution_value'])
        if diff.sum() > 0:
            return i
            break
        else:
            continue


def create_df_sol(p):
    # print(p)
    sol = [0 for i in range(256)]
    for idx in range(64):
        sol[(64 * (p[idx] - 1)) + idx] = 1.0
    df_s = opt_df.copy()
    df_s['solution_value'] = sol
    return df_s


uniq = find_uniqueness(df_sol)

# For security reasons, this code does not contain the password pairs. Feel free to add your own
VALID_USERNAME_PASSWORD_PAIRS = []

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

## Logo loading
image_filename = '3d_sudoku_logo_3.png'
encoded_logo = base64.b64encode(open(image_filename, 'rb').read())
logo = html.Img(src='data:image/png;base64,{}'.format(encoded_logo.decode()), style={'width': '30vh'})

### Dashboard
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

########-------- PUT EVERYTHING HERE FOR THE MAIN BUTTONS --------########
## Level options
level_options = [{'label': 'Easy', 'value': 1, 'disabled': False},
                 {'label': 'Medium', 'value': 2, 'disabled': False},
                 {'label': 'Difficult', 'value': 3, 'disabled': False},
                 {'label': 'Expert', 'value': 4, 'disabled': False}]
sel_radio = dcc.RadioItems(id='level-picker', options=level_options, value=1, style={'width': '90%'},
                           labelStyle={'display': 'inline-block'})

# Color options
col_options = [{'label': 'Blue', 'value': 1},
               {'label': 'Green', 'value': 2},
               {'label': 'Red', 'value': 3},
               {'label': 'Yellow', 'value': 4}]
sel_color = dcc.Dropdown(id='color-picker', options=col_options, value=1, style={'width': '90%'})

# Play-Reset buttons
play_btn = html.Button('Play', id='play_btn', style={'display': 'block'}, n_clicks_timestamp=0)
reset_btn = html.Button('Reset', id='reset_btn', style={'display': 'none'}, n_clicks_timestamp=0)
submit_btn = html.Button('Submit', id='submit_btn', style={'display': 'none'}, n_clicks_timestamp=0)

########-------- PUT EVERYTHING HERE FOR GAME TAB --------########
sudoku_plot = dcc.Graph(id='sudoku_plot')
stopwatch = daq.LEDDisplay(id='my-LED-display', label="Timer", value=0)

content_main_tab = html.Div(children=[
    html.Div(sudoku_plot, id='sud-plot',
             style={'vertical-align': 'center', 'horizontal-align': 'center', 'width': '100vh'}),
    html.Div(id='hidden-div', style={'display': 'none'}),
    html.Div(id='hidden-div-initial-puzzle', style={'display': 'none'}),
    html.Div([
        html.Div(stopwatch, style={'float': 'left', 'width': '50vh'}),
        html.Div([
            html.Div([
                html.H1(id='result', children='')
            ]),
        ], className="modal-container", id="result-container",
            style={'float': 'right', 'horizontal-align': 'center', 'width': '50vh', 'display': 'none'}),
    ], style={'width': '90%', 'display': 'inline-block'}),

    html.Div([dcc.Interval(id='interval1', interval=1000, n_intervals=0), html.H1(id='label1', children='')])
],
    style={'width': '90%', 'height': '100%'})

########-------- MAKING APP LAYOUT --------########
app.layout = html.Div([
    html.Div([
        html.Div(children=logo, style={'height': '100', 'display': 'inline-block', 'vertical-align': 'center'}),
        html.Label(id='overall-title', children='Select difficulty level: '),
        html.P(),
        html.Div(id='level-radio-items', children=[sel_radio]),
        html.P(),
        html.P(),
        html.P(),
        html.Div(id='color-dropdown', children=[sel_color]),
        html.P(),
        html.P(),
        html.P(),
        html.Div(id='play-reset-buttons', children=[play_btn, reset_btn, submit_btn],
                 style={'display': 'inline-block'}),
    ],
        style={'float': 'left', 'width': '31vh', 'margin': {'r': 20, 't': 0, 'b': 0, 'l': 0},
               'borderRight': 'thin lightgrey solid', 'padding': '10px 5px', 'height': '100vh'}),
    html.Div([
        dcc.Tabs(id="tab_game", children=[
            dcc.Tab(id='tab_sudoku', label='3D-Sudoku', value='3D-Sudoku', children=[
                content_main_tab
            ], style={'float': 'right', 'width': '100vh', 'height': '100vh'}),
        ], style={'float': 'right', 'width': '85vh'})
    ]),
    html.Div(id='intermediate-value', style={'display': 'none'}),
])


########-------- CALLBACKS MAIN BUTTONS --------########
### Play button
@app.callback(
    Output('play_btn', 'style'),
    [Input('play_btn', 'n_clicks_timestamp'),
     Input('reset_btn', 'n_clicks_timestamp')]
)
def dis_en_play_btn(click_p, click_r):
    if int(click_p) > int(click_r):
        # print('Play was most recently clicked')
        return {'display': 'none'}
    else:
        print('checking buttons appearance: Play clicking')

    if int(click_p) == 0 and int(click_r) == 0:
        return {'display': 'block'}


### Submit button
@app.callback(
    Output('submit_btn', 'style'),
    [Input('play_btn', 'n_clicks_timestamp'),
     Input('reset_btn', 'n_clicks_timestamp'),
     Input('submit_btn', 'n_clicks_timestamp')]
)
def dis_en_submit_btn(click_p, click_r, click_s):
    if int(click_p) > int(click_r) and int(click_p) > int(click_s):
        # print('Play was most recently clicked')
        return {'display': 'block'}
    elif int(click_r) > int(click_p) and int(click_r) > int(click_s):
        # print('Reset was most recently clicked')
        return {'display': 'none'}
    elif int(click_s) > int(click_p) and int(click_s) > int(click_r):
        # print('Submit was most recently clicked')
        return {'display': 'none'}
    else:
        print('checking buttons appearance: Submit clicking')

    if int(click_p) == 0 and int(click_r) == 0 and int(click_s) == 0:
        return {'display': 'none'}


### Reset button
@app.callback(
    Output('reset_btn', 'style'),
    [Input('submit_btn', 'n_clicks_timestamp'),
     Input('reset_btn', 'n_clicks_timestamp')]
)
def dis_en_reset_btn(click_s, click_r):
    if int(click_s) > int(click_r):
        # print('Submit was most recently clicked')
        return {'display': 'block'}
    elif int(click_r) > int(click_s):
        # print('Reset was most recently clicked')
        return {'display': 'none'}
    else:
        print('checking buttons appearance: Reset clicking')

    if int(click_s) == 0 and int(click_r) == 0:
        return {'display': 'none'}


### Radio buttons for levels
@app.callback(
    Output('level-radio-items', 'style'),
    [Input('play_btn', 'n_clicks_timestamp'),
     Input('reset_btn', 'n_clicks_timestamp')]
)
def disable_enable_radio_items(click_p, click_r):
    if int(click_p) > int(click_r):
        # print('Play was most recently clicked')
        return {'display': 'none'}
    elif int(click_r) > int(click_p):
        # print('Reset was most recently clicked')
        return {'display': 'block'}
    else:
        print('checking buttons appearance: Radio btns')


@app.callback(
    Output('hidden-div-initial-puzzle', 'children'),
    [Input('level-picker', 'value'), Input('play_btn', 'n_clicks')]
)
def create_sudoku(lvl, play_clicks):
    df_p = df_sol.copy()
    if lvl is None:
        lvl = 1

    if play_clicks is not None:
        puzzle1 = give_puzzle(uniq, lvl, df_p)
        print('the original puzzle: ', puzzle1)
        return puzzle1
    else:
        return None


@app.callback(
    Output('hidden-div', 'children'),
    [Input('sudoku_plot', 'clickData'), Input('color-picker', 'value')],
    [State('hidden-div', 'children')])
def get_selected_data(clickData, clr, previous):
    if clickData is not None:
        result = clickData['points']
        result[-1]['new_value'] = clr
        if previous:
            previous_list = json.loads(previous)
            if previous_list is not None:
                result = previous_list + result

        return json.dumps(result)


@app.callback(
    Output('sudoku_plot', 'figure'),
    [Input('hidden-div-initial-puzzle', 'children'), Input('hidden-div', 'children')]
)
def create_update_sudoku(original_puzzle, data):
    if original_puzzle:
        if data:
            data_points = json.loads(data)
            for p in data_points:
                if p['marker.color'] == 'grey':
                    pxy = p["pointNumber"]
                    original_puzzle[pxy] = p['new_value']
                else:
                    print('you cannot change the color of this ball')

            # last_x = data_points[-1]['x']
            # last_y = data_points[-1]['y']
            # last_z = data_points[-1]['z']

        list_colors = [give_color(c) for c in original_puzzle]

        x = [0 for i in range(16)] + [1 for i in range(16)] + [2 for i in range(16)] + [3 for i in range(16)]
        y = [0 for i in range(4)] + [1 for i in range(4)] + [2 for i in range(4)] + [3 for i in range(4)]
        y = y * 4
        z = [i for i in range(4)] * 16

        trace1 = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=20,
                color=list_colors,
                opacity=0.8
            ))
        layout = go.Layout(
            margin=dict(l=5, r=5, b=5, t=5),
            scene=dict(xaxis=dict(ticks='', showticklabels=False),
                       yaxis=dict(ticks='', showticklabels=False),
                       zaxis=dict(ticks='', showticklabels=False),
                       ),
            width=620,
            height=620
        )

        return {'data': [trace1],
                'layout': layout}


@app.callback(
    Output('interval1', 'interval'),
    [Input('submit_btn', 'n_clicks')])
def update_interval_submit(click_s):
    if click_s:
        sub_time = 60 * 60 * 1000
    else:
        sub_time = 1000
    return sub_time


@app.callback(Output('my-LED-display', 'value'),
              [Input('play_btn', 'n_clicks_timestamp'), Input('interval1', 'n_intervals'),
               Input('reset_btn', 'n_clicks_timestamp'), Input('submit_btn', 'n_clicks_timestamp')])
def update_interval(click_p, n, click_r, click_s):
    if int(click_p) > int(click_r) and int(click_p) > int(click_s):
        # print('Play was most recently clicked')
        return str(n)
    elif int(click_r) > int(click_p) and int(click_r) > int(click_s):
        # print('Reset was most recently clicked')
        return str(0)
    elif int(click_s) > int(click_p) and int(click_s) > int(click_r):
        # print('Submit was most recently clicked')
        return str(n)
    else:
        print('None of the buttons have been clicked yet')


@app.callback(
    Output('intermediate-value', 'children'),
    [Input('level-picker', 'value'), Input('play_btn', 'n_clicks'), Input('hidden-div', 'children')])
def get_list_colors(lvl, play_clicks, data):
    df_p = df_sol.copy()
    if lvl == None:
        lvl = 1

    if play_clicks:
        puzzle1 = give_puzzle(uniq, lvl, df_p)
        if data:
            data_points = json.loads(data)
            for p in data_points:
                if p['marker.color'] == 'grey':
                    pxy = p["pointNumber"]
                    puzzle1[pxy] = p['new_value']
                else:
                    print('you cannot change the color of this ball')

        return puzzle1


##submit reset btns
@app.callback(Output('result', 'children'),
              [Input('submit_btn', 'n_clicks_timestamp'), Input('reset_btn', 'n_clicks_timestamp'),
               Input('intermediate-value', 'children')])
def give_result(click_s, click_r, list_string_colors):
    if int(click_s) > int(click_r):
        df_solved = create_df_sol(list_string_colors)
        return check_game(df_solved)
    if int(click_r) > int(click_s):
        return 'Press F5 to start a new game'


@app.callback(Output('result-container', 'style'),
              [Input('submit_btn', 'n_clicks'), Input('reset_btn', 'n_clicks')])
def modal_display_status(click_s, click_r):
    if click_s is not None or click_r is not None:
        return {'display': 'inline'}
    else:
        return {'display': 'none'}


### Stop game
@app.callback(
    Output('sud-plot', 'style'),
    [Input('reset_btn', 'n_clicks')])
def reset_puzzle(click_r):
    if click_r:
        return {'display': 'none'}


########-------- RUNNING MAIN SCRIPT --------########

if __name__ == '__main__':
    app.run_server(debug=True)
