import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

# Load the dataset
df = pd.read_csv('data/prepocessed_data.csv')

# Drop the unnecessary index column
df = df.drop(columns=['Unnamed: 0'])

# List of variables for dropdowns
variables = df.columns

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Insurance Charges Dashboard", className="text-center mb-4"), width=12)),

    # Scatter Plot selection
    dbc.Row([
        dbc.Col([
            html.Label("Select X variable for Scatter Plot:"),
            dcc.Dropdown(
                id='scatter-x-dropdown',
                options=[{'label': var, 'value': var} for var in variables],
                value='age'  # Default value
            ),
            html.Label("Select Y variable for Scatter Plot:"),
            dcc.Dropdown(
                id='scatter-y-dropdown',
                options=[{'label': var, 'value': var} for var in variables],
                value='charges'  # Default value
            ),
            dcc.Graph(id='scatter-plot'),
        ], width=6),

        # Pie Chart selection
        dbc.Col([
            html.Label("Select variable for Pie Chart:"),
            dcc.Dropdown(
                id='pie-dropdown',
                options=[{'label': var, 'value': var} for var in ['sex', 'smoker', 'region']],
                value='smoker'  # Default value
            ),
            dcc.Graph(id='pie-chart'),
        ], width=6),
    ]),

    # Histogram and Correlation Matrix
    dbc.Row([
        dbc.Col([
            html.Label("Select variable for Histogram:"),
            dcc.Dropdown(
                id='hist-dropdown',
                options=[{'label': var, 'value': var} for var in variables],
                value='charges'  # Default value
            ),
            dcc.Graph(id='hist-plot'),
        ], width=6),

        dbc.Col(dcc.Graph(figure=px.imshow(df.corr(), text_auto=True, title="Correlation Matrix")), width=6),
    ]),

    # Line Plot and Box Plot
    dbc.Row([
        dbc.Col([
            html.Label("Select Y variable for Line Plot:"),
            dcc.Dropdown(
                id='line-y-dropdown',
                options=[{'label': var, 'value': var} for var in variables],
                value='charges'  # Default value
            ),
            dcc.Graph(id='line-plot'),
        ], width=6),

        dbc.Col([
            html.Label("Select variable for Boxplot:"),
            dcc.Dropdown(
                id='boxplot-dropdown',
                options=[{'label': var, 'value': var} for var in variables],
                value='charges'  # Default value
            ),
            dcc.Graph(id='boxplot'),
        ], width=6),
    ])
])

# Scatter Plot Callback
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-x-dropdown', 'value'),
     Input('scatter-y-dropdown', 'value')]
)
def update_scatter_plot(x_var, y_var):
    fig = px.scatter(df, x=x_var, y=y_var, color='smoker', title=f"Scatter Plot: {x_var} vs {y_var}")
    return fig

# Pie Chart Callback
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('pie-dropdown', 'value')]
)
def update_pie_chart(pie_var):
    fig = px.pie(df, names=pie_var, title=f"Pie Chart: {pie_var}")
    return fig

# Histogram Callback
@app.callback(
    Output('hist-plot', 'figure'),
    [Input('hist-dropdown', 'value')]
)
def update_histogram(hist_var):
    fig = px.histogram(df, x=hist_var, title=f"Histogram: {hist_var}")
    return fig

# Line Plot Callback
@app.callback(
    Output('line-plot', 'figure'),
    [Input('line-y-dropdown', 'value')]
)
def update_line_plot(y_var):
    fig = px.line(df, x='age', y=y_var, title=f"Line Plot: {y_var} vs Age")
    return fig

# Boxplot Callback
@app.callback(
    Output('boxplot', 'figure'),
    [Input('boxplot-dropdown', 'value')]
)
def update_boxplot(box_var):
    fig = px.box(df, y=box_var, title=f"Boxplot: {box_var}")
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
