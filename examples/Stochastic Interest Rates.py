from quantutils.models.interest_rates import merton, vasicek
import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import numpy as np
import pandas as pd
from scipy.stats import norm, gaussian_kde

x0 = 0.05
mu = 0.01
sigma = 0.6

MC = 10000
n_steps = 10000
dt = 1/n_steps
# simulations = [merton(x0=x0,
#                       mu=mu,
#                       sigma=sigma,
#                       n_steps=n_steps,
#                       dt=dt)[1] for i in range(MC)]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Stochastic Models of Interest Rates'),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Merton Model', 'value': 'merton'},
            {'label': 'Vasicek Model', 'value': 'vasicek'}
        ],
        value='merton'
    ),
    dcc.Tabs(id="all_tabs", value='description_tab', children=[
        dcc.Tab(label='Description', value='description_tab'),
        dcc.Tab(label='Visualizations', value='visualizations_tab'),
    ]),
    html.Div(id='1D_graph')
])


@app.callback(Output('1D_graph', 'children'),
              Input('all_tabs', 'value'),
              )
def render_content(tab):
    template = "seaborn"
    if tab == 'merton':
        t, xt = merton(x0=x0,
                       mu=mu,
                       sigma=sigma,
                       n_steps=n_steps,
                       dt=dt)
        df = pd.DataFrame.from_dict({"t": t,
                                     "Sim1": xt})
        fig = px.area(df, x="t", y="Sim1",
                      template=template)
        fig.update_xaxes(rangeslider_visible=True)
        content = html.Div([
            html.H3('Merton Model simulation'),
            dcc.Graph(
                id='graph-1-tabs',
                figure=fig
            )
        ])
    elif tab == 'vasicek':
        t, xt = vasicek(x0=x0,
                        a=x0,
                        b=mu,
                        sigma=sigma,
                        n_steps=n_steps,
                        dt=dt)
        df = pd.DataFrame.from_dict({"t": t,
                                     "Sim1": xt})
        fig = px.area(df, x="t", y="Sim1",
                      template=template)
        fig.update_xaxes(rangeslider_visible=True)
        content = html.Div([
            html.H3('Vasicek Model simulation'),
            dcc.Graph(
                id='graph-2-tabs',
                figure=fig
            )
        ])
    else:
        raise ValueError("Tab not defined")
    return content


if __name__ == '__main__':
    app.run_server()
