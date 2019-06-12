import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

pd.options.mode.chained_assignment = None

mapbox_access_token = 'pk.eyJ1IjoiY3VzaGVuciIsImEiOiJjandycTBiYnkwNTV3NDlwOTR1dTB0anp3In0.83YepiwDLwttwavOpB3sww'
colors = ['#cd222a', '#5ca363']

def generate_graph():
    address = 'https://data.melbourne.vic.gov.au/resource/vh2v-4nfs.json?$limit=5000'
    query_result = pd.read_json(address)

    desired_variables = ['bay_id', 'st_marker_id', 'lat', 'lon', 'status']
    sensor_data = query_result[desired_variables]
    color_mapping = {'Unoccupied':'#cd222a', 'Present':'#5ca363'}
    sensor_data['status_color'] = sensor_data['status'].map(color_mapping)

    fig = {
        'data':[
            {
             'type':'scattermapbox',
             'lat':sensor_data['lat'].tolist(),
             'lon':sensor_data['lon'].tolist(),
             'mode':'markers',
             'marker': {
                'size':14,
                'color':sensor_data['status_color'].tolist(),
                },
            'text':['Melbourne']
            }
        ],
        'layout': {
            'hovermode':'closest',
            'mapbox': {
                'bearing':0,
                'center': {
                'lat':-37.814,
                'lon':144.965
            },
            'pitch':0,
            'zoom':13
            }
        },
        'token': mapbox_access_token
        }

    return fig
