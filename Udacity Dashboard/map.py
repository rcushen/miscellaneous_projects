import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

mapbox_access_token = 'pk.eyJ1IjoiY3VzaGVuciIsImEiOiJjandycTBiYnkwNTV3NDlwOTR1dTB0anp3In0.83YepiwDLwttwavOpB3sww'
colors = ['#cd222a', '#5ca363']

def generate_graph():
    address = 'https://data.melbourne.vic.gov.au/resource/vh2v-4nfs.json?$limit=5000'
    query_result = pd.read_json(address)


    sensor_data = query_result[['bay_id', 'st_marker_id', 'lat', 'lon', 'status']]
    sensor_data['status_color'] = sensor_data['status'].map({'Unoccupied':'#cd222a', 'Present':'#5ca363'})

    data = [
        go.Scattermapbox(
            lat=sensor_data['lat'].tolist(),
            lon=sensor_data['lon'].tolist(),
            mode='markers',
            #marker=go.scattermapbox.Marker(size=3, color=sensor_data['status']),
            marker=dict(size=3, color=sensor_data['status_color'].tolist()),
            text=['Melbourne'],
        )
    ]

    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=-37.814,
                lon=144.965
            ),
            pitch=0,
            zoom=13
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    return fig

#py.iplot(generate_graph(), filename='Melbourne Mapbox')
