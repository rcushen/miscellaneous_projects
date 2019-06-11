import pandas as pd
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def generate_graph():
    print('Downloading data...')
    address = 'https://data.melbourne.vic.gov.au/resource/vh2v-4nfs.json?$limit=5000'
    query_result = pd.read_json(address)
    print('...Data downloaded')

    sensor_data = query_result[['bay_id', 'st_marker_id', 'lat', 'lon', 'status']]
    counts = sensor_data.status.value_counts()

    fig = {
      "data": [
        {
          "values": counts.tolist(),
          "labels": counts.index.tolist(),
          "domain": {"column": 0},
          "name": "Available Spaces",
          "hoverinfo":"label+percent+value",
          "hole": .3,
          "type": "pie"
        }],
      "layout": {
            "title":"Available Parking Spaces",
        }
    }

    return fig

# py.iplot(generate_graph(), filename='donut')
