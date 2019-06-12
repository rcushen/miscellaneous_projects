from flask import Flask
from flask import render_template, jsonify

import map

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    fig = map.generate_graph()
    return render_template('home.html', fig=fig)
