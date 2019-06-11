from flask import Flask
from flask import render_template, jsonify

import map

app = Flask(__name__, static_url_path='/static')

fig = map.generate_graph()

@app.route('/')
def index():
    return render_template('home.html', fig=fig)
