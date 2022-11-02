import requests
import json
import pandas as pd

df = pd.read_csv("tmdb_5000_movies.csv")
from flask import Flask,request,render_template


movies_title = sorted(list(df["original_title"]))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def recomend():

    movie_name = request.form.get('Movie_Titles')

    out = requests.post('http://127.0.0.1:8000/predict', data=json.dumps({'movie_name': movie_name}))

    return render_template('index.html', prediction_text=out.json(),name=f"Movies Similar to {movie_name} :-",movies_title=movies_title)


if __name__ == '__main__':
    app.run(debug=True)

