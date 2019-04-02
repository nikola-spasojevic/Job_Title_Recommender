from flask import Flask,render_template,url_for,request
from flask_wtf import Form, FlaskForm
from wtforms import TextField
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)

cities = ["Bratislava",
          "Banska Bystrica",
          "Presov",
          "Povazska Bystrica",
          "zilina",
          "Kosice",
          "Ruzomberok",
          "Zvolen",
          "Poprad"]


class SearchForm(Form):
    autocomp = TextField('Insert City', id='city_autocomplete')


@app.route('/_autocomplete', methods=['GET'])
def autocomplete():
    return Response(json.dumps(cities), mimetype='application/json')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = SearchForm(request.form)
    return render_template("search.html", form=form)

if __name__ == '__main__':
	app.run()