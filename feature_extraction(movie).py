import pandas as pd
import numpy as np
from ast import literal_eval
pd.set_option("display.max_columns",200)

df_credits = pd.read_csv("tmdb_5000_credits.csv")
df_movies = pd.read_csv("tmdb_5000_movies.csv")

df_credits = df_credits.drop(columns=["title"])
df_comb = pd.merge(df_credits,df_movies,left_on="movie_id",right_on="id",how="outer")

features = ["cast", "crew", "keywords", "genres"]
for feature in features:
    df_comb[feature] = df_comb[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

df_comb["director"] = df_comb["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    df_comb[feature] = df_comb[feature].apply(get_list)

def clean_data(row):
    if isinstance(row, list):
        return [str.lower(i.replace(" ", "")) for i in row]
    else:
        if isinstance(row, str):
            return str.lower(row.replace(" ", ""))
        else:
            return ""

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    df_comb[feature] = df_comb[feature].apply(clean_data)

def combine_feature(features):
    return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['director'] + ' ' + ' '.join(features['genres'])

df_comb["soup"] = df_comb.apply(combine_feature, axis=1)

df_comb.to_csv("extracted_features.csv",index=False)
