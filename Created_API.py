import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

df_comb = pd.read_csv("extracted_features.csv")

count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(df_comb["soup"])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df_comb = df_comb.reset_index()

indices = pd.Series(df_comb.index, index=df_comb["title"]).drop_duplicates()


class requestbody(BaseModel):
    movie_name : str

@app.post('/predict')
def recommend(data:requestbody):
    def get_recommendations(title, cosine_sim=cosine_sim2):
        idx = indices[title]
        similarity_scores = list(enumerate(cosine_sim[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:11]
        # (a, b) where a is id of movie, b is similarity_scores
        movies_indices = [ind[0] for ind in similarity_scores]
        movies = df_comb["title"].iloc[movies_indices]
        return movies

    recommendation = get_recommendations(data.movie_name, cosine_sim2)

    return list(recommendation)



