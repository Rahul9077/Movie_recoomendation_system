import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_comb = pd.read_csv("extracted_features.csv")

count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(df_comb["soup"])

print(count_matrix.shape)
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim2.shape)

df_comb = df_comb.reset_index()
# indices = pd.Series(df_comb.index, index=df_comb['title'])
indices = pd.Series(df_comb.index, index=df_comb["title"]).drop_duplicates()

# print(indices.head())

def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores= sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores= similarity_scores[1:11]
    # (a, b) where a is id of movie, b is similarity_scores
    movies_indices = [ind[0] for ind in similarity_scores]
    movies = df_comb["title"].iloc[movies_indices]
    return movies


name = input("Enter the movie name")
recommendation = get_recommendations(name, cosine_sim2)
print("Movies Similar to ",name)
print(recommendation)
