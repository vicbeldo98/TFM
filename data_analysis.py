import pandas as pd

df_ratings = pd.read_csv("data/raw/ml-100K/ratings.csv")

# Borrar las películas que han aparecido menos de 5 veces

n_reviews_per_movie = df_ratings.groupby("movieId")["userId"].count()

new_movies = []

for i in n_reviews_per_movie.iteritems():
    if i[1] >= 10:
        new_movies.append(i[0])

print(f"Hay un total de {len(new_movies)} películas que han recibido al menos 5 reviews")

new_df = df_ratings[df_ratings["movieId"].isin(new_movies)]

# Borrar los usuarios que hayan puntuado menos de 5 películas

n_reviews_per_user = new_df.groupby("userId")["movieId"].count()

new_users = []

for i in n_reviews_per_user.iteritems():
    if i[1] >= 10:
        new_users.append(i[0])

print(f"Hay un total de {len(new_users)} usuarios que han puntuado al menos 5 películas")

new_df = new_df[new_df["userId"].isin(new_users)]

new_df.to_csv("data/raw/ml-100K/ratings_clean.csv")

# Make sure dataset is correct

df_ratings = pd.read_csv("data/raw/ml-100K/ratings_clean.csv")

n_reviews_per_movie = df_ratings.groupby("movieId")["userId"].count()

new_movies = []

for i in n_reviews_per_movie.iteritems():
    if i[1] < 10:
        print("BREAK!!!!!")
        break

reviews_per_user = new_df.groupby("userId")["movieId"].count()

new_users = []

for i in n_reviews_per_user.iteritems():
    if i[1] < 10:
        print("BREAK!!!!!")
        break
nusers = df_ratings["userId"].nunique()
nmovies = df_ratings["movieId"].nunique()
print(f"Number of total users: df_ratings: {nusers}")
print(f"Number of total movies: {nmovies}")





