
###########################################
# PROJECT: Hybrid Recommender System
###########################################

# Estimating for the user whose ID is given, using item-based and user-based recomennder methods.


###########################################
# Task 1: Preparing the Data
###########################################

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)

# Step 1: Reading the Movie and Rating datasets
movie = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
movie.head()
movie.shape

# Data set containing UserID, movie name, movie vote and time information
rating = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/rating.csv')
rating.head()
rating.shape
rating["userId"].nunique()


# Step 2: Adding the names and genre of the movies to the rating dataset using the movie movie set.
# Only the id's of the movies that the users in the rating have voted.
# We add the movie names and genre of the ids from the movie dataset.
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape


# Step 3: Calculate the total number of votes for each movie. We exclude movies with less than 1000 votes from the data set.
# We calculate how many people voted for each movie.
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts

# We keep the names of movies with less than 1000 votes in rare_movies.
# And we subtract from the dataset
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape


# We create a pivot table for the dataframe in the index, where userIDs, movie names in the columns and ratings as values.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()


# Let's functionalize all the operations above
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


###########################################
# Determining the Movies Watched by the User to Suggest
###########################################

# Step 1: A random user id is chosen.
random_user = 108170

# Step 2: A new dataframe named random_user_df is created, consisting of observation units of the selected user.
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape

# Step 3: The movies voted by the selected user are assigned to a list called movies_watched.
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
movies_watched

movie.columns[movie.notna().any()].to_list()

###########################################
# Task 3: Accessing Data and Ids of Other Users Watching the Same Movies
###########################################

# Step 1: Select the columns of the movies watched by the selected user from user_movie_df and create a new dataframe named movies_watched_df.
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

# Step 2: Create a new dataframe named user_movie_count for each user that contains the number of movies watched by the selected user.
# And we create a new df.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head(5)

# Step 3: We consider those who watch 60 percent or more of the movies voted by the selected user as similar users.
# We created a list called users_same_movies from the ids of these users.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies)



#############################################
# Determining the Users to be Suggested and Most Similar Users
###########################################

# Step 1: Filtering the movies_watched_df dataframe to find the ids of the users that are similar to the selected user in the user_same_movies list.
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

# Step 2: Creating a new corr_df dataframe where users' correlations with each other will be found.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

#corr_df[corr_df["user_id_1"] == random_user]



# Step 3: Creating a new dataframe named top_users, filtering out users with high correlation (over 0.65) with the selected user.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()

# Step 4: Merge top_users dataframe with rating dataset
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()



###########################################
# Task 5: Calculating Weighted Average Recommendation Score and Keeping Top 5 Movies
###########################################

# Step 1: Creating a new variable named weighted_rating, which is the product of each user's corr and rating.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Step 2: Create a new recommendation_df, which contains the movie id and the average value of the weighted ratings of all users for each movie.
# dataframe creation.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Step 3: Step3: Select movies with weighted rating greater than 3.5 in recommendation_df and sort by weighted rating.
# Save the first 5 observations as movies_to_be_recommend.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


# Step 4: Bringing the names of the 5 recommended movies.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)



#############################################
# Item-Based Recommendation
#############################################

# Making item-based suggestions based on the name of the movie that the user last watched and gave the highest rating.
user = 108170

# Step 1: Reading the movie,rating datasets.
movie = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

# Step 2: Obtaining the id of the movie with the most up-to-date score from the movies that the user to be recommended gives 5 points.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Step 3 : Filtering the user_movie_df dataframe created in the User based recommendation section according to the selected movie id.
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Step 4: Using the filtered dataframe, find the correlation of the selected movie with the other movies and rank them.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# Function that performs the last two steps
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Step 5: Giving the first 5 movies as suggestions except the selected movie itself
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
#1 to 6th. 0 has the movie itself. We left him out.
movies_from_item_based[1:6].index


# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,
# The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']



