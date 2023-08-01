# Libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import display

# Input Data
ratings = pd.read_csv('D:/college/Projects/PyCham Projects/Sentiment analysis/product_ratings_final.csv',
                      encoding='latin-1')
display(ratings.sample(n=5, random_state=42))


# Data Prep
def apply_pivot(df, fillby=None):
    if fillby is not None:
        return df.pivot_table(index='userId', columns='prod_name', values='rating').fillna(fillby)
    return df.pivot_table(index='userId', columns='prod_name', values='rating')


train, test = train_test_split(ratings, test_size=0.30, random_state=42)
test = test[test.userId.isin(train.userId)]

df_train_pivot = apply_pivot(df=train, fillby=0)
df_test_pivot = apply_pivot(df=test, fillby=0)

dummy_train = train.copy()
dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x >= 1 else 1)
dummy_train = apply_pivot(df=dummy_train, fillby=1)

dummy_test = test.copy()
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x >= 1 else 0)
dummy_test = apply_pivot(df=dummy_test, fillby=0)

var = df_train_pivot[(df_train_pivot['0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest'] != 0) |
                     (df_train_pivot['4C Grated Parmesan Cheese 100% Natural 8oz Shaker'] != 0)]
display(var.sample(n=7))

# Calculate Similarity

mean = np.nanmean(apply_pivot(df=train), axis=1)
df_train_substracted = (apply_pivot(df=train).T - mean).T
# making rating 0 where user has not given any rating
df_train_substracted.fillna(0, inplace=True)
# creating user similarity matrix
user_correlation = 1 - pairwise_distances(df_train_substracted, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
# convert user correlation into dataframe
user_correlation_df = pd.DataFrame(user_correlation)
user_correlation_df['userId'] = df_train_substracted.index
user_correlation_df.set_index('userId', inplace=True)
user_correlation_df.columns = df_train_substracted.index.tolist()

var1 = user_correlation.shape, df_train_pivot.shape
display(var1)

# Predict Ratings (User-User)

user_predicted_ratings = np.dot(user_correlation, df_train_pivot)
# finding product not rated by user
user_final_rating = np.multiply(user_predicted_ratings, dummy_train)


# Finding top N recommendations for user

def find_top_recommendations(pred_rating_df, userid, topn):
    recommendation = pred_rating_df.loc[userid].sort_values(ascending=False)[0:topn]
    recommendation = pd.DataFrame(recommendation).reset_index().rename(columns={userid: 'predicted_ratings'})
    return recommendation


user_input = str(input("Enter your user id"))
recommendation_user_user = find_top_recommendations(user_final_rating, user_input, 5)
recommendation_user_user['userId'] = user_input

print("Recommended products for user id:{} as below".format(user_input))
display(recommendation_user_user)
print("Earlier rated products by user id:{} as below".format(user_input))
display(train[train['userId'] == user_input].sort_values(['rating'], ascending=False))

# Evaluation User-User

# filtering user correlation for user which are in test
user_correlation_test_df = user_correlation_df[user_correlation_df.index.isin(test.userId)]
user_correlation_test_df = user_correlation_test_df[list(set(test.userId))]

# get test user predicted ratings
test_user_predicted_ratings = np.dot(user_correlation_test_df, df_test_pivot)
test_user_predicted_ratings = np.multiply(test_user_predicted_ratings, dummy_test)

# get NAN where user never rated as it shouldn't contribute in RMSE
test_user_predicted_ratings = test_user_predicted_ratings[test_user_predicted_ratings > 0]
scaler = MinMaxScaler(feature_range=(1, 5))
scaler.fit(test_user_predicted_ratings)
test_user_predicted_ratings = scaler.transform(test_user_predicted_ratings)

total_nan_nan = np.count_nonzero(~np.isnan(test_user_predicted_ratings))
rmse = (np.sum(np.sum((apply_pivot(df=test) - test_user_predicted_ratings) ** 2)) / total_nan_nan) ** 0.5
print(rmse)

# Saving the model for sentimental analysis
filename = 'analysis_model.pkl'
pickle.dump(user_final_rating, open(filename, 'wb'))
