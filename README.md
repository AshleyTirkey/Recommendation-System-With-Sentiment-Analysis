# Recommendation-System-With-Sentiment-Analysis

A recommendation system that learns from customers as they interact online and suggest products that customers are likely to find valuable according to their needs. Recommendation systems filter data by using different algorithms and recommending the most relevant items to a wide variety of users. This is accomplished by taking into consideration past user behavior on a platform as a basis for products to recommend to the user as these products are estimated to be those most likely for the user to buy.

Libraries Used:
1. Pandas - used for analyzing, cleaning, exploring and manipulating data in the provided data sets.
2. Numpy - used to perform a wide variety of mathermatical operation on arrays. It adds powerfull data structures to Python that gaurantees efficient calculation with matrices and arrays, and supplies with large number of mathermatical functions.
3. Pickle - used for serializing and deserializing a Python object data structure i.e converts Python data structure into a byte stream to store it in a file/database. We used pickle in order to save our models for further use.
4. Sklearn - used to implement machile learning models and statistical modelling. With the use of this library we are able to implement models such as regression, classifciation, clustering and statistical tools for analyzing these models which are used to implement this system.
5. IPython - provide rich architecture for interactve computing.
6. Imblearn - provides tool for dealing with imbalanced data which we encounter in our project.


# Basic Flow
1. Building the product recommendation system with a defined User to User based approach

   Persons who have shared the same interests in past i.e who have liked the same products are likely to have similar interests in the future. In this way, similar users likely to have similar tastes. Say there 
   are two users, Ram and Shyam; Ram likes the set of products {P1, P2, P3} and Shyam likes the set of products {P1, P2, P4}. We can see already that there is good similarity between Ram and Shyam as they favor 
   products in common. But this also means that we can recommend product P4 to Ram as its liked already by Shyam.

2. Building sentiment analysis with the help of logistic regression

   Sentiment analysis is a natural language processing (NLP) technique that is used to determine whether data is positive, negative, or neutral. Based on the product reviews contained in our data, we can build 
   a Machine Learning model that gives the corresponding sentiments for each of the products contained in the data.

3. Then implementing the sentiment analysis into our previously made product recommendation system

   We see how the recommendations from [1] can be improved using sentiment analysis from [2] on the reviews given by users to the recommended products. Basically, the sentiment analysis 
   model helps us to fine-tune the recommendations that we get from the recommender system.

# Results

We generate a product’s ranking score with a formula (W1 x predicted rating of recommended product + W2 x normalized sentiment score on scale of 1–5 of recommended product) and use it to rank and sort product recommendations or filter them out depending on the number of recommendations we want to show. In this way, the higher the product’s ranking score, the better the product’s rating and review. A scale of 1–5 is used for the sentiment score as ratings also use the same scale — and usually, users give more weight to reviews than to ratings. So, we have assigned w1=1 and w2=2 (i.e., double weighting is given to reviews).
![image](https://github.com/AshleyTirkey/Recommendation-System-With-Sentiment-Analysis/assets/87265518/d6043cfa-1feb-4cb5-94e2-ea9a456ba538)
