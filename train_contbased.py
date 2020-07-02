# Importing packages and Data
import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
# Importing data

movies = pd.read_csv('movies.csv',sep = ',',delimiter=',')
ratings = pd.read_csv('ratings.csv')
movies.dropna(inplace=True)
def datapreprocessing():
    # spliting the genres
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
  
    return movies[:30000]
def content_model():
    # initializing the empty list of recommended movies
    data = datapreprocessing()
    # instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data['keyWords'])
    indices = pd.Series(data['title'])
    return pickle.dump(cosine_similarity(count_matrix, count_matrix), open('matrix.pkl','wb'))
content_model()
    
    
