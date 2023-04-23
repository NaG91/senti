# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 07:49:41 2023

@author: Nagarajan
"""

import pandas as pd
from tqdm import tqdm
df = pd.read_csv(r'D:\learn\senti\data\one.csv',sep='|')
df1 = pd.read_csv(r'D:\learn\senti\data\two.csv',sep='|')

df = pd.concat([df, df1])
del df1

from flair.models import TextClassifier
from flair.data import Sentence
sia = TextClassifier.load('en-sentiment')
def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return "Good"
    elif "NEGATIVE" in str(score):
        return "Bad"
    else:
        return "neu"
list(df)

tqdm.pandas()

df["sentiment"] = df["Description"].progress_apply(flair_prediction)

len(df[df['sentiment'] == df['Is_Response']]) / len(df)

a = df[df['sentiment'] != df['Is_Response']]
df['Browser_Used'].value_counts()/len(df)
a['Browser_Used'].value_counts()/len(a)


