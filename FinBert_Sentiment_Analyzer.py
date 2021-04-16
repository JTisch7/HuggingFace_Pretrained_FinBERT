# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 13:05:57 2021

@author: Jonathan
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import time
from tqdm import tqdm

#get FinBert from Hugging Face
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert', from_pt=True)
model = TFBertForSequenceClassification.from_pretrained("ProsusAI/finbert", from_pt=True)

#get news dataset
dfNews = pd.read_csv('data/newsAMZN2yrs.csv', index_col=0)
dfNews = dfNews[:50]

#tokenize text and get sentiment from FinBert model
def sentScore(dfNews):
    df = dfNews.copy()
    
    df['pos'] = 0
    df['neg'] = 0
    df['neut'] = 0
    df['pred'] = 0
    
    MAX_LEN = 160
    class_names = ['positive', 'negative', 'neutral']
    i = 0
    
    start = time.time()
    for sentence in tqdm(dfNews['title']):
        encoded_new = tokenizer.encode_plus(
                                            sentence,                      
                                            add_special_tokens = True,      
                                            max_length = MAX_LEN,             
                                            padding = True,
                                            return_attention_mask = True,     
                                            return_tensors = 'tf',            
                                            )
        
        input_idst = (encoded_new['input_ids'])
        attention_maskst = (encoded_new['attention_mask'])
        
        new_test_output = model(input_idst, token_type_ids=None, 
                              attention_mask=attention_maskst)
        
        predicted = new_test_output[0].numpy()
        flat_predictions = np.concatenate(predicted, axis=0)
        new_predictions = np.argmax(flat_predictions).flatten()
        
        df.loc[i,'pos'] = predicted[0][0]
        df.loc[i,'neg'] = predicted[0][1]
        df.loc[i,'neut'] = predicted[0][2] 
        df.loc[i,'pred'] = class_names[new_predictions[0]] 
        i += 1
             
    finish = time.time()
    totalTime = (finish - start)/60
    print(totalTime)
    return df

dfSent = sentScore(dfNews)


