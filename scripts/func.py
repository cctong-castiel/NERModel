import numpy as np 
import pandas as pd 
import re
import hashlib

def get_digest(file_path):
    h = hashlib.sha256()

    with open(file_path, 'rb') as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()

def mark_sent(df):
    '''To mark sentence number from 0 till the end of full stop'''
    num = 0
    sent_lab = []
    for index in range(len(df)):
        if index == 0:
            sent_lab.append("Sentence:{}".format(num))
        else:
            if re.search(r'[。?!]', str(df['word'][index-1])):
                num += 1
                sent_lab.append("Sentence:{}".format(num))
            else:
                sent_lab.append("Sentence:{}".format(num))
    df['sentence'] = sent_lab
    return df

class SentenceGetter(object):
    '''Class to Get the sentence in this format:
       [(Token_1, POS_1, Tag_1), ..., (Token_n, POS_n, Tag_n)]'''
    
    def __init__(self, data):
        '''Args: data is the pandas.DataFrame which contains the above dataset'''
        self.n_sent = 1
        self.data = data
        self.empty = False
        '''
        agg_func = lambda s: [(w,p,t) for w,p,t in zip(s['word'].values.tolist(),
                                                        #s['POS'].value.tolist()
                                                        s['ent_tag'].values.tolist())]'''
        agg_func = lambda s: [(w,t) for w,t in zip(s['word'].values.tolist(),
                                                   s['ent_tag'].values.tolist())]
        self.grouped = self.data.groupby("sentence").apply(agg_func)
        print(self.grouped)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        '''Return one sentence'''
        try:
            s = self.grouped['sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None
            