import numpy as np 
import pandas as pd
from itertools import groupby
from sklearn_crfsuite.metrics import flat_classification_report
import tensorflow
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical #One-Hot encode
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF 
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from scripts.func import SentenceGetter

def mtrain(array_text, model_path):
    """To train NER model
    Input: array_text = [{'word':'bus','ent_tag':'THING','sentence':0}, {...}, ...]
    Output: zipped model and uploaded to S3"""

    # variables
    var = {
        'max_len': 75,
        'emb_dim': 40,
        'epochs': 20,
        'batch_size': 512,
        'opt': 'rmsprop'
    }

    # prepare words, tags and id
    l_sentences = [i['sentence'] for i in array_text]

    l_words = [i['word'] for i in array_text]
    words = list(set(l_words))
    n_words = len(words)

    l_tags = [i['ent_tag'] for i in array_text]
    tags = list(set(l_tags))
    n_tags = len(tags)

    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx['UNK'] = 1
    word2idx['PAD'] = 0
    idx2word = {i: w for w, i in word2idx.items()}

    tag2idx = {t: i+1 for i, t in enumerate(tags)}
    tag2idx['PAD'] = 0
    idx2tag = {i: w for w, i in tag2idx.items()}

    # sentence getter
    df = pd.DataFrame({'word':l_words, 'ent_tag':l_tags, 'sentence':l_sentences})
    getter = SentenceGetter(df)
    sentences = getter.sentences

    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=var['max_len'], sequences=X, padding='post', value=word2idx['PAD'])
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=var['max_len'], sequences=y, padding='post', value=tag2idx['PAD'])
    y = [to_categorical(i, num_classes=n_tags+1) for i in y]

    # embedding layer
    input_l = Input(shape=(var['max_len'], ))
    model = Embedding(input_dim=n_words + 2,
                      output_dim=var['emb_dim'],
                      input_length=var['max_len'],
                      mask_zero=False)(input_l)

    # Bi-LSTM layer
    model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(50, activation='relu'))(model)

    # CRF
    crf = CRF(n_tags + 1)
    out = crf(model)

    # model compile and fit
    model = Model(input_l, out)
    model.compile(optimizer=var['opt'], loss=crf.loss_function, metrics=[crf.accuracy])
    history = model.fit(X, np.array(y),
                        batch_size=var['batch_size'],
                        epochs=var['epochs'],
                        validation_split=0.1,
                        verbose=2)
    model.save(model_path)

    return word2idx, tag2idx

def mrun(array_text, model_path, word2idx, tag2idx):
    """To load NER model
    Input: model object
    Output: a list of entities"""

    # variables
    var = {
        'test_size': 0.2,
        'max_len': 75,
        'emb_dim': 40,
        'epochs': 20,
        'batch_size': 512,
        'opt': 'rmsprop'
    }

    # imoprt training data
    l_sentences = [i['sentence'] for i in array_text]
    l_words = [i['word'] for i in array_text]
    words = list(set(l_words))
    n_words = len(words)
    l_tags = [i['ent_tag'] for i in array_text]
    n_tags = len(tag2idx.keys())

    # preprocessing step
    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: w for w, i in tag2idx.items()}

    df = pd.DataFrame({'word':l_words, 'ent_tag':l_tags, 'sentence':l_sentences})
    getter = SentenceGetter(data)
    sentences = getter.sentences

    test = []
    for s in sentences:
        t = []
        for pair in s:
            if pair[0] in word2idx:
                t.append(pair)
            else:
                lst = list(pair)
                lst[0] = "UNK"
                pair = tuple(lst)
                t.append(pair)
        test.append(t)

    X = [[word2idx[w[0]] for w in s] for s in test]
    X = pad_sequences(maxlen=var['max_len'], sequences=X, padding='post', value=word2idx['PAD'])
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=var['max_len'], sequences=y, padding='post', value=tag2idx['PAD'])
    y = [to_categorical(i, num_classes=n_tags+1) for i in y]

    # load model
    model = keras.models.load_model(model_path, custom_objects={
        'CRF': CRF,
        'crf_loss': crf_loss,
        'crf_viterbi_accuracy': crf_viterbi_accuracy
    })

    # prediction
    pred_cat = model.predict(X)
    y_pred = np.argmax(pred_cat, axis=-1)
    y_pred_tag = [[idx2tag[i] for i in row] for row in y_pred]

    # get NER entities
    l_X_id = [item for sublist in X for item in sublist]
    l_sent = [idx2word[X_id] for X_id in l_X_id]
    l_entities = [item for sublist in y_pred_tag for item in sublist]

    l_join_ent = []
    netagged_words = list(zip(l_sent, l_entities))
    for tag, chunk in groupby(netagged_words, lambda x:x[1]):
        if tag != 'O' and tag != 'PAD' and tag == array_text['entities']:
            join_ent = "".join(w for w, t in chunk)
            l_join_ent.append(join_ent)
    s_join_ent = list(set(l_join_ent))
    return l_join_ent, s_join_ent
