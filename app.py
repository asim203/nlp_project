from unicodedata import name
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import pickle
import re
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
# spacy for lemmatization
import spacy

import requests
# from flask_cors import CORS
from flask import Flask, request, jsonify
app = Flask(__name__)
# CORS(app)

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','know','get','say','go',
                    'well'])
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

with open('id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)
f.close()

lda_model = gensim.models.ldamodel.LdaModel.load('model/my_model.model')

def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
    return email

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'

@app.route('/gettopics', methods=['POST'])
def gettopics():
    raw_message=request.form['data']
    data_body=[parse_raw_message(raw_message)['body']]
    wrds = list(sent_to_words(data_body))
    nostop=remove_stopwords(wrds)
    lemet= lemmatization(nostop, allowed_postags=['NOUN', 'ADJ', 'VERB'])
    corpus = [id2word.doc2bow(text) for text in lemet]
    top_topics=lda_model.get_document_topics(corpus[0], minimum_probability=0.1)
    tpcs=[]
    for x,y in top_topics:
        all_topics=lda_model.show_topic(x)
        for q,_ in all_topics:
            tpcs.append(q)
    return jsonify(tpcs)


if __name__ == "__main__":
    app.run()