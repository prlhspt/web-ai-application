from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import joblib
from konlpy.tag import Okt
import re
import os
from tensorflow import keras
from keras.models import load_model
from keras.applications.vgg16 import VGG16, decode_predictions
import numpy as np
from PIL import Image

app = Flask(__name__)
app.debug=True

stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
tfidf_vector = None
model_lr = None
dtmvector = None
model_nb = None
vgg = None
model_iris_deep = None
model_iris_lr = None
model_iris_svm = None
model_iris_dt = None

def load_lr():
    global tfidf_vector, model_lr
    tfidf_vector = joblib.load(os.path.join(app.root_path, 'model/movie_lr_dtm.pkl'))
    model_lr = joblib.load(os.path.join(app.root_path, 'model/movie_lr.pkl'))

def tw_tokenizer(text):
    # 입력 인자로 들어온 text 를 형태소 단어로 토큰화 하여 list 객체 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

def lr_transform(review):
    review = re.sub(r"\d+", " ", review)
    test_dtm = tfidf_vector.transform([review])
    return test_dtm

def load_nb():
    global dtmvector, model_nb
    dtmvector = joblib.load(os.path.join(app.root_path, 'model/movie_nb_dtm.pkl'))
    model_nb = joblib.load(os.path.join(app.root_path, 'model/movie_nb.pkl'))

def nb_transform(review):
    review = review.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    morphs = okt.morphs(review, stem=True) # 토큰화
    test = ' '.join(morph for morph in morphs if not morph in stopwords)
    test_dtm = dtmvector.transform([test])
    return test_dtm

def load_iris():
    global model_iris_deep, model_iris_lr, model_iris_svm, model_iris_dt
    model_iris_deep = load_model(os.path.join(app.root_path, 'model/iris.hdf5'))
    model_iris_lr = joblib.load(os.path.join(app.root_path, 'model/iris_lr.pkl'))
    model_iris_svm = joblib.load(os.path.join(app.root_path, 'model/iris_svm.pkl'))
    model_iris_dt = joblib.load(os.path.join(app.root_path, 'model/iris_dt.pkl'))

@app.route('/')
def index():
    menu = {'home':True, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':False, 'profile':False}
    return render_template('home.html', menu=menu)

@app.route('/regression', methods=['GET', 'POST'])
def rgrs():
    menu = {'home':False, 'regression':True, 'senti':False, 'classification':False, 'clustering':False}
    if request.method == 'GET':
        return render_template('regression.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        sp = int(request.form['species'])       # Species
        species = sp_names[sp]
        swid = 0.63711424 * slen - 0.53485016 * plen + 0.55807355 * pwid - 0.12647156 * sp + 0.78264901
        swid = round(swid, 4)
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 'species':species}
        return render_template('reg_result.html', menu=menu, iris=iris)

@app.route('/sentiment', methods=['GET', 'POST'])
def stmt():
    menu = {'home':False, 'rgrs':False, 'stmt':True, 'clsf':False, 'clst':False, 'user':False, 'profile':False}
    if request.method == 'GET':
        return render_template('sentiment.html', menu=menu)
    else:
        review = request.form['review']
        review_dtm_lr = lr_transform(review)
        lr_result = model_lr.predict(review_dtm_lr)[0]
        review_dtm_nb = nb_transform(review)
        nb_result = model_nb.predict(review_dtm_nb)[0]
        lr = '긍정' if lr_result else '부정'
        nb = '긍정' if nb_result else '부정'
        movie = {'review':review, 'lr':lr, 'nb':nb}

        return render_template('sentiment_result.html', menu=menu, movie=movie)

@app.route('/classification')
def clsf():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False, 'profile':False}
    return render_template('home.html', menu=menu)

@app.route('/classification_iris', methods=['GET', 'POST'])
def clsf_iris():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False, 'profile':False}
    if request.method == 'GET':
        return render_template('classification_iris.html', menu=menu)
    else:
        sp_names=['Setosa', 'Vergicolor', 'Virginica']
        slen = float(request.form['slen'])
        plen = float(request.form['plen'])
        swid = float(request.form['swid'])
        pwid = float(request.form['pwid'])

        test_data=np.array([slen, plen, swid, pwid]).reshape(1, 4)

        species_lr = sp_names[model_iris_lr.predict(test_data)[0]]
        species_svm = sp_names[model_iris_svm.predict(test_data)[0]]
        species_dt = sp_names[model_iris_dt.predict(test_data)[0]]
        species_deep = sp_names[model_iris_deep.predict_classes(test_data)[0]]

        iris={'slen':slen, 'plen': plen, 'pwid':pwid, 'swid':swid, 'species_lr':species_lr, 'species_svm':species_svm, "species_dt":species_dt, "species_deep":species_deep}
    
        return render_template('clsf_result.html', menu=menu, iris=iris)

@app.route('/clustering')
def clst():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':True, 'user':False, 'profile':False}
    return render_template('home.html', menu=menu)

@app.route('/user')
def user():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':True, 'profile':False}
    return render_template('home.html', menu=menu)


@app.route('/profile')
def profile():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':False, 'profile':True}
    return render_template('profile.html', menu=menu)

if __name__ == '__main__':
    load_movie_lr()
    load_movie_nb()
    load_iris()
    app.run()

