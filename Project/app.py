from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
import pandas as pd
import re
from PIL import Image
from konlpy.tag import Okt
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
from clu_util import cluster_util
import numpy as np
from glob import glob
import cv2, os, random
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout


app = Flask(__name__)
app.debug = True

model_dogvscat = load_model(os.path.join(app.root_path, 'model/dogs_vs_cats-cnn.hdf5'))
model_face = load_model(os.path.join(app.root_path, 'model/mul_face2.hdf5'))

vgg = VGG16()
okt = Okt()
movie_lr = None
movie_lr_dtm = None
def load_movie_lr():
    global movie_lr, movie_lr_dtm
    movie_lr = joblib.load(os.path.join(app.root_path, 'model/movie_lr.pkl'))
    movie_lr_dtm = joblib.load(os.path.join(app.root_path, 'model/movie_lr_dtm.pkl'))

def tw_tokenizer(text):
    # 입력 인자로 들어온 text 를 형태소 단어로 토큰화 하여 list 객체 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

movie_nb = None
movie_nb_dtm = None
def load_movie_nb():
    global movie_nb, movie_nb_dtm
    movie_nb = joblib.load(os.path.join(app.root_path, 'model/movie_nb.pkl'))
    movie_nb_dtm = joblib.load(os.path.join(app.root_path, 'model/movie_nb_dtm.pkl'))

def nb_transform(review):
    stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    review = review.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    morphs = okt.morphs(review, stem=True)
    temp = ' '.join(morph for morph in morphs if not morph in stopwords)
    return temp

model_iris_lr = None
model_iris_svm = None
model_iris_dt = None
model_iris_deep = None
def load_iris():
    global model_iris_lr, model_iris_svm, model_iris_dt, model_iris_deep
    model_iris_lr = joblib.load(os.path.join(app.root_path, 'model/iris_lr.pkl'))
    model_iris_svm = joblib.load(os.path.join(app.root_path, 'model/iris_svm.pkl'))
    model_iris_dt = joblib.load(os.path.join(app.root_path, 'model/iris_dt.pkl'))
    model_iris_deep = load_model(os.path.join(app.root_path, 'model/iris.hdf5'))

model_diabetes_lr = None
model_diabetes_svm = None
model_diabetes_dt = None
model_diabetes_deep = None
def load_diabetes():
    global model_diabetes_lr, model_diabetes_svm, model_diabetes_dt, model_diabetes_deep
    model_diabetes_lr = joblib.load(os.path.join(app.root_path, 'model/diabetes_lr.pkl'))
    model_diabetes_svm = joblib.load(os.path.join(app.root_path, 'model/diabetes_svm.pkl'))
    model_diabetes_dt = joblib.load(os.path.join(app.root_path, 'model/diabetes_dt.pkl'))
    model_diabetes_deep = load_model(os.path.join(app.root_path, 'model/diabetes_deep.hdf5'))

@app.route('/')
def index():
    menu = {'home':True, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/regression', methods=['GET', 'POST'])
def regression():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
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
def sentiment():
    menu = {'home':False, 'rgrs':False, 'stmt':True, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('sentiment.html', menu=menu)
    else:
        res_str = ['부정', '긍정']
        review = request.form['review']
        # Logistic Regression 처리
        review_lr = re.sub(r"\d+", " ", review)
        review_lr_dtm = movie_lr_dtm.transform([review_lr])
        result_lr = res_str[movie_lr.predict(review_lr_dtm)[0]]
        # Naive Bayes 처리
        review_nb = nb_transform(review)
        review_nb_dtm = movie_nb_dtm.transform([review_nb])
        result_nb = res_str[movie_nb.predict(review_nb_dtm)[0]]
        # 결과 처리
        movie = {'review':review, 'result_lr':result_lr, 'result_nb':result_nb}
        return render_template('senti_result.html', menu=menu, movie=movie)



@app.route('/classification', methods=['GET', 'POST'])
def classification():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification.html', menu=menu)
    else:
        f = request.files['image']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + \
                    secure_filename(f.filename)
        f.save(filename)

        img = np.array(Image.open(filename).resize((224, 224)))
        yhat = vgg.predict(img.reshape(-1, 224, 224, 3))
        label_key = np.argmax(yhat)
        label = decode_predictions(yhat)
        label = label[0][0]
        return render_template('cla_result.html', menu=menu,
                                filename = secure_filename(f.filename),
                                name=label[1], pct='%.2f' % (label[2]*100))


@app.route('/classification_iris', methods=['GET', 'POST'])
def classification_iris():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification_iris.html', menu=menu)
    else:
        sp_names = ['Setosa', 'Versicolor', 'Virginica']
        slen = float(request.form['slen'])      # Sepal Length
        swid = float(request.form['swid'])      # Sepal Width
        plen = float(request.form['plen'])      # Petal Length
        pwid = float(request.form['pwid'])      # Petal Width
        test_data = np.array([slen, swid, plen, pwid]).reshape(1,4)
        species_lr = sp_names[model_iris_lr.predict(test_data)[0]]
        species_svm = sp_names[model_iris_svm.predict(test_data)[0]]
        species_dt = sp_names[model_iris_dt.predict(test_data)[0]]
        species_deep = sp_names[model_iris_deep.predict_classes(test_data)[0]]
        iris = {'slen':slen, 'swid':swid, 'plen':plen, 'pwid':pwid, 
                'species_lr':species_lr, 'species_svm':species_svm,
                'species_dt':species_dt, 'species_deep':species_deep}
        return render_template('cla_iris_result.html', menu=menu, iris=iris)

@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':True, 'user':False}
    if request.method == 'GET':
        return render_template('clustering.html', menu=menu)
    else:
        f = request.files['csv']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + \
                    secure_filename(f.filename)
        f.save(filename)
        ncls = int(request.form['K'])
        cluster_util(app, ncls, secure_filename(f.filename))
        img_file = os.path.join(app.root_path, 'static/images/kmc.png')
        mtime = int(os.stat(img_file).st_mtime)
        return render_template('clu_result.html', menu=menu, K=ncls, mtime=mtime)

@app.route('/member/<name>')
def member(name):
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':True}
    return render_template('user.html', menu=menu, name=name)

@app.route('/classification_cnn', methods=['GET', 'POST'])
def classification_cnn():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification_cnn.html', menu=menu)
        
    else:
        f = request.files['cd']
        filename = os.path.join(app.root_path, 'static/images/cnn/') + \
                    secure_filename(f.filename)
        f.save(filename)
        dogcats= []
        img = np.array(Image.open(filename))
        dogcat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dogcat = cv2.resize(dogcat, (96, 96))
        dogcat = image.img_to_array(dogcat)
        dogcats.append(dogcat)

        classes = ["cat","dog"]
        dogcats = np.asarray(dogcats).astype('float32')
        pct = np.asarray(dogcats).astype('float32') / 255
        print(dogcats.shape)
        kind = classes[model_dogvscat.predict_classes(dogcats)[0][0]]
        pct = model_dogvscat.predict(pct)
        pct = round(pct[0][0]*100, 3)
        return render_template('cla_cnn_result.html', menu=menu,
                                filename = secure_filename(f.filename),
                                pct=pct, kind=kind)



@app.route('/face', methods=['GET', 'POST'])
def face():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('face.html', menu=menu)
        
    else:
        f = request.files['face']
        filename = os.path.join(app.root_path, 'static/images/uploads/') + \
                    secure_filename(f.filename)
        f.save(filename)

        faces = []
        ROW, COL = 96, 96
        path = 'static/images/uploads'
        face_img = os.path.join(path, secure_filename(f.filename))
       # for face_img in glob(face_path):
        face = cv2.imread(face_img)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (ROW, COL))
        face = image.img_to_array(face)
        faces.append(face)

            

        kind = np.asarray(faces).astype('float32')
        classes = ['개상', '고양이상']

        kind = model_face.predict_classes(kind)

        pct = np.asarray(faces).astype('float32') / 255
        pct = model_face.predict(pct)
        kind = classes[kind[0][0]]
        pct = round(pct[0][0]*100, 3)

        return render_template('face_result.html', menu=menu,
                                filename = secure_filename(f.filename),
                                pct=pct, kind=kind)

@app.route('/classification_diabetes', methods=['GET', 'POST'])
def classification_diabetes():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('classification_diabetes.html', menu=menu)
    else:
        outcome = ['Negative','Positive']
        glucose = float(request.form['glucose'])      # Sepal Length
        bp = float(request.form['bp'])      # Sepal Width
        bmi = float(request.form['bmi'])      # Petal Length
        age = float(request.form['age'])      # Petal Width
        test_data = np.array([glucose, bp, bmi, age]).reshape(1,4)
        outcome_lr = outcome[model_diabetes_lr.predict(test_data)[0]]
        outcome_svm = outcome[model_diabetes_svm.predict(test_data)[0]]
        outcome_dt = outcome[model_diabetes_dt.predict(test_data)[0]]
        outcome_deep = outcome[model_diabetes_deep.predict_classes(test_data)[0][0]]
        diabetes = {'glucose': glucose, 'bp': bp, 'bmi': bmi, 'age': age, 
                'outcome_lr':outcome_lr, 'outcome_svm':outcome_svm,
                'outcome_dt':outcome_dt, 'outcome_deep':outcome_deep}
        return render_template('cla_diabetes_result.html', menu=menu, diabetes = diabetes)

if __name__ == '__main__':
    load_diabetes()
    load_movie_lr()
    load_movie_nb()
    load_iris()
    app.run(host='0.0.0.0')     # 외부 접속 허용시 host='0.0.0.0' 추가