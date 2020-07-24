from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    menu = {'home':True, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/regression', methods=['GET', 'POST'])
def rgrs():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('base.html', menu=menu)
    else:
        return render_template('base.html', menu=menu)

@app.route('/iris', methods=['GET', 'POST'])
def rgrs_result():
    menu = {'home':False, 'rgrs':True, 'stmt':False, 'clsf':False, 'clst':False, 'user':False}
    if request.method == 'GET':
        return render_template('base.html', menu=menu)
    else:

        slen = float(request.form['slen'])
        plen = float(request.form['plen'])
        pwid = float(request.form['pwid'])
        species = float(request.form['species'])
        swid = (0.59652908 * slen + -0.56653158 * plen +  0.60342179 * pwid + -0.07734281 * species + 1.0430890827987302)
        
        slen = round(slen, 3)
        plen = round(plen, 3)
        pwid = round(pwid, 3)
        swid = round(swid, 3)
        species = int(species)

        iris={'slen':slen, 'plen': plen, 'pwid':pwid, 'species':species, 'swid':swid}
    
        return render_template('regression_result.html', menu=menu, iris=iris)# slen = slen_, plen = plen_, swid = swid_, pwid = pwid_, species = species_)

@app.route('/sentiment')
def stmt():
    menu = {'home':False, 'rgrs':False, 'stmt':True, 'clsf':False, 'clst':False, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/classification')
def clsf():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':True, 'clst':False, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/clustering')
def clst():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':True, 'user':False}
    return render_template('home.html', menu=menu)

@app.route('/user')
def user():
    menu = {'home':False, 'rgrs':False, 'stmt':False, 'clsf':False, 'clst':False, 'user':True}
    return render_template('home.html', menu=menu)


if __name__ == '__main__':
    app.run(debug=True)

