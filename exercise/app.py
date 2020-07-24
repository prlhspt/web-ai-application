from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('1_bootstrap.html')

@app.route('/typo', methods=['GET', 'POST'])
def typo():
    return render_template('4_typography.html')

@app.route("/iris", methods = ['GET', 'POST'])
def iris():
    if request.method == 'GET':
        return render_template('1_bootstrap.html')
    else:
        slen_ = float(request.form['slen'])
        plen_ = float(request.form['plen'])
        swid_ = float(request.form['swid'])
        pwid_ = float(request.form['pwid'])
        species_ = request.form['species']
        comment_ = request.form['text']
        return render_template('14.iris_result.html' , slen = slen_, plen = plen_, swid = swid_, pwid = pwid_, species = species_, comment = comment_)

@app.route('/project')
def project():
    return render_template('test_page.html')

@app.route('/hello')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html')

if __name__ == '__main__':
    app.run(debug=True)

