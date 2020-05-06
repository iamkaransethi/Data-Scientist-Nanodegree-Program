import os
from flask import Flask, flash, render_template, redirect, request, send_file, url_for
from numpy import squeeze
from utils import generate_random_name, is_allowed_file, make_thumbnail
from dog_app import breed_prediction

app = Flask(__name__)

#cmd : export SECRET_KEY="Your Secret Key"
app.config['SECRET_KEY'] = os.environ['SECRET_KEY'] 

# cmd : export UPLOAD_FOLDER="/tmp/fuzzvis"
app.config['UPLOAD_FOLDER'] = os.environ['UPLOAD_FOLDER']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']
        
        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # if the file is "legit"
        if image_file and is_allowed_file(image_file.filename):
            passed = False
            try:
                filename = generate_random_name(image_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(filepath)
                passed = make_thumbnail(filepath)
            except Exception:
                passed = False

            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                flash('An error occurred, try again.')
                return redirect(request.url)


@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    image_url = url_for('images', filename=filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predictions = breed_prediction(image_path)

    return render_template(
        'predict.html',
        plot_script=predictions,
        # plot_div=,
        image_url=image_url
    )


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500


@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
