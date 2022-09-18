from flask import Flask , render_template , request , Response, send_file, jsonify
from PIL import Image
import os
import classify as cl



app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "C:/Users/yourc/Desktop/FlaskDemo/Demo/101_ObjectCategories/accordion/"


@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            return render_template("index.html", uploaded_image=image.filename)
    return render_template("index.html")


@app.route('/uploads/<filename>')
def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config["IMAGE_UPLOADS"], filename)

@app.route('/prediction'):

folder_path = cl.classify_images(send_uploaded_file(filename=''))


if __name__ =="__main__":
    app.run()