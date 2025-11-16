from flask import Flask, render_template, request
from PIL import Image
import io
from network import classify_image  # CNN classification function

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Check if file is in request
    if 'image' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['image']

    if file.filename == '':
        return 'No selected file!', 400

    # Open image
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return f"Error opening image: {str(e)}", 400

    # Classify
    try:
        label = classify_image(img)
    except Exception as e:
        return f"Error during classification: {str(e)}", 500

    return f"<h2>Classification: {label}</h2>"


if __name__ == '__main__':
    # Accessible locally
    app.run(debug=False, use_reloader=False)
