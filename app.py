import os
import re
import uuid

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image, UnidentifiedImageError

import CNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
BACKGROUND_INDEX = 4          # class "Background_without_leaves"
LOW_CONFIDENCE = 60.0         # below this %, the result page shows a caution

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------- data
def clean(value):
    """CSV cell -> stripped string, or None when empty / NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def https(url):
    """Hotlinked images on http:// are blocked as mixed content on https sites."""
    url = clean(url)
    if url and url.startswith('http://'):
        url = 'https://' + url[len('http://'):]
    return url


def split_lines(text):
    # some CSV rows glue sentences together ("...Wisconsin.Bacterial spot..."); add the missing space
    text = re.sub(r'([a-z]{2}[.!?])([A-Z])', r'\1 \2', clean(text) or '')
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_catalog():
    diseases = pd.read_csv(os.path.join(BASE_DIR, 'disease_info.csv'), encoding='cp1252')
    supplements = pd.read_csv(os.path.join(BASE_DIR, 'supplement_info.csv'), encoding='cp1252')

    catalog = []
    for i in range(len(diseases)):
        d, s = diseases.iloc[i], supplements.iloc[i]
        name = clean(d['disease_name']) or ''
        crop, sep, condition = name.partition(' : ')
        if not sep:
            crop, condition = '', name

        if i == BACKGROUND_INDEX:
            status = 'none'
        elif 'healthy' in name.lower():
            status = 'healthy'
        else:
            status = 'diseased'

        catalog.append({
            'name': name,
            'crop': crop,
            'condition': condition,
            'status': status,
            'description': split_lines(d['description']),
            'steps': split_lines(d['Possible Steps']),
            'image_url': https(d['image_url']),
            'supplement': clean(s['supplement name']),
            'supplement_image': https(s['supplement image']),
            'buy_link': clean(s['buy link']),
        })
    return catalog


CATALOG = build_catalog()


# ---------------------------------------------------------------- model
model = CNN.CNN(39)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'plant_disease_model_1_latest.pt'),
                                 map_location='cpu'))
model.eval()


def prediction(image_path):
    """Return (class index, confidence in %)."""
    with Image.open(image_path) as img:
        img = img.convert('RGB').resize((224, 224))   # convert: PNGs with alpha / greyscale crash otherwise
    batch = TF.to_tensor(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[0]
    confidence, index = torch.max(probs, dim=0)
    return int(index), float(confidence) * 100


# ---------------------------------------------------------------- app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB uploads


@app.route('/')
def home_page():
    return render_template('home.html', class_count=sum(1 for c in CATALOG if c['status'] != 'none'))


@app.route('/contact')
def contact():
    return render_template('contact-us.html')


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'GET':
        return redirect(url_for('ai_engine_page'))

    file = request.files.get('image')
    if not file or not file.filename:
        return render_template('index.html', error='Choose a leaf photo first.'), 400

    extension = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if extension not in ALLOWED_EXTENSIONS:
        return render_template('index.html', error='Unsupported file type. Use a JPG, PNG or WebP image.'), 400

    saved_name = f'{uuid.uuid4().hex}.{extension}'     # never trust the client's filename
    file_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(file_path)

    try:
        index, confidence = prediction(file_path)
    except (UnidentifiedImageError, OSError):
        os.remove(file_path)
        return render_template('index.html', error='That file could not be read as an image. Try another photo.'), 400

    return render_template(
        'submit.html',
        item=CATALOG[index],
        confidence=confidence,
        low_confidence=confidence < LOW_CONFIDENCE,
        upload_url=url_for('static', filename=f'uploads/{saved_name}'),
    )


@app.route('/market')
def market():
    products = [c for c in CATALOG if c['status'] != 'none' and c['supplement']]
    return render_template('market.html', products=products)


@app.errorhandler(413)
def too_large(_):
    return render_template('index.html', error='That image is larger than 10 MB. Try a smaller photo.'), 413


if __name__ == '__main__':
    app.run(debug=True)
