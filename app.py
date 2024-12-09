from flask import Flask, request, jsonify, render_template, url_for
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
import open_clip

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Adjusted model and tokenizer initialization
model_name = "ViT-B-32-quickgelu"  # Replace with a valid model name
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
tokenizer = open_clip.get_tokenizer(model_name)

# Load embeddings
try:
    embeddings_df = pd.read_pickle('image_embeddings.pickle')
    image_paths = embeddings_df.iloc[:, 0].values
    embeddings = np.vstack(embeddings_df.iloc[:, 1].values)
except Exception as e:
    print(f"Error loading embeddings: {e}")
    raise

# PCA Initialization
pca = PCA(n_components=50)
try:
    pca.fit(embeddings)
except Exception as e:
    print(f"Error initializing PCA: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    try:
        query_type = request.form['query_type']
        embedding_type = request.form.get('embedding_type', 'clip')
        results = []

        if query_type == 'text':
            text_query = request.form.get('text_query', '')
            if not text_query:
                return jsonify({'error': 'Text query is missing'}), 400

            text_embedding = F.normalize(model.encode_text(tokenizer([text_query])))
            scores = torch.mm(torch.from_numpy(embeddings), text_embedding.T).squeeze(1)
            top_indices = torch.topk(scores, 5).indices.numpy()
            results = [{'image_path': url_for('static', filename=f"coco_images_resized/{image_paths[i]}"),
                        'similarity': float(scores[i])} for i in top_indices]

        elif query_type == 'image':
            if 'image_query' not in request.files:
                return jsonify({'error': 'Image file is missing'}), 400

            file = request.files['image_query']
            if not file:
                return jsonify({'error': 'Invalid image file'}), 400

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = preprocess(Image.open(filepath)).unsqueeze(0)
            image_embedding = F.normalize(model.encode_image(image))
            scores = torch.mm(torch.from_numpy(embeddings), image_embedding.T).squeeze(1)
            top_indices = torch.topk(scores, 5).indices.numpy()
            results = [{'image_path': url_for('static', filename=f"coco_images_resized/{image_paths[i]}"),
                        'similarity': float(scores[i])} for i in top_indices]

            os.remove(filepath)

        elif query_type == 'hybrid':
            text_query = request.form.get('text_query', '')
            weight = float(request.form.get('weight', 0.5))
            if 'image_query' not in request.files or not text_query:
                return jsonify({'error': 'Hybrid query requires both text and image inputs'}), 400

            file = request.files['image_query']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            text_embedding = F.normalize(model.encode_text(tokenizer([text_query])))
            image = preprocess(Image.open(filepath)).unsqueeze(0)
            image_embedding = F.normalize(model.encode_image(image))

            hybrid_embedding = F.normalize(weight * text_embedding + (1 - weight) * image_embedding)
            scores = torch.mm(torch.from_numpy(embeddings), hybrid_embedding.T).squeeze(1)
            top_indices = torch.topk(scores, 5).indices.numpy()
            results = [{'image_path': url_for('static', filename=f"coco_images_resized/{image_paths[i]}"),
                        'similarity': float(scores[i])} for i in top_indices]

            os.remove(filepath)

        else:
            return jsonify({'error': 'Invalid query type'}), 400

        return jsonify({'results': results})

    except Exception as e:
        print(f"Error in /search: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)
