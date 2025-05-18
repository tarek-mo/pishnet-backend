from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import string
from nltk.corpus import stopwords
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import json_util
import json
import datetime


# Load environment variables from .env file
load_dotenv()

# Load model and vectorizer
model = joblib.load('phishing_lr_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
mongo_client = MongoClient(DB_CONNECTION_STRING)

# Select your DB and collection
db = mongo_client["phishnet"]  # You can change this name
predictions_collection = db["predictions"]

# Initialize Flask app
app = Flask(__name__)

# Define allowed frontend origin
ALLOWED_ORIGIN = os.getenv('ALLOWED_ORIGIN', 'http://localhost:3000')

# Enable CORS
CORS(app, origins=[ALLOWED_ORIGIN])

# Preprocessing function
def preprocess_email(text):
    """Apply all preprocessing steps to email text"""
    if not isinstance(text, str):
        return ''

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove unnecessary symbols but preserve important ones
    # Keep '@', '#', '.', '-', '_', '!'
    text = re.sub(r'[^\w\s@#.\-_!]', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'never'}
    stop_words = stop_words.difference(negation_words)

    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    text = ' '.join(filtered)

    return text

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('email_text', '')
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400
    # Preprocess and predict
    cleaned_text = preprocess_email(email_text)
    tfidf_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_input)[0]
    probabilities = model.predict_proba(tfidf_input)[0]
    label = 'Phishing Email' if prediction == 1 else 'Safe Email'
    confidence = round(probabilities[prediction] * 100, 2)
    # Save to MongoDB Atlas
    predictions_collection.insert_one({
        "original_text": email_text,
        "cleaned_text": cleaned_text,
        "prediction": label,
        "confidence": confidence,
        "created_at": datetime.datetime.now()

    })
    return jsonify({
        'prediction': label,
        'confidence': f"{confidence}%"
    })


@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        predictions_cursor = predictions_collection.find(
            {}, {"cleaned_text": 0}
        )
        predictions = []
        for doc in predictions_cursor:
            doc['_id'] = str(doc['_id'])  # Convert ObjectId to plain string
            predictions.append(doc)
        return jsonify(predictions), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
