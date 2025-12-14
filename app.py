import os
import pickle
import re
import emoji
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template
from googleapiclient.discovery import build
from dotenv import load_dotenv

# --- 0. SETUP AND LOAD EVERYTHING ---
load_dotenv()
app = Flask(__name__)

# Load Model, Tokenizer, and Preprocessing Functions (same as before)
MODEL_PATH = os.path.join('model_artifacts', 'sarcasm_humor_model.keras')
TOKENIZER_PATH = os.path.join('model_artifacts', 'tokenizer.pkl')
MAXLEN = 100
model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
slang_dict = {"lol": "laughing out loud", "lmfao": "laughing my freaking ass off", "smh": "shaking my head"}
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = emoji.demojize(text, delimiters=(" ", " ")); text = text.lower()
    words = text.split(); expanded_words = [slang_dict.get(word, word) for word in words]
    text = " ".join(expanded_words); text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text); text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z0-9\s_]', '', text); text = " ".join(text.split())
    return text

# --- 1. HELPER FUNCTIONS FOR YOUTUBE ---
def get_video_id_from_url(url):
    """Extracts the YouTube video ID from a URL using regex."""
    match = re.search(r"(?<=v=)[a-zA-Z0-9_-]+", url)
    if match:
        return match.group(0)
    return None

def get_youtube_comments(video_id, api_key, max_results=50):
    """Fetches comments from a YouTube video using the API."""
    try:
        youtube_service = build('youtube', 'v3', developerKey=api_key)
        request = youtube_service.commentThreads().list(
            part='snippet', videoId=video_id, textFormat='plainText', maxResults=max_results)
        response = request.execute()
        comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
        return comments
    except Exception as e:
        print(f"API Error: {e}")
        return []

# --- 2. EXISTING FLASK ROUTES (Unchanged) ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    processed_input = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([processed_input])
    padded_sequence = pad_sequences(sequence, maxlen=MAXLEN, padding='post', truncating='post')
    prediction_probs = model.predict(padded_sequence)
    predicted_label_index = np.argmax(prediction_probs, axis=1)[0]
    label_map = {0: "Not Sarcastic/Humorous", 1: "Sarcastic", 2: "Humor"}
    prediction_text = label_map.get(predicted_label_index, "Unknown")
    return render_template('result.html', original_text=user_input, prediction=prediction_text)

# --- 3. THE NEW FLASK ROUTE FOR YOUTUBE ANALYSIS ---
@app.route('/analyze_youtube', methods=['POST'])
def analyze_youtube():
    video_url = request.form['youtube_url']
    video_id = get_video_id_from_url(video_url)
    results = []
    
    if not video_id:
        # Handle invalid URL
        return render_template('youtube_results.html', results=results, video_url=video_url)

    api_key = os.getenv("YOUTUBE_API_KEY")
    comments = get_youtube_comments(video_id, api_key)

    if comments:
        processed_comments = [preprocess_text(c) for c in comments]
        sequences = tokenizer.texts_to_sequences(processed_comments)
        padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding='post', truncating='post')
        predictions = model.predict(padded_sequences)
        predicted_indices = np.argmax(predictions, axis=1)
        
        label_map = {0: "Not Sarcastic/Humorous", 1: "Sarcastic", 2: "Humor"}
        
        for i, comment in enumerate(comments):
            prediction_label = label_map[predicted_indices[i]]
            results.append({"comment": comment, "prediction": prediction_label})
            
    return render_template('youtube_results.html', results=results, video_url=video_url)

# --- 4. RUN THE APPLICATION ---
if __name__ == '__main__':
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', port=5000, debug=True)