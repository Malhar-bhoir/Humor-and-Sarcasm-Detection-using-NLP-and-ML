
-----

# Sarcasm & Humor Detection Web App

A Deep Learning-powered web application designed to analyze text and detect underlying sentiments, specifically **Sarcasm** and **Humor**. The application supports both direct text input and automated analysis of comments from **YouTube videos**.

## 📌 Project Overview

This project bridges Natural Language Processing (NLP) and Web Development. It uses a **Long Short-Term Memory (LSTM)** neural network model trained on a diverse dataset to classify text into three categories:

1.  **Sarcastic**
2.  **Humor**
3.  **Neutral**

The application is wrapped in a **Flask** web framework, allowing users to interact with the model via a modern, responsive interface.

## 🚀 Key Features

  * **Real-Time Text Analysis:** Analyze any custom text string instantly.
  * **YouTube Integration:** Paste a YouTube video URL to fetch and analyze the top comments automatically using the YouTube Data API v3.
  * **Advanced Preprocessing:** Custom NLP pipeline that handles emoji translation, slang expansion, and noise removal.
  * **Visual Results:** Clear, badge-based classification results for easy interpretation.

## 🛠️ Tech Stack

  * **Frontend:** HTML5, CSS3, Bootstrap 5
  * **Backend:** Python 3, Flask
  * **Machine Learning:** TensorFlow 2.0, Keras (LSTM Architecture)
  * **Data Processing:** Pandas, NumPy, Pickle
  * **External API:** Google YouTube Data API v3

## 📂 Project Structure

```bash
├── app.py                 # Main Flask application entry point
├── preprocessing.py       # Custom text cleaning and tokenization logic
├── model/
│   ├── sarcasm_model.h5   # Trained Keras LSTM model
│   └── tokenizer.pickle   # Saved tokenizer object
├── templates/
│   ├── index.html         # Homepage (Input forms)
│   ├── result.html        # Result page for text analysis
│   └── youtube_results.html # Result page for YouTube comments
├── static/
│   └── style.css          # Custom styling
├── .env                   # Environment variables (API Keys)
└── requirements.txt       # Project dependencies
```

## ⚙️ Installation & Setup

### 1\. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 2\. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Configure API Keys

Create a `.env` file in the root directory to store your YouTube API key securely.

```bash
touch .env
```

Open the file and add your key:

```env
YOUTUBE_API_KEY=your_actual_api_key_here
```

### 5\. Run the Application

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000/`.

## 🧠 Model Architecture

The core of the application is a Recurrent Neural Network (RNN) with the following layers:

1.  **Embedding Layer:** Maps words to dense vectors.
2.  **SpatialDropout1D:** Reduces overfitting.
3.  **LSTM Layer:** Captures sequential dependencies and context in text.
4.  **Dense Output Layer:** Uses Softmax activation for multi-class classification.

**Performance:** The model currently achieves an accuracy of approximately **72%** on the test dataset.

## 🔮 Future Scope

  * **Multilingual Support:** Extending the model to detect sarcasm in languages other than English.
  * **Transformer Integration:** Upgrading the LSTM model to BERT or RoBERTa for higher accuracy.
  * **Browser Extension:** Creating a plugin for on-the-fly analysis while browsing social media.

