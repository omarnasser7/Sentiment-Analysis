# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import re
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from PIL import Image
# import base64

# # Streamlit page configuration
# st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# # Function to get base64 of binary file for background image
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# # Function to set background image
# def set_background(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = f'''
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{bin_str}");
#         background-size: cover;
#     }}
#     </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# # Set background image
# set_background('background.jpg')  

# # Title of the app
# st.title("Sentiment Analysis Model")

# # Load the pre-trained sentiment analysis model
# Sentiment_Analysis = tf.keras.models.load_model("Sentiment_Analysis_model.keras")

# # Cache the data loading and preprocessing steps
# @st.cache_data
# def load_and_preprocess_data():
#     # Load the dataset
#     df = pd.read_csv("Twitter_Data.csv")
    
#     # Clean the dataset
#     df['clean_text'].fillna('', inplace=True)
#     df.dropna(subset=['category'], inplace=True)
    
#     # Extract the text data
#     X = df['clean_text']
    
#     # Fit the tokenizer
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(X)
    
#     max_seq_length = 52  # Maximum sequence length for padding
    
#     return tokenizer, max_seq_length

# # Load the tokenizer and max sequence length only once
# tokenizer, max_seq_length = load_and_preprocess_data()

# # Function to classify sentiment
# def classify_sentiment(text):
#     # Convert text to lowercase
#     text = text.lower()
    
#     # Remove special characters and punctuation
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Tokenize the text
#     tokens = word_tokenize(text)
    
#     # Remove digits
#     tokens = [word for word in tokens if not word.isdigit()]
    
#     # Lemmatize the tokens
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
#     # Rejoin the tokens into a single string
#     text = ' '.join(tokens)
    
#     # Convert the text to a sequence and pad it
#     text_sequence = tokenizer.texts_to_sequences([text])
#     padded_sequence = pad_sequences(text_sequence, maxlen=max_seq_length)
    
#     # Make a prediction using the model
#     prediction = Sentiment_Analysis.predict(padded_sequence)
    
#     # Get the predicted label (0: Negative, 1: Neutral, 2: Positive)
#     predicted_label = np.argmax(prediction)
    
#     # Map the predicted label to a sentiment
#     sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
#     sentiment = sentiment_mapping[predicted_label]
    
#     return text, sentiment

# # Text input for user to analyze sentiment
# user_input = st.text_area("Enter a text for sentiment analysis:")

# # Analyze sentiment when the button is clicked
# if st.button("Analyze"):
#     if user_input.strip() != "":
#         processed_text, sentiment = classify_sentiment(user_input)
        
#         # Display the processed text
#         st.markdown(f"### Processed Text:")
#         st.write(f"**{processed_text}**")
        
#         # Display the sentiment with color styling
#         sentiment_color_map = {
#             "Negative": "#FF4B4B",  # Red
#             "Neutral": "#1d1c1f",   
#             "Positive": "#4CAF50"   # Green
#         }
        
#         sentiment_color = sentiment_color_map[sentiment]
#         st.markdown(f"### Sentiment:")
#         st.markdown(f"<span style='color:{sentiment_color}; font-size:24px;'>**{sentiment}**</span>", unsafe_allow_html=True)
#     else:
#         st.write("Please enter a valid text.")

import nltk
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from PIL import Image
import base64

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Function to get base64 of binary file for background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background image
set_background('background.jpg')  

# Title of the app
st.title("Sentiment Analysis Model")

# Load the pre-trained sentiment analysis model
try:
    Sentiment_Analysis = tf.keras.models.load_model("Sentiment_Analysis_model.keras")
except Exception as e:
    st.error("Error loading the model. Please ensure the model file is in the correct path.")
    st.stop()

# Cache the data loading and preprocessing steps
@st.cache_data
def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv("Twitter_Data.csv")
    
    # Clean the dataset
    df['clean_text'].fillna('', inplace=True)
    df.dropna(subset=['category'], inplace=True)
    
    # Extract the text data
    X = df['clean_text']
    
    # Fit the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    
    max_seq_length = 52  # Maximum sequence length for padding
    
    return tokenizer, max_seq_length

# Load the tokenizer and max sequence length only once
tokenizer, max_seq_length = load_and_preprocess_data()

# Function to classify sentiment
def classify_sentiment(text):
    # Convert text to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Tokenize, remove digits, and lemmatize tokens
    tokens = [WordNetLemmatizer().lemmatize(word) for word in word_tokenize(text) if not word.isdigit()]
    
    # Convert the processed text to a sequence and pad it
    text_sequence = tokenizer.texts_to_sequences([' '.join(tokens)])
    padded_sequence = pad_sequences(text_sequence, maxlen=max_seq_length)
    
    # Make a prediction using the model
    prediction = Sentiment_Analysis.predict(padded_sequence)
    predicted_label = np.argmax(prediction)
    
    # Map the predicted label to a sentiment
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return ' '.join(tokens), sentiment_mapping[predicted_label]

# Text input for user to analyze sentiment
user_input = st.text_area("Enter a text for sentiment analysis:")

# Analyze sentiment when the button is clicked
if st.button("Analyze"):
    if user_input.strip():
        processed_text, sentiment = classify_sentiment(user_input)
        
        # Display the processed text and sentiment
        st.markdown(f"### Processed Text: **{processed_text}**")
        
        sentiment_color_map = {"Negative": "#FF4B4B", "Neutral": "#1d1c1f", "Positive": "#4CAF50"}
        sentiment_color = sentiment_color_map[sentiment]
        
        st.markdown(f"### Sentiment: <span style='color:{sentiment_color}; font-size:24px;'>**{sentiment}**</span>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a valid text.")
