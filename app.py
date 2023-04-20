from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model
model = joblib.load('naive_bayes_model.joblib')
# Load the count vectorizer
count_vectorizer = joblib.load('count_vectorizer.joblib')

# Define a function to preprocess user input
def preprocess_input(user_input):
    # Convert to lowercase
    user_input = user_input.lower()
    # Remove URLs
    user_input = re.sub(r'http\S+', '', user_input)
    # Tokenize words
    words = word_tokenize(user_input)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Join words back into a sentence
    user_input = ' '.join(words)
    return user_input

# Define a Flask app
app = Flask(__name__)

# Define a route to receive user input and return predicted sentiment
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    # Get user input from Dialogflow
    user_input = request.json['queryResult']['queryText']
    # Preprocess user input
    user_input = preprocess_input(user_input)
    # Vectorize user input using the count vectorizer
    user_input_vectorized = count_vectorizer.transform([user_input])
    # Predict sentiment using the loaded model
    sentiment = model.predict(user_input_vectorized)
    # Return the predicted sentiment to Dialogflow
    return jsonify({
    'fulfillmentText': 'The sentiment is ' + str(sentiment[0])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
