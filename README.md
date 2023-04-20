# Sentiment Analysis Webhook

This is a webhook for performing sentiment analysis on user input using a pre-trained Naive Bayes model.

## Getting Started

### Prerequisites

This webhook requires Python 3.x and the following Python libraries:

- Flask
- joblib
- nltk

### Installation

1. Clone this repository
2. Install the required Python libraries using pip: `pip install flask joblib nltk`
3. Download the required NLTK data by running the following commands:

```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. Start the Flask app by running `python app.py`

## Usage

Send a POST request to the `/predict_sentiment` endpoint with the user input as a JSON object. The predicted sentiment will be returned as a JSON response.

Example:

```
POST /predict_sentiment
{
  "queryResult": {
    "queryText": "I love this product!"
  }
}
```

Response:

```
{
  "fulfillmentText": "The sentiment is positive"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
