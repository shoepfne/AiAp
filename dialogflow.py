# https://www.pragnakalp.com/dialogflow-fulfillment-webhook-tutorial/

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from flask import Flask, request, make_response, jsonify

# initialize the flask app
app = Flask(__name__)

# The training data folder must be passed as first argument
languages_data_folder = "data/paragraphs"
dataset = load_files(languages_data_folder)

# Split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.5)

# TASK: Build a vectorizer that splits strings into sequence of 1 to 3
# characters instead of word tokens
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char', use_idf=False)

# TASK: Build a vectorizer / classifier pipeline using the previous analyzer
# the pipeline instance should stored in a variable named clf
clf = Pipeline([
    ('vec', vectorizer),
    ('clf', Perceptron()),
])

# TASK: Fit the pipeline on the training set
clf.fit(docs_train, y_train)


# default route
@app.route('/')
def index():
    return 'Webhook is running under /webhook'


# function for responses
def results():
    # build a request object
    req = request.get_json(force=True)

    # fetch action from json
    text = req.get('queryResult').get('queryText')
    app.logger.info(text)

    res = predict(text)
    app.logger.info(res)

    # return a fulfillment response
    return {"fulfillmentMessages": [{"text": {"text": [res]}}]}


# create a route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # return response
    return make_response(jsonify(results()))


def predict(text):
    texts = [text]
    predicted = clf.predict(texts)

    for s, p in zip(texts, predicted):
        return dataset.target_names[p]


# run the app
if __name__ == '__main__':
    app.run()
