# https://www.pragnakalp.com/dialogflow-fulfillment-webhook-tutorial/

import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from flask import Flask, request, make_response, jsonify

language_mapping = {
    'ar': 'Arabic',
    'de': 'German',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'it': 'Italian',
    'ja': 'Japanese',
    'nl': 'Dutch',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ru': 'Russian',
}

# initialize the flask app
app = Flask(__name__)

# TASK: Build a vectorizer that splits strings into sequence of 1 to 3
# characters instead of word tokens
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char', use_idf=False)

# TASK: Build a vectorizer / classifier pipeline using the previous analyzer
# the pipeline instance should stored in a variable named clf
clf = Pipeline([
    ('vec', vectorizer),
    ('clf', Perceptron()),
])


# default route
@app.route('/')
def index():
    return 'Webhook is running under /webhook'


def results():
    req = request.get_json(force=True)
    intent = req.get('queryResult').get('intent')

    if intent.get('displayName') == 'Default Fallback Intent':
        global text
        text = req.get('queryResult').get('queryText')
        res = 'Your text was identified as %s. Is that correct?' % language_mapping[predict()]

        return {'fulfillmentMessages': [{'text': {'text': [res]}}]}
    elif intent.get('displayName') == 'Default Fallback Intent.no.correction':
        language = req.get('queryResult').get('parameters').get('language')

        save_text(language)

        res = 'Thank you for teaching me %s! Please try again.' % language_mapping[language]
        return {'fulfillmentMessages': [{'text': {'text': [res]}}]}


@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    return make_response(jsonify(results()))


def predict():
    texts = [text]
    predicted = clf.predict(texts)

    for s, p in zip(texts, predicted):
        return dataset.target_names[p]


def train_model():
    global dataset
    languages_data_folder = 'data/paragraphs'
    dataset = load_files(languages_data_folder)

    # Split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.5)

    # TASK: Fit the pipeline on the training set
    clf.fit(docs_train, y_train)


train_model()


def save_text(language):
    text_folder = 'data/paragraphs'
    text_lang_folder = os.path.join(text_folder, language)

    if not os.path.exists(text_lang_folder):
        os.makedirs(text_lang_folder)

    current_time = round(time.time() * 1000)
    text_filename = os.path.join(text_lang_folder, '%s_%d.txt' % (language, current_time))

    with open(text_filename, 'wb') as f:
        f.write(text.encode('utf-8', 'ignore'))

    train_model()


# run the app
if __name__ == '__main__':
    app.run()
