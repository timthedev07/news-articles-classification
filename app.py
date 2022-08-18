import tensorflow as tf
from flask import Flask, render_template, request
from nltk.corpus import stopwords
import os
stop_words = set(stopwords.words('english'))

SAVED_MODEL_DIR = "model"

app = Flask(__name__)


@tf.function
def customStandardization(text: tf.Tensor):
    # to lower case
    text = tf.strings.lower(text)
    # expand contraction
    pairs = [
        ("won't", "will not"),
        ("can't", "can not"),
        ("n't", " not"),
        ("'re", " are"),
        ("'s", " is"),
        ("'d", " would"),
        ("'ll", " will"),
        ("'t", " not"),
        ("'ve", " have"),
        ("'m", " am"),
    ]
    for contracted, replacement in pairs:
        text = tf.strings.regex_replace(text, contracted, replacement)

    # clean special symbols
    text = tf.strings.regex_replace(text, "<br />", " ")
    text = tf.strings.regex_replace(
        text, r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", " ")
    text = tf.strings.regex_replace(text, r'@([A-Za-z0-9_]+)', " ")
    text = tf.strings.regex_replace(text, r"\([^)]*\)", " ")
    text = tf.strings.regex_replace(text, r"[^A-Za-z0-9]+", " ")

    # remove stopwords
    for i in stop_words:
        text = tf.strings.regex_replace(
            text, f"[^A-Za-z0-9_]+{i}[^A-Za-z0-9_]+", " ")

    return text


def loadModel():
    custom_objects = {"customStandardization": customStandardization}
    with tf.keras.utils.custom_object_scope(custom_objects):
        loaded = tf.keras.models.load_model(SAVED_MODEL_DIR)
        return loaded


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        data = request.get_json()

        if not data["text"]:
            return "Bad Request", 400

        model = loadModel()
        [res] = model.predict([data["text"]])

        return {
            "distribution": str(list(res)[:4]),
        }, 200
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
