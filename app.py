import tensorflow as tf
from flask import Flask, render_template, request
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

SAVED_MODEL_DIR = "model"

app = Flask(__name__)

LABELS = ["ARTS", "ARTS & CULTURE", "BLACK VOICES", "BUSINESS", "COLLEGE",
          "COMEDY", "CRIME", "CULTURE & ARTS", "DIVORCE", "EDUCATION",
          "ENTERTAINMENT", "ENVIRONMENT", "FIFTY", "FOOD & DRINK",
          "GOOD NEWS", "GREEN", "HEALTHY LIVING", "HOME & LIVING", "IMPACT",
          "LATINO VOICES", "MEDIA", "MONEY", "PARENTING", "PARENTS",
          "POLITICS", "QUEER VOICES", "RELIGION", "SCIENCE", "SPORTS",
          "STYLE", "STYLE & BEAUTY", "TASTE", "TECH", "THE WORLDPOST",
          "TRAVEL", "WEDDINGS", "WEIRD NEWS", "WELLNESS", "WOMEN",
          "WORLD NEWS", "WORLDPOST"]


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
    text = tf.strings.regex_replace(
        text, r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", " ")
    text = tf.strings.regex_replace(text, r'@([A-Za-z0-9_]+)', " ")
    text = tf.strings.regex_replace(text, r"[^A-Za-z0-9]+", " ")

    # remove stopwords
    for i in stopWords:
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

        buffer = list(res)
        topRange = 4
        final = []

        for _ in range(topRange):
            curr = max(buffer)
            currInd = buffer.index(curr)
            buffer.remove(curr)
            final.append({
                "probability": str(curr),
                "label": LABELS[currInd],
            })

        return {
            "distribution": final,
        }, 200
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
