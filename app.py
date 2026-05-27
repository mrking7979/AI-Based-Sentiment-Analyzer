from flask import Flask, render_template, request
from transformers import pipeline
from pymongo import MongoClient
from datetime import datetime
import matplotlib

# FAST BACKEND FOR MATPLOTLIB
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

# --------------------------------
# FLASK APP
# --------------------------------
app = Flask(__name__)

# --------------------------------
# LOAD AI MODEL ONLY ONCE
# --------------------------------
print("Loading AI model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("AI Model Loaded Successfully!")

# --------------------------------
# MONGODB CONNECTION
# --------------------------------
client = MongoClient("mongodb://localhost:27017/")

db = client["sentimentDB"]

collection = db["sentimentResults"]

# --------------------------------
# CREATE CHART FUNCTION
# --------------------------------
def create_chart():

    data = list(collection.find())

    if len(data) == 0:
        return

    labels = []

    for item in data:
        labels.append(item["label"])

    positive_count = labels.count("POSITIVE")

    negative_count = labels.count("NEGATIVE")

    plt.figure(figsize=(4,4))

    plt.pie(
        [positive_count, negative_count],
        labels=["POSITIVE", "NEGATIVE"],
        autopct="%1.1f%%"
    )

    plt.title("Sentiment Report")

    plt.savefig("static/chart.png")

    plt.close()

# --------------------------------
# HOME ROUTE
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        user_text = request.form["text"]

        # --------------------------------
        # FAST CUSTOM CHECK
        # --------------------------------
        bad_words = [
            "fuck",
            "hate",
            "idiot",
            "stupid",
            "worst",
            "bad"
        ]

        # --------------------------------
        # CUSTOM NEGATIVE DETECTION
        # --------------------------------
        if any(word in user_text.lower() for word in bad_words):

            label = "NEGATIVE"

            score = 99.0

        else:

            prediction = sentiment_pipeline(user_text)[0]

            label = prediction["label"]

            score = round(prediction["score"] * 100, 2)

        # --------------------------------
        # RESULT
        # --------------------------------
        result = {
            "text": user_text,
            "label": label,
            "score": score,
            "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }

        # SAVE TO DATABASE
        collection.insert_one(result)

        # CREATE CHART
        create_chart()

    # --------------------------------
    # HISTORY
    # --------------------------------
    history = list(
        collection.find().sort("_id", -1).limit(5)
    )

    return render_template(
        "index.html",
        result=result,
        history=history
    )

# --------------------------------
# RUN APP
# --------------------------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )