from flask import Flask, render_template, request
import pickle
import re
import string

# -------------------------------
# 1. Initialize Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# 2. Load trained model & vectorizer
# -------------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    print("⚠️ Error loading model:", e)

# -------------------------------
# 3. Text Cleaning Function (IMPORTANT FIX)
# -------------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

# -------------------------------
# 4. Home Route
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------
# 5. Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        message = request.form.get("message")

        # Check empty input
        if not message or message.strip() == "":
            return render_template("index.html", prediction="⚠️ Please enter a message")

        # 🔥 CLEAN TEXT BEFORE PREDICTION (MAIN FIX)
        cleaned_message = clean_text(message)

        # Transform text
        data = vectorizer.transform([cleaned_message])

        # Predict
        prediction = model.predict(data)[0]

        # Result formatting
        if prediction == 1:
            result = "🚫 Spam Message"
        else:
            result = "✅ Not Spam"

        return render_template("index.html", prediction=result)

    except Exception as e:
        print("Error:", e)
        return render_template("index.html", prediction="⚠️ Error occurred")

# -------------------------------
# 6. Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
