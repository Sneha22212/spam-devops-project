import pandas as pd
import pickle
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -------------------------------
# 2. Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text

df['message'] = df['message'].apply(clean_text)

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# -------------------------------
# 4. TF-IDF Vectorization (Improved)
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.9,
    min_df=2,
    ngram_range=(1,2)   # unigram + bigram (better accuracy)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 5. Train BEST Model
# -------------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

# -------------------------------
# 6. Evaluation
# -------------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -------------------------------
# 7. Save Model
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel trained & saved successfully!")
