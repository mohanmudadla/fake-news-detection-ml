import pandas as pd
import re

# Load data
fake = pd.read_csv("Fake.csv", encoding="latin1", low_memory=False)
real = pd.read_csv("True.csv", encoding="latin1", low_memory=False)

# Add labels
fake["label"] = 0
real["label"] = 1

# Combine datasets
data = pd.concat([fake, real])

# Keep only required columns
data = data[["text", "label"]]

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data["text"] = data["text"].apply(clean_text)

# Split data
from sklearn.model_selection import train_test_split

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Check accuracy
from sklearn.metrics import accuracy_score

pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))


# Prediction function
def predict_news(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    
    result = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    if result == 1:
        return "Real News", max(prob)
    else:
        return "Fake News", max(prob)


# Test input
sample = "Breaking: Government announces new policy today"

prediction, confidence = predict_news(sample)

print("\nPrediction:", prediction)
print("Confidence:", confidence)
def get_keywords(text):
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    scores = vec.toarray()[0]
    
    top_indices = scores.argsort()[-5:]
    keywords = [feature_names[i] for i in top_indices]

    return keywords