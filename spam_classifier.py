import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    "message": [
        "Win a free iPhone now",
        "Congratulations, you won a lottery",
        "Meeting at 5 PM today",
        "Please submit the report",
        "Claim your prize now",
        "Project discussion tomorrow"
    ],
    "label": ["spam", "spam", "ham", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])

# Target variable
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom message
message = ["You have won a free gift"]
message_vector = vectorizer.transform(message)

prediction = model.predict(message_vector)

print("Prediction:", prediction[0])