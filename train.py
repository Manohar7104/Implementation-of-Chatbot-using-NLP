import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open("intents.json") as f:
    intents = json.load(f)

tags = []
patterns = []

for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

vectorizer = TfidfVectorizer(ngram_range=(1,4))
X = vectorizer.fit_transform(patterns)

model = LogisticRegression(max_iter=10000)
model.fit(X, tags)

# save files
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))
pickle.dump(model, open("model.pkl","wb"))

print("Model saved ✅")