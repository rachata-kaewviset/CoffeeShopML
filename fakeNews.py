import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from LogisticRegression import LogisticRegression

data = pd.read_csv('cleaned_fakenews.csv')

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1, stratify=y)

vectorizer = TfidfVectorizer(stop_words='english',max_features=1600)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

lr_classifier = LogisticRegression(learning_rate=5, num_iterations=1000)
lr_classifier.fit(X_train_tfidf.toarray(), y_train)

y_pred = lr_classifier.predict(X_test_tfidf.toarray())
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
