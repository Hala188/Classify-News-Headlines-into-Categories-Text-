# News Headline Classification Project using Machine Learning

## Project Description:
This project aims to classify news headlines into four categories: **World, Sports, Business, and Science/Technology**. Using the **AG News** dataset, we preprocess the text data, transform it using TF-IDF vectorization, and train multiple classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors, and Gradient Boosting). The models are evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Steps and Implementation:

### 1. Data Loading:
We load the training and testing data stored in `parquet` files using the `pandas` library.

```python
import pandas as pd

# Load datasets
train_df = pd.read_parquet("train.parquet")
test_df = pd.read_parquet("test.parquet")

print(train_df.head())

```
`4 classes:
0 World
1 sports
2 Business
3 sci/tec`
### 2. Data Preprocessing:
We clean the text by converting it to lowercase, removing non-alphanumeric characters, tokenizing, and removing stopwords.

```python

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        return " ".join(words)
    return ""
# Apply text cleaning to both train and test sets
train_df['cleaned_text'] = train_df['text'].apply(clean_text)
test_df['cleaned_text'] = test_df['text'].apply(clean_text)
```
### 3. Feature Extraction:
We convert the cleaned text into numerical features using TF-IDF vectorization.

```python

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
X_test = vectorizer.transform(test_df['cleaned_text'])

y_train = train_df['label']
y_test = test_df['label']
```
### 4. Model Training and Evaluation:

We train and evaluate four classifiers: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), and Gradient Boosting.

Evaluation is performed using accuracy, precision, recall, and F1-score.

```python

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "KNN": KNeighborsClassifier(n_neighbors=3, algorithm="auto"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

results = []

for name, model in models.items():
    print(f"🛠️ {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-score": report["weighted avg"]["f1-score"]
    })

df_results = pd.DataFrame(results)
print(df_results)
```

### 5.result:

![image](https://github.com/user-attachments/assets/0d574663-fd87-4be1-9587-186406a1ea1c)

### 6. Key Observations:

Logistic Regression shows the best performance across all metrics.

Decision Tree has lower performance, indicating that it might not be capturing the data complexity with the current settings.

K-Nearest Neighbors (KNN) demonstrates strong and balanced performance.

Gradient Boosting performs well, though it is slightly outperformed by Logistic Regression and KNN.
