News Headline Classification Project using Machine Learning
Project Description:
The goal of this project is to classify news headlines into categories such as World, Sports, Business, and Science/Technology. The AG News dataset is used to train classification models using multiple algorithms and compare their performance using metrics like accuracy, precision, recall, and F1-score.

Steps:
1. Data Loading:
The data is loaded using the pandas library to read the parquet files:

python
Copy
Edit
train_df = pd.read_parquet("train.parquet")
test_df = pd.read_parquet("test.parquet")
2. Data Preprocessing:
Text Cleaning: The text is preprocessed by converting it to lowercase, removing non-alphabetic characters, and tokenizing it into words. Additionally, stopwords are removed from the text:
python
Copy
Edit
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        return " ".join(words)
    return ""
TF-IDF Vectorization: We use TfidfVectorizer to convert the cleaned text into numerical features for model training:
python
Copy
Edit
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
X_test = vectorizer.transform(test_df['cleaned_text'])
3. Model Training and Evaluation:
We train and evaluate four classifiers: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), and Gradient Boosting.
Models are evaluated using accuracy, precision, recall, and F1-score.
python
Copy
Edit
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "KNN": KNeighborsClassifier(n_neighbors=3, algorithm="auto"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

results = []

for name, model in models.items():
    print(f"🛠️{name}...")
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
4. Results:
Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	0.917368	0.917144	0.917368	0.917162
Decision Tree	0.450000	0.746169	0.450000	0.432561
KNN	0.894605	0.894320	0.894605	0.894410
Gradient Boosting	0.835526	0.838717	0.835526	0.836380
5. Key Observations:
Logistic Regression performed the best in all metrics, making it the top choice for this classification task.
Decision Tree showed the lowest performance, likely due to its inability to capture the complexity of the data with the current settings.
K-Nearest Neighbors (KNN) performed well with balanced precision and recall.
Gradient Boosting showed decent results, though slightly behind Logistic Regression and KNN.
