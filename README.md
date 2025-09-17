
# Name Classifier - Person, Company, or University

This repository contains a machine learning solution for classifying text strings (names) into one of three categories: `Person`, `Company`, or `University`.


## Summary & Critical Evaluation

The solution combines classical machine learning with feature engineering using TF-IDF vectorization and logistic regression. The key steps included:
- **Data cleaning**: Removing noise such as special characters, excessive whitespace, and non-ASCII symbols.
- **Feature engineering**: Using n-gram TF-IDF features to capture word patterns.
- **Model training**: Logistic Regression with stratified train-test split to handle class imbalance.
- **Deployment**: A Dockerized Flask API for prediction.

### Observations & Challenges:
- **Class Imbalance**: The dataset was highly skewed (81% Person, 2% University), which negatively affected recall for underrepresented classes.
- **Noisy Data**: Titles (Mr, Dr, Miss) and suffixes (Pty, Ltd) were helpful indicators, but their presence wasn't consistent.
- **Language Sensitivity**: University names in different languages (e.g., Afrikaans) sometimes reduced accuracy, indicating a need for multilingual awareness or richer embeddings.

### Improvements:
- Implemented bigram features and limited vector space (`max_features=1000`) to balance performance and model generalization.
- Cleaned and normalized text using `unidecode` and regular expressions.
- Incorporated basic error handling during preprocessing and prediction.


## How to Run

Ensure you have **Docker installed**.

### 1. Build the Docker image:

```bash
docker build -t local/classifier/model-v1 -f deployment/Dockerfile .
```

### 2. Run the container:

```bash
docker run -p 8080:8080 local/classifier/model-v1
```

The API will now be available at `http://localhost:8080`.

### 3. To test it locally you can use curl: example

```bash
curl -X POST http://0.0.0.0:8080/v1/models/model:predict \
     -H "Content-Type: application/json" \
     -d '{"instances": ["The AI Team LTD","Moses Thobakgale"]}'


```


---

## Files and Structure

```
.
├── deployment/
│   └── Dockerfile            # Container setup instructions
├── notebooks                 # notebooks
├── save_model                # trained model and fitted vectorizer
├── notebook.ipynb            # Jupyter Notebook with full EDA and training
└── README.md                 # This file
```


## Assumptions

- Names can contain accents, punctuation, and whitespace – all of which need normalization.
- Titles (e.g., Mr, Prof) and keywords (e.g., university, ltd) are useful for classifying text.
- Since class imbalance exists, we used stratified train-test splitting to preserve label ratios.

---

## Performance

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | ~92%      |
| Vectorizer    | TF-IDF (1,2)-grams |
| Model         | Logistic Regression |

Confusion matrix and classification reports are available in the Jupyter notebook for full transparency.

## Future Improvements

- Handle honorifics (Mr, Miss, etc.) more explicitly as separate features.
- Improve handling of multilingual names, especially for universities.
- Rebalance the dataset through augmentation or weighted loss functions.



