"""Classical machine learning: predict whether a review is positive from its text.

Compares Logistic Regression, a Decision Tree and XGBoost (gradient boosting)
on TF-IDF features, using accuracy, F1 and ROC-AUC.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config


def train_and_compare(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=config.RANDOM_STATE,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", min_df=2)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # Reviews are usually mostly positive, so weight classes to keep the model honest.
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, class_weight="balanced",
                                                random_state=config.RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 scale_pos_weight=pos_weight, eval_metric="logloss",
                                 random_state=config.RANDOM_STATE),
    }

    results = []
    for name, model in models.items():
        print(f"  training {name}...")
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        results.append({
            "model": name,
            "accuracy": round(accuracy_score(y_test, pred), 3),
            "f1": round(f1_score(y_test, pred), 3),
            "roc_auc": round(roc_auc_score(y_test, proba), 3),
        })

    # Most telling words from the logistic regression model.
    lr = models["Logistic Regression"]
    terms = vectorizer.get_feature_names_out()
    coefs = pd.Series(lr.coef_[0], index=terms).sort_values()
    top_words = pd.DataFrame({
        "most_negative": coefs.head(10).index,
        "most_positive": coefs.tail(10).index[::-1],
    })

    return pd.DataFrame(results).sort_values("f1", ascending=False), top_words
