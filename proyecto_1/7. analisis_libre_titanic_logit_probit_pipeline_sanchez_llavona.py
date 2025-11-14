import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import statsmodels.api as sm

def main(csv_path):
    df = pd.read_csv(csv_path)

    # --- Target ---
    target = "Survived" if "Survived" in df.columns else None
    if target is None:
        for c in df.columns:
            vals = df[c].dropna().unique()
            if len(vals) <= 3 and set(pd.Series(vals).dropna().unique()).issubset({0, 1}):
                target = c
                break
    if target is None:
        raise ValueError("No binary target found. Add 'Survived' or a 0/1 column.")

    y = df[target].astype(int)

    # --- Features ---
    candidate_features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","IsAlone","Title"]
    X_cols = [c for c in candidate_features if c in df.columns]
    if len(X_cols) < 3:
        excluded = {target}
        excluded |= {c for c in df.columns if c.lower() in {"passengerid","name","ticket","cabin"}}
        X_cols = [c for c in df.columns if c not in excluded]

    X = df[X_cols].copy()

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    # --- Preprocess ---
    num_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    preprocess = ColumnTransformer([("num", num_tf, numeric), ("cat", cat_tf, categorical)])
    X_clean = preprocess.fit_transform(X)

    feature_names = numeric[:]
    if categorical:
        ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
        feature_names += ohe.get_feature_names_out(categorical).tolist()

    X_clean_df = pd.DataFrame(
        X_clean.toarray() if hasattr(X_clean, "toarray") else X_clean,
        columns=feature_names, index=X.index
    )

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean_df, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Constante ---
    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_test_const  = sm.add_constant(X_test,  has_constant="add")

    # --- Modelos ---
    logit_res  = sm.Logit(y_train, X_train_const).fit(disp=False)
    probit_res = sm.Probit(y_train, X_train_const).fit(disp=False)

    # --- Evaluación ---
    for name, res in [("LOGIT", logit_res), ("PROBIT", probit_res)]:
        y_prob = res.predict(X_test_const)
        y_pred = (y_prob >= 0.5).astype(int)
        acc  = accuracy_score(y_test, y_pred)
        pre  = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)
        cm   = confusion_matrix(y_test, y_pred)
        print(f"\n--- {name} ---")
        print(res.summary())
        print(f"Accuracy:  {acc:.3f}  Precision: {pre:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  AUC: {auc:.3f}")
        print("Confusion matrix (rows=true, cols=pred):\n", cm)

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("csv", nargs="?", default=None, help="Ruta al CSV")
    args, unknown = parser.parse_known_args()

    # Manejo de argumentos tipo Jupyter (--f=kernel.json)
    if args.csv and args.csv.lower().endswith((".csv", ".csv.gz", ".gz")) and os.path.exists(args.csv):
        csv_path = args.csv
    else:
        csv_path = r"C:\Users\leodo\OneDrive\Escritorio\special topics\proyecto_1\titanic_1.csv"

    print("Usando CSV:", csv_path)
    main(csv_path)
