import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def clean(texts, labels):
    pairs = [(t.strip(), l) for t, l in zip(texts, labels) if t.strip()]
    texts_c, labels_c = zip(*pairs)
    return list(texts_c), list(labels_c)

RAW_DATA_DIR = "./raw/"

with open(RAW_DATA_DIR + "x_train.txt", encoding="utf-8") as f:
    X_train_raw = [line.strip() for line in f]

with open(RAW_DATA_DIR + "y_train.txt", encoding="utf-8") as f:
    y_train_raw = [line.strip() for line in f]

with open(RAW_DATA_DIR + "x_test.txt", encoding="utf-8") as f:
    X_test_raw = [line.strip() for line in f]

with open(RAW_DATA_DIR + "y_test.txt", encoding="utf-8") as f:
    y_test_raw = [line.strip() for line in f]


print("Example Datapoint : ", X_train_raw[0])
print(f"  Train samples : {len(X_train_raw):,}")
print(f"  Test  samples : {len(X_test_raw):,}")
print(f"  Languages     : {len(set(y_train_raw))}")

X_train_raw, y_train_raw = clean(X_train_raw, y_train_raw)
X_test_raw,  y_test_raw  = clean(X_test_raw,  y_test_raw)

le = LabelEncoder()
le.fit(y_train_raw + y_test_raw)        
y_train = le.transform(y_train_raw)
y_test  = le.transform(y_test_raw)

joblib.dump(le, "label_encoder.joblib")

