import os
import sys
import csv
import numpy as np

# make sure src is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from models.naive_bayes import NaiveBayes
from utils import train_test_split, accuracy_score, precision_score, recall_score, f1_score


def load_and_prepare(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if any(cell.strip() != '' for cell in row)]

    n_cols = len(header)
    cols = [[] for _ in range(n_cols)]
    for r in rows:
        for i in range(n_cols):
            cols[i].append(r[i])

    prepared = []
    for col_vals in cols:
        is_num = True
        numeric = []
        for v in col_vals:
            try:
                numeric.append(float(v))
            except Exception:
                is_num = False
                break
        if is_num:
            prepared.append(np.array(numeric, dtype=float))
        else:
            mapping = {}
            mapped = []
            nxt = 0
            for v in col_vals:
                if v not in mapping:
                    mapping[v] = nxt
                    nxt += 1
                mapped.append(mapping[v])
            prepared.append(np.array(mapped, dtype=float))

    idx = header.index('HeartDisease')
    y = prepared[idx].astype(int)
    X_cols = [prepared[i] for i in range(len(prepared)) if i != idx]
    X = np.vstack(X_cols).T.astype(float)
    return X, y


def main():
    data_path = os.path.join(ROOT, 'data', 'heart.csv')
    X, y = load_and_prepare(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"accuracy: {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"f1: {f1:.4f}")

    if acc >= 0.7:
        print("Requirement met: accuracy >= 70%")
    else:
        print("Requirement not met: accuracy < 70%")


if __name__ == '__main__':
    main()
