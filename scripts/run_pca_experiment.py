import sys, os, time, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from mi import relevance_scores
from graph import build_corr_graph
from selection import select_features
from pca import PCA
from knn import KNN

BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/"
os.makedirs("data", exist_ok=True)

def load(name):
    path = os.path.join("data", name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + name, path)
    a = np.loadtxt(path)
    return a[:, :-1], a[:, -1].astype(int)

def run(label, Xtr, Xte, ytr, yte, nfeat):
    t0 = time.time()
    pred = KNN(5).fit(Xtr, ytr).predict(Xte)
    t = time.time() - t0
    return dict(setting=label, dims=nfeat,
                accuracy=round(accuracy_score(yte, pred), 4),
                macro_f1=round(f1_score(yte, pred, average="macro"), 4),
                time_s=round(t, 2))

def main(synthetic=False):
    if synthetic:
        rng = np.random.default_rng(0)
        Xtr = rng.standard_normal((500, 12)); Xtr[:, 5] = Xtr[:, 0] + .1 * rng.standard_normal(500)
        ytr = (Xtr[:, 0] + Xtr[:, 1] > 0).astype(int) + 1
        Xte = rng.standard_normal((100, 12)); Xte[:, 5] = Xte[:, 0] + .1 * rng.standard_normal(100)
        yte = (Xte[:, 0] + Xte[:, 1] > 0).astype(int) + 1
    else:
        Xtr, ytr = load("sat.trn")
        Xte, yte = load("sat.tst")
    # z-score with training statistics
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    rows = [run("Raw", Xtr, Xte, ytr, yte, Xtr.shape[1])]
    C = np.corrcoef(Xtr.T)
    scores = relevance_scores(Xtr, ytr)
    for t in (0.95, 0.9):
        S = sorted(select_features(scores, build_corr_graph(np.abs(C), t)))
        rows.append(run(f"FS only (t={t})", Xtr[:, S], Xte[:, S], ytr, yte, len(S)))
    p = PCA(0.95).fit(Xtr)
    rows.append(run("PCA only (95% var)", p.transform(Xtr), p.transform(Xte), ytr, yte, len(p.values)))
    for t in (0.95, 0.9):
        S = sorted(select_features(scores, build_corr_graph(np.abs(C), t)))
        p = PCA(0.95).fit(Xtr[:, S])
        rows.append(run(f"FS (t={t}) + PCA", p.transform(Xtr[:, S]), p.transform(Xte[:, S]), ytr, yte, len(p.values)))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if not synthetic:
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/pca_experiment.csv", index=False)

if __name__ == "__main__":
    main(synthetic="--synthetic" in sys.argv)
