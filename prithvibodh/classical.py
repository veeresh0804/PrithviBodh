"""Classical models from the slides: Random Forest and SVM."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_rf(cfg):
    p = cfg["models"]["rf"]
    return RandomForestClassifier(n_estimators=p["n_estimators"],
                                  min_samples_leaf=p["min_samples_leaf"],
                                  class_weight="balanced", n_jobs=-1,
                                  random_state=cfg["seed"])


def make_svm(cfg):
    p = cfg["models"]["svm"]
    return make_pipeline(StandardScaler(),
                         SVC(C=p["C"], gamma=p["gamma"], probability=True,
                             class_weight="balanced", random_state=cfg["seed"]))


class ColumnModel:
    """Wrap an sklearn model so it takes the full feature matrix and uses some columns."""

    def __init__(self, model, col_idx, n_classes):
        self.model, self.col_idx, self.n_classes = model, list(col_idx), n_classes

    def fit(self, X_all, y):
        self.model.fit(X_all[:, self.col_idx], y)
        return self

    def predict_proba(self, X_all):
        import numpy as np
        p = self.model.predict_proba(X_all[:, self.col_idx])
        full = np.zeros((len(X_all), self.n_classes), dtype="float32")
        full[:, self.model.classes_] = p
        return full
