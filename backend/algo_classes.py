import joblib
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"


def clean_text(text: str) -> list:
    return [text]


class ComplementNB:
    def __init__(self):
        self.model = joblib.load(WEIGHTS_DIR / "clf_ComplementNB.pkl")
        self.vectorizer = joblib.load(WEIGHTS_DIR / "vectorizer_char_wb_2_4.pkl")
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def __call__(self, string: str):
        features = self.vectorizer.transform(clean_text(string))
        prediction = self.model.predict(features)
        return self.label_encoder.inverse_transform(prediction)[0]


class LinearSVC:
    def __init__(self):
        self.model = joblib.load(WEIGHTS_DIR / "clf_LinearSVC.pkl")
        self.vectorizer = joblib.load(WEIGHTS_DIR / "vectorizer_char_wb_2_4.pkl")
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def __call__(self, string: str):
        features = self.vectorizer.transform(clean_text(string))
        prediction = self.model.predict(features)
        return self.label_encoder.inverse_transform(prediction)[0]


class PassiveAggressive:
    def __init__(self):
        self.model = joblib.load(WEIGHTS_DIR / "clf_PassiveAggressive.pkl")
        self.vectorizer = joblib.load(WEIGHTS_DIR / "vectorizer_char_wb_2_4.pkl")
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def __call__(self, string: str):
        features = self.vectorizer.transform(clean_text(string))
        prediction = self.model.predict(features)
        return self.label_encoder.inverse_transform(prediction)[0]


class RidgeClassifier:
    def __init__(self):
        self.model = joblib.load(WEIGHTS_DIR / "clf_RidgeClassifier.pkl")
        self.vectorizer = joblib.load(WEIGHTS_DIR / "vectorizer_char_wb_2_4.pkl")
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def __call__(self, string: str):
        features = self.vectorizer.transform(clean_text(string))
        prediction = self.model.predict(features)
        return self.label_encoder.inverse_transform(prediction)[0]


class SGDClassifier:
    def __init__(self):
        self.model = joblib.load(WEIGHTS_DIR / "clf_SGDClassifier.pkl")
        self.vectorizer = joblib.load(WEIGHTS_DIR / "vectorizer_char_wb_2_4.pkl")
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def __call__(self, string: str):
        features = self.vectorizer.transform(clean_text(string))
        prediction = self.model.predict(features)
        return self.label_encoder.inverse_transform(prediction)[0]

class LangdetectStyleComplementNB:
    def __init__(self):
        self.model = joblib.load(WEIGHTS_DIR / "langdetect_style_complement_nb.pkl")
        self.vectorizer = joblib.load(WEIGHTS_DIR / "vectorizer_char_wb_1_3_langdetect.pkl")
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def __call__(self, string: str):
        features = self.vectorizer.transform(clean_text(string))
        prediction = self.model.predict(features)
        return self.label_encoder.inverse_transform(prediction)[0]

if __name__ == "__main__":
    complementNB = ComplementNB()
    print(complementNB("Hello, My name is Piyush Singh Bhati"))
