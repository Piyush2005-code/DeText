import joblib
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


CHAR_VOCAB = 65536
CHAR_EMBED = 128
NUM_FILTERS = 256
FILTER_SIZES = [2, 3, 4, 5, 6, 7]
MAX_SEQ_LEN = 768

# Neural architectures constants
NUM_CLASSES = 235

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
if torch.backends.mps.is_available():
    device = "mps"

DEVICE = torch.device(device)
# Apple MPS does not support EmbeddingBag — force CPU for those models
CPU_DEVICE = torch.device("cpu")

def char_ngrams(text, min_n=2, max_n=4):
    text = f" {text} "
    ngrams = []
    for n in range(min_n, max_n + 1):
        ngrams += [text[i:i+n] for i in range(len(text) - n + 1)]
    return ngrams

def hash_ngram(ngram, bucket_size):
    h = 2166136261
    for ch in ngram.encode("utf-8", errors="replace"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return (h % (bucket_size - 1)) + 1


WEIGHTS_DIR = Path(__file__).parent / "weights"


def clean_text(text: str) -> list:
    return [text]


class CharCNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conv_results = []
        for conv, bn in zip(self.convs, self.bns):
            c = F.relu(bn(conv(x)))
            pooled = torch.max(c, dim=2)[0]
            conv_results.append(pooled)
        out = torch.cat(conv_results, dim=1)
        out = self.dropout(out)
        return self.fc(out)


class CharCNN:
    def __init__(self):
        self.model = CharCNNModel(vocab_size=CHAR_VOCAB, embed_dim=CHAR_EMBED,
                         num_classes=NUM_CLASSES, filter_sizes=FILTER_SIZES, num_filters=NUM_FILTERS)
        self.model.load_state_dict(torch.load(WEIGHTS_DIR / "charcnn_highcap_weights.pth", map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def __call__(self, text: str):
        text = f" {text} "[:MAX_SEQ_LEN]
        # Using ord(ch) instead of hash(ch) for stability across processes
        ids = [((ord(ch) % (CHAR_VOCAB - 1)) + 1) for ch in text]
        if len(ids) < 7:
            ids += [0] * (7 - len(ids))
        x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            preds = self.model(x).argmax(dim=1)
        return self.label_encoder.inverse_transform(preds.cpu().numpy())[0]



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


##Neural Network Models

class FastTextModel(nn.Module):
    def __init__(self, bucket_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(bucket_size, embed_dim, mode="mean", padding_idx=0)
        self.classifier = nn.AdaptiveLogSoftmaxWithLoss(
            embed_dim, num_classes, cutoffs=[num_classes//4, num_classes//2], div_value=2.0
        )
    def forward(self, x):
        emb = self.embedding(x)
        return self.classifier.log_prob(emb)

class FastText:
    def __init__(self):
        self.bucket_size = 2_000_000
        self.model = FastTextModel(self.bucket_size, 64, NUM_CLASSES)
        self.model.load_state_dict(torch.load(WEIGHTS_DIR / "fasttext_weights.pth", map_location=CPU_DEVICE))
        self.model.to(CPU_DEVICE).eval()
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def __call__(self, text: str):
        ngrams = char_ngrams(text, 2, 4)
        ids = [hash_ngram(g, self.bucket_size) for g in ngrams] if ngrams else [1]
        x = torch.tensor([ids], dtype=torch.long).to(CPU_DEVICE)
        with torch.no_grad():
            preds = self.model(x).argmax(dim=1)
        return self.label_encoder.inverse_transform(preds.cpu().numpy())[0]

class GlotLIDModel(nn.Module):
    def __init__(self, bucket_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(bucket_size, embed_dim, mode="mean", padding_idx=0)
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        return self.classifier(self.embedding(x))

class GlotLID:
    def __init__(self):
        self.bucket_size = 2_000_000
        self.model = GlotLIDModel(self.bucket_size, 128, NUM_CLASSES)
        self.model.load_state_dict(torch.load(WEIGHTS_DIR / "glotlid_weights.pth", map_location=CPU_DEVICE))
        self.model.to(CPU_DEVICE).eval()
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def __call__(self, text: str):
        ngrams = char_ngrams(text, 2, 5)
        ids = [hash_ngram(g, self.bucket_size) for g in ngrams] if ngrams else [1]
        x = torch.tensor([ids], dtype=torch.long).to(CPU_DEVICE)
        with torch.no_grad():
            preds = self.model(x).argmax(dim=1)
        return self.label_encoder.inverse_transform(preds.cpu().numpy())[0]

class CLD3Model(nn.Module):
    def __init__(self, bucket_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.emb1 = nn.EmbeddingBag(bucket_size, embed_dim, mode="mean", padding_idx=0)
        self.emb2 = nn.EmbeddingBag(bucket_size, embed_dim, mode="mean", padding_idx=0)
        self.emb3 = nn.EmbeddingBag(bucket_size, embed_dim, mode="mean", padding_idx=0)
        self.hidden = nn.Linear(embed_dim * 3, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, uni, bi, tri):
        x = torch.cat([self.emb1(uni), self.emb2(bi), self.emb3(tri)], dim=1)
        x = self.relu(self.hidden(x))
        return self.classifier(x)

class CLD3:
    def __init__(self):
        self.bucket_size = 1_000_000
        self.model = CLD3Model(self.bucket_size, 64, 256, NUM_CLASSES)
        self.model.load_state_dict(torch.load(WEIGHTS_DIR / "cld3_weights.pth", map_location=CPU_DEVICE))
        self.model.to(CPU_DEVICE).eval()
        self.label_encoder = joblib.load(WEIGHTS_DIR / "label_encoder.pkl")

    def _get_ids(self, text, n):
        text = f" {text} "
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        if not ngrams: return [1]
        return [hash_ngram(g, self.bucket_size) for g in ngrams]

    def __call__(self, text: str):
        uni = torch.tensor([self._get_ids(text, 1)], dtype=torch.long).to(CPU_DEVICE)
        bi = torch.tensor([self._get_ids(text, 2)], dtype=torch.long).to(CPU_DEVICE)
        tri = torch.tensor([self._get_ids(text, 3)], dtype=torch.long).to(CPU_DEVICE)
        with torch.no_grad():
            preds = self.model(uni, bi, tri).argmax(dim=1)
        return self.label_encoder.inverse_transform(preds.cpu().numpy())[0]


##Testing modules
if __name__ == "__main__":
    test_sentence = "Hello, My name is Piyush Singh Bhati"

    nb = ComplementNB()
    print(f"ComplementNB: {nb(test_sentence)}")
    
    cnn = CharCNN()
    print(f"CharCNN: {cnn(test_sentence)}")
