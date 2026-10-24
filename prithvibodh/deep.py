"""Deep models from the slides: CNN, Transformer and U-Net (PyTorch, free Colab GPU).

* TempCNN      - 1-D CNN over the season sequence of each pixel
* Transformer  - self-attention over season tokens of each pixel
* U-Net        - spatial model on image chips, trained from sparse point labels
                 (only labelled pixels contribute to the loss)
"""
import numpy as np
import torch
from torch import nn

from .features import to_temporal

IGNORE = 255


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype("float32")
    w = counts.sum() / np.maximum(counts, 1) / n_classes
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------- per-pixel temporal models
class TempCNN(nn.Module):
    def __init__(self, n_feat, n_seasons, n_classes, width=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout),
            nn.Flatten(), nn.Linear(width * n_seasons, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes))

    def forward(self, x):            # x: (batch, seasons, features)
        return self.net(x.transpose(1, 2))


class SeasonTransformer(nn.Module):
    def __init__(self, n_feat, n_seasons, n_classes, d_model=64, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.season_emb = nn.Parameter(torch.zeros(1, n_seasons, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 128, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        h = self.encoder(self.proj(x) + self.season_emb)
        return self.head(h.mean(dim=1))


class TemporalModel:
    """sklearn-style wrapper: fit / predict_proba on the full feature matrix."""

    def __init__(self, kind, cfg, n_classes):
        self.kind, self.cfg, self.n_classes = kind, cfg, n_classes
        self.p = cfg["models"]["temporal"]

    def _prep(self, X_all):
        x = to_temporal(np.asarray(X_all, dtype="float32"), self.cfg)
        return (x - self.mean) / self.std

    def fit(self, X_all, y):
        set_seed(self.cfg["seed"])
        x = to_temporal(np.asarray(X_all, dtype="float32"), self.cfg)
        self.mean = x.reshape(-1, x.shape[-1]).mean(0)
        self.std = x.reshape(-1, x.shape[-1]).std(0) + 1e-6
        x = (x - self.mean) / self.std
        n_seasons, n_feat = x.shape[1], x.shape[2]
        Net = TempCNN if self.kind == "cnn" else SeasonTransformer
        self.net = Net(n_feat, n_seasons, self.n_classes).to(device())
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.p["lr"], weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, self.n_classes).to(device()))
        X_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        self.net.train()
        for _ in range(self.p["epochs"]):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), self.p["batch_size"]):
                b = perm[i:i + self.p["batch_size"]]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(self.net(X_t[b].to(device())), y_t[b].to(device()))
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X_all, batch=8192):
        self.net.eval()
        x = torch.tensor(self._prep(X_all), dtype=torch.float32)
        out = [torch.softmax(self.net(x[i:i + batch].to(device())), 1).cpu().numpy()
               for i in range(0, len(x), batch)]
        return np.concatenate(out).astype("float32")


# ---------------------------------------------------------------- U-Net
def _block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, in_ch, n_classes, base=32):
        super().__init__()
        self.e1, self.e2, self.e3 = _block(in_ch, base), _block(base, base * 2), _block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):            # x: (batch, bands, H, W), H and W divisible by 4
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def read_chips(stack_path, df, cfg, chip=64):
    """Cut chips centred on each point from the exported int16 stack.

    Returns X (n, bands, chip, chip) in real units and a sparse label mask Y where
    every labelled point of df that falls inside the chip is marked.
    """
    import rasterio
    from rasterio.warp import transform

    from .features import scale_factors

    scales = scale_factors(cfg)[:, None, None]
    half = chip // 2
    with rasterio.open(stack_path) as src:
        xs, ys = transform("EPSG:4326", src.crs, df["lon"].tolist(), df["lat"].tolist())
        rc = [src.index(x, y) for x, y in zip(xs, ys)]
        rows = np.array([r for r, _ in rc])
        cols = np.array([c for _, c in rc])
        labels = df["label"].to_numpy()
        X, Y = [], []
        for r, c in zip(rows, cols):
            win = rasterio.windows.Window(c - half, r - half, chip, chip)
            arr = src.read(window=win, boundless=True, fill_value=0).astype("float32") / scales
            mask = np.full((chip, chip), IGNORE, dtype="int64")
            inside = (np.abs(rows - r) < half) & (np.abs(cols - c) < half)
            for rr, cc, lab in zip(rows[inside], cols[inside], labels[inside]):
                mask[rr - r + half, cc - c + half] = lab
            X.append(arr)
            Y.append(mask)
    return np.stack(X), np.stack(Y)


class UNetModel:
    def __init__(self, cfg, n_classes):
        self.cfg, self.n_classes, self.p = cfg, n_classes, cfg["models"]["unet"]

    def fit(self, X, Y):
        set_seed(self.cfg["seed"])
        self.mean = X.mean(axis=(0, 2, 3), keepdims=True)
        self.std = X.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        Xn = torch.tensor((X - self.mean) / self.std, dtype=torch.float32)
        Yt = torch.tensor(Y, dtype=torch.long)
        y_lab = Y[Y != IGNORE]
        self.net = UNet(X.shape[1], self.n_classes).to(device())
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.p["lr"], weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_lab, self.n_classes).to(device()),
                                      ignore_index=IGNORE)
        self.net.train()
        for _ in range(self.p["epochs"]):
            perm = torch.randperm(len(Xn))
            for i in range(0, len(Xn), self.p["batch_size"]):
                b = perm[i:i + self.p["batch_size"]]
                xb, yb = Xn[b].to(device()), Yt[b].to(device())
                if torch.rand(1).item() < 0.5:                       # flip augmentation
                    xb, yb = xb.flip(-1), yb.flip(-1)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba_chips(self, X, batch=16):
        self.net.eval()
        Xn = torch.tensor((X - self.mean) / self.std, dtype=torch.float32)
        out = [torch.softmax(self.net(Xn[i:i + batch].to(device())), 1).cpu().numpy()
               for i in range(0, len(Xn), batch)]
        return np.concatenate(out)                    # (n, classes, H, W)

    def predict_proba_centre(self, X):
        p = self.predict_proba_chips(X)
        h = X.shape[-1] // 2
        return p[:, :, h, h].astype("float32")
