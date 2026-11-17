import numpy as np

class NRIRBrain:
    def __init__(self, dim=8, slots=128, layers=4, topk=8,
                 lr=0.05, lr_k=0.02, g=0.5):
        self.dim = dim
        self.slots = slots
        self.layers = layers
        self.topk = topk
        self.lr = lr
        self.lr_k = lr_k
        self.g = g

        self.Ks = [np.random.randn(slots, dim) * 0.3 for _ in range(layers)]
        self.Vs = [np.random.randn(slots, dim) * 0.3 for _ in range(layers)]
        for l in range(layers):
            self.Vs[l] = self._col_orthogonalize(self.Vs[l])

    def _col_orthogonalize(self, M):
        Q, R = np.linalg.qr(M)
        signs = np.sign(np.diag(R))
        Q = Q * signs
        return Q

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def sparse_attn(self, scores):
        attn = self.softmax(scores)
        k = min(self.topk, self.slots)
        idx = np.argsort(attn)[-k:]
        mask = np.zeros_like(attn)
        mask[idx] = 1.0
        attn = attn * mask
        attn = attn / (np.sum(attn) + 1e-8)
        return attn, idx

    def forward(self, x):
        h = x.reshape(1, -1)
        h = h / (np.linalg.norm(h) + 1e-8)

        attns, idxs, hs = [], [], [h.copy()]

        for l in range(self.layers):
            scores = h @ self.Ks[l].T
            attn, idx = self.sparse_attn(scores)
            delta = attn @ self.Vs[l]

            h_new = (1 - self.g) * h + self.g * delta
            h = h_new / (np.linalg.norm(h_new) + 1e-8)

            attns.append(attn)
            idxs.append(idx)
            hs.append(h.copy())

        return h, attns, idxs, hs

    def step(self, x, y_target=None):
        x = x.reshape(1, -1)
        pred, attns, idxs, hs = self.forward(x)

        err = x / (np.linalg.norm(x) + 1e-8) - pred

        for l in range(self.layers - 1, -1, -1):
            idx = idxs[l]
            attn_active = attns[l][0, idx]

            delta_V = np.zeros_like(self.Vs[l])
            delta_V[idx] = attn_active[:, None] * err
            self.Vs[l] += self.lr * delta_V

            h_in = hs[l][0] if l == 0 else hs[l-1][0]
            h_prev = hs[l] if l > 0 else x[0]
            delta_K = np.zeros_like(self.Ks[l])
            delta_K[idx] = attn_active[:, None] * err * h_prev
            self.Ks[l] += self.lr_k * delta_K

            sigma = np.linalg.norm(self.Vs[l].T, 2)
            alpha = 1.0 / (1.0 + sigma)
            if l > 0:
                err = alpha * (err + err @ self.Vs[l].T)

            for M in [self.Ks[l], self.Vs[l]]:
                norm = np.linalg.norm(M, axis=-1, keepdims=True)
                M *= np.minimum(1.0, 10.0 / np.maximum(norm, 1e-9))

            if hasattr(self, '_step_count'):
                self._step_count += 1
            else:
                self._step_count = 0
            if self._step_count % 100 == 0:
                self.Vs[l] = self._col_orthogonalize(self.Vs[l])

        return pred


if __name__ == "__main__":
    np.random.seed(0)
    brain = NRIRBrain(dim=8, slots=128, layers=4, topk=8, g=0.5)

    def task(x):
        return np.sin(x[:, 0:1]) + x[:, 1:2] * x[:, 2:3]

    X = np.random.randn(2000, 8)
    Y = task(X)

    print("NRIR multi-layer online learning...")
    for epoch in range(10):
        preds = []
        for i in range(len(X)):
            pred = brain.step(X[i], Y[i])
            preds.append(pred.flatten())
        preds = np.array(preds)
        r2 = 1 - np.sum((Y - preds)**2) / np.sum((Y - Y.mean())**2)
        print(f"epoch {epoch:02d}  R2 = {r2:+.4f}")
