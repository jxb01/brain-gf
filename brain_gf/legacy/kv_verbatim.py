import numpy as np

class LayeredKV_Brain:
    def __init__(self, dim=8, hid=16, slots=16):
        self.K1 = np.random.randn(slots, dim) * 0.5
        self.V1 = np.random.randn(slots, dim) * 0.5
        self.K2 = np.random.randn(slots, hid) * 0.5
        self.V2 = np.random.randn(slots, hid) * 0.5
        self.lr_k = 0.02
        self.lr_v = 0.05

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, x):
        x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        attn1 = self.softmax(x_norm @ self.K1.T)
        h1 = attn1 @ self.V1
        h1_norm = h1 / (np.linalg.norm(h1, axis=-1, keepdims=True) + 1e-6)
        attn2 = self.softmax(h1_norm @ self.K2.T)
        pred = attn2 @ self.V2
        return pred, attn1, attn2, h1, h1_norm

    def step(self, x, y_target):
        x = x.reshape(1, -1)
        pred, attn1, attn2, h1, h1_norm = self.forward(x)
        err = y_target - pred
        delta_K2 = attn2.T @ (err * h1_norm)
        self.K2 += self.lr_k * delta_K2
        delta_V2 = attn2.T @ err
        self.V2 += self.lr_v * delta_V2
        err_h1 = err @ self.V2.T
        delta_K1 = attn1.T @ (err_h1 * x_norm)
        self.K1 += self.lr_k * delta_K1
        delta_V1 = attn1.T @ err_h1
        self.V1 += self.lr_v * delta_V1
        for M in [self.K1, self.V1, self.K2, self.V2]:
            norm = np.linalg.norm(M, axis=-1, keepdims=True)
            M *= np.minimum(1.0, 10.0 / np.maximum(norm, 1e-9))
        return pred
