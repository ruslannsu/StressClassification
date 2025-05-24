import numpy as np
import src.data_load
from src.data_load import data_load


class GradModel:
    def __init__(self, data_frame_path):
        self.params = np.array([0.0, 0.0, 0.0, 0.0])
        self.learning_rate = np.array([0.1, 0.01, 0.001, 0.0001])
        self.data_loader = data_load.DataLoad(data_frame_path)
        self.N = 2000
        self.data_size = 1000
    def loss_func(self, x, y):
        w = self.params
        margin = np.dot(w, x) * y
        return 2 / (1 + np.exp(margin))

    def d_loss_func(self, x, y):
        margin = np.dot(self.params, x) * y
        return -2 * (1 + np.exp(margin)) ** (-2) * np.exp(margin) * x * y

    def fit(self):
        for i in range(self.N):
            dQ = 0
            for j in range(self.data_size):
                dQ = self.w - np.dot(self.learning_rate, self)




model = GradModel("../../data/synthetic_stress_data.csv")
model.data_loader.load_data()
model.data_loader.explore_data()
print(53)
