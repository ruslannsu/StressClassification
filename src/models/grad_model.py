import numpy as np
import src.data_load
from src.data_load import data_load


class GradModel:
    def __init__(self, data_frame_path):
        self.params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.learning_rate = np.array([0.01, 0.001, 0.0001, 0.0001, 0.001, 0.01])
        self.data_loader = data_load.DataLoad(data_frame_path)
        self.data_list = self.data_loader.load_data()
        self.N = 5000
        self.data_size = 1000
    def loss_func(self, x, y):
        w = self.params
        margin = np.clip(np.dot(self.params, x) * y, -100, 100)
        return 2 / (1 + np.exp(margin))

    def d_loss_func(self, x, y):
        margin = np.clip(np.dot(self.params, x) * y, -100, 100)
        return -2 * (1 + np.exp(margin)) ** (-2) * np.exp(margin) * x * y

    def fit(self):
        X = self.data_loader.get_x()
        print(X.shape)
        X = X[1:501]
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        Y = self.data_loader.get_y()
        Y = Y[1:501]
        dQ = np.zeros_like(self.params)
        for i in range(self.N):
            dQ = np.zeros_like(self.params)
            loss_sum = 0
            for j in range(self.data_size // 2):
                dQ += self.d_loss_func(X[j], Y[j])
                loss_sum += self.loss_func(X[j], Y[j])
            self.params = self.params - self.learning_rate * dQ
        print(self.params)
        print("finished")


    def predict(self, x):
        score = np.dot(self.params, x)
        if (score >= 0):
            return 1
        if (score < 0):
            return 0



model = GradModel("../../data/synthetic_stress_data_binary.csv")
model.fit()

J = model.data_loader.get_x()
J = J[500:1001]
J = (J - np.mean(J, axis=0)) / np.std(J, axis=0)
Y = model.data_loader.get_y()
Y = Y[500:1001]
count_good_answers = 0
for i in range(500):
    if (Y[i] == 0) and (model.predict(J[i]) == 0):
        count_good_answers += 1
    if (Y[i] == 1) and (model.predict(J[i]) == 1):
        count_good_answers += 1

print(count_good_answers)

