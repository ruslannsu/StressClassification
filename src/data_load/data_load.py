import pandas as pd

class DataLoad:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_frame = None
    def load_data(self):
        self.data_frame = pd.read_csv(self.data_path)

    def explore_data(self):
        self.data_frame.head()
        self.data_frame.describe()
        self.data_frame.info()
        self.data_frame.count()
        self.data_frame.describe()

    def all_to_num(self):
        self.data_frame = self.data_frame.astype('float64')

    def get_x(self):
        return self.data_frame.drop(columns = ["stress_level"]).values

    def get_y(self):
        return self.data_frame["stress_level"].values

