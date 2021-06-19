from .base_training import BaseTrainData
from .fine_tuning import FineTuneData

class RegisterCOCONote:
    def __init__(self, num_shots):
        self.num_shots = num_shots

    def _register_datasets(self, ):
        self.base_train_data_instance = BaseTrainData()
        self.fine_tune_data_instance = FineTuneData(num_shots=self.num_shots)

        self.base_train_data_instance._register_datasets()
        self.fine_tune_data_instance._register_datasets()