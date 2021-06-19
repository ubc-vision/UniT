from .base_training import BaseTrainData
from .fine_tuning import FineTuneData


class RegisterCOCO:
    def __init__(self, split_id, num_shots):
        self.split_id = split_id
        self.num_shots = num_shots

    def _register_datasets(self, ):
        self.base_train_data_instance = BaseTrainData(
            novel_id=self.split_id - 1
        )
        self.fine_tune_data_instance = FineTuneData(
            novel_id=self.split_id - 1,
            num_shots=self.num_shots
        )

        self.base_train_data_instance._register_datasets()
        self.fine_tune_data_instance._register_datasets()