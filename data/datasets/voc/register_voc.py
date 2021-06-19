from .base_training import BaseTrainData
from .fine_tuning import FineTuneData


class RegisterVOC:
    def __init__(self, split_id, num_shots, filter_07_base=False):
        self.split_id = split_id
        self.num_shots = num_shots
        self.filter_07_base = filter_07_base

    def _register_datasets(self, ):
        self.base_train_data_instance = BaseTrainData(
            novel_id=self.split_id - 1,
            filter_07_base=self.filter_07_base
        )
        self.fine_tune_data_instance = FineTuneData(
            novel_id=self.split_id - 1,
            num_shots=self.num_shots
        )

        self.base_train_data_instance._register_datasets()
        self.fine_tune_data_instance._register_datasets()