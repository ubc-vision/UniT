# FSOD-SPECIFIC IMPORTS
from ...data_utils import parse_cfg, cfg, listDataset, MetaDataset, build_dataset

# GENERAL IMPORTS
import yaml

from torchvision import transforms


class FineTuneFSODAdaptor:
    def __init__(self, novel_id=3, num_shots=10,
                 path_config='config_fine_tuning.yaml',
                 gpus=[0,], num_workers=2,
                 batch_size_query=5, batch_size_support=5,
                 width_query=416, height_query=416,
                 width_support=416, height_support=416):
        # user args
        self.novel_id = novel_id
        self.num_shots = num_shots

        assert(self.novel_id == 3)
        assert(self.num_shots in [10, 30])

        # defaults
        self.gpus = gpus
        self.num_workers = num_workers

        # not used in any downstream usage
        self.width_query = width_query
        self.height_query = height_query

        # not used in any downstream usage
        self.width_support = width_support
        self.height_support = height_support

        # not used in any downstream usage
        self.batch_size_query = batch_size_query
        self.batch_size_support = batch_size_support

        # read config options
        self.path_config = path_config
        self.data_options = self._read_data_options_config(self.path_config)

        # override them with user args
        self.data_options['novelid'] = self.novel_id
        self.data_options['meta'] = \
            self.data_options['meta'].replace('10shot', '{}shot'.format(self.num_shots))

        # config query
        cfg.config_data(self.data_options)

        # config support
        cfg['batch_size'] = self.batch_size_query
        cfg.config_meta({
            'feat_layer': 0,
            'channels': 4,
            'width': self.width_support,
            'height': self.height_support
        })

        # to make cfg accessible to this class's instances. Kinda sad how the
        # original code is written!
        self.cfg = cfg

        # TRAIN
        query_set_filtered = self._pre_filter_query_list(self.data_options)
        self.query_dataset_train = listDataset(
            query_set_filtered,
            shape=(self.width_query, self.height_query),
            shuffle=False,
            transform=transforms.Compose([transforms.ToTensor(),]),
            train=True,
            seen=0,
            batch_size=self.batch_size_query,  # this is not doing anything!
            # `num_workers` as an argument isn't doing anything except updating 
            # `seen` that controls width for data aug
            num_workers=self.num_workers
        )
        # this works both for train and validation case
        self.support_set = MetaDataset(
                    self.data_options['meta'], train=True)

        # VALIDATION
        self.query_dataset_val = listDataset(
            self.data_options['valid'],
            shape=(self.width_query, self.height_query),
            shuffle=False,
            transform=transforms.Compose([transforms.ToTensor(),])
        )

    def _pre_filter_query_list(self, data_options):
        return build_dataset(data_options)

    def _read_data_options_config(self, path):
        with open(path, 'r') as f:
            data_options = yaml.safe_load(f)

        data_options['gpus'] = ','.join([str(e) for e in self.gpus])
        data_options['neg'] = str(data_options['neg'])
        return data_options


if __name__ == "__main__":
    base_pipe_instance = FineTuneFSODAdaptor(novel_id=1, num_shots=10)