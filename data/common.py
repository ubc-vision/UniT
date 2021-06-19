import math
import random
import torch.utils.data as data

from detectron2.utils.serialize import PicklableWrapper


__all__ = ["SupportExamplesSampler", "MapSupportDataset"]


class SupportExamplesSampler(data.Dataset):
    """
    flag_eval [boolean]: if True, sample deterministically
    """
    def __init__(self, dataset, num_shots, flag_eval=False):
        self._dataset = dataset
        self._num_shots = num_shots
        self._flag_eval = flag_eval

        if self._flag_eval:
            assert isinstance(self._num_shots, int)

    def __len__(self):
        if not self._flag_eval:
            return 999999
        else:
            min_num_images_across_classes = min([len(v) for k, v in self._dataset.items()])
            return max(
                math.floor(min_num_images_across_classes / self._num_shots),
                1
            )

    def __getitem__(self, idx):
        """
        - num_shot is sampled from the provided list
        """
        _dataset_sampled = {}
        if not self._flag_eval:
            num_shots = random.choices(self._num_shots, k=1)[0]
            for id_class, annotations_class in self._dataset.items():
                _dataset_sampled[id_class] = random.choices(annotations_class, k=num_shots)
        else:
            """
            - idx used here to make val dataset's sampling deterministic
            """
            for id_class, annotations_class in self._dataset.items():
                _dataset_sampled[id_class] = []

                idx_start = idx * self._num_shots
                if idx_start < len(annotations_class):
                    idx_end = min(len(annotations_class), ((idx + 1) * self._num_shots))
                    _dataset_sampled[id_class] = annotations_class[
                        idx_start: idx_end
                    ]

        return _dataset_sampled


class MapSupportDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.
    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        _dataset_sampled = self._dataset[idx]
        _dataset_dict_loaded = {}
        for id_class, annotations_class in _dataset_sampled.items():
            _dataset_dict_loaded[id_class] = [self._map_func(e) for e in annotations_class]
        return _dataset_dict_loaded