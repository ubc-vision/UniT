import os
import glob
import pickle
import numpy as np

from tqdm import tqdm
from scipy.io import loadmat


PATH_PROPOSALS_ROOT = "/h/skhandel/FewshotDetection/WSASOD/data/data_utils/data/coco_proposals/"

PATH_MCG_TRAIN = os.path.join(PATH_PROPOSALS_ROOT, 'MCG-COCO-train2014-boxes')
PATH_MCG_VAL = os.path.join(PATH_PROPOSALS_ROOT, 'MCG-COCO-val2014-boxes')

assert(os.path.isdir(PATH_MCG_TRAIN))
assert(os.path.isdir(PATH_MCG_VAL))


def swap_axes(boxes):
    """Swaps x and y axes."""
    boxes = boxes.copy()
    boxes = np.stack((boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]), axis=1)
    return boxes


class CocoMCG:
    def __init__(self, path_mat_files):
        self.ids, self.boxes, self.scores = self.load_data(path_mat_files)

    def load_data(self, path_data):
        ids, boxes, scores = [], [], []

        mat_files = sorted(glob.glob(os.path.join(path_data, '*')))

        for mat_file in tqdm(mat_files):
            loaded_mat_ = loadmat(mat_file)

            boxes_ = loaded_mat_['boxes']
            boxes_ = self._preprocess_boxes(boxes_)

            scores_ = loaded_mat_['scores']
            scores_ = np.squeeze(scores_)

            ids.append(str(int(os.path.splitext(os.path.split(mat_file)[1])[0].split("_")[-1])))
            boxes.append(boxes_)
            scores.append(scores_)

        return ids, boxes, scores

    def _preprocess_boxes(self, boxes):
        # (box_count, 4)
        # dtype: float32

        # box format: (y_min, x_min, y_max, x_max)
        boxes = boxes.astype(np.float32) - 1

        # box format: (x_min, y_min, x_max, y_max)
        boxes = swap_axes(boxes)

        return boxes

    def write_into_det2_format(self, path):
        dict_data = {}
        dict_data['ids'] = self.ids
        dict_data['boxes'] = self.boxes
        dict_data['scores'] = self.scores

        assert not os.path.isfile(path)

        with open(path, 'wb') as fp:
            pickle.dump(dict_data, fp)


if __name__ == "__main__":
    mcg_instance_val = CocoMCG(PATH_MCG_VAL)
    mcg_instance_val.write_into_det2_format(os.path.join(PATH_PROPOSALS_ROOT, "val2014_coco_processed.pkl"))

    mcg_instance_train = CocoMCG(PATH_MCG_TRAIN)
    mcg_instance_train.write_into_det2_format(os.path.join(PATH_PROPOSALS_ROOT, "train2014_coco_processed.pkl"))