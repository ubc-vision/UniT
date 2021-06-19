import pickle
import numpy as np

for file_name in ['voc_2007_test.pkl', 'voc_2007_trainval.pkl', 'voc_2012_trainval.pkl']:
    dir_name = '/h/skhandel/FewshotDetection/WSASOD/data/data_utils/data/voc_proposals/'
    input_file_path = dir_name + file_name
    output_file_path = dir_name + file_name.split(".")[0] + "_detectron.pkl"

    input_data = pickle.load(open(input_file_path, 'rb'))
    output_data = {}
    if file_name != 'voc_2012_trainval.pkl':
        output_data['ids'] = [("%06d" % i) for i in input_data['indexes']]
    else:
        output_data['ids'] = [str(i)[:4] + "_" + str(i)[4:] for i in input_data['indexes']]
    output_data['boxes'] = [x.astype(np.float32) for x in input_data['boxes']]
    output_data['objectness_logits'] = input_data['scores']
    with open(output_file_path, 'wb') as f:
        pickle.dump(output_data, f, -1)