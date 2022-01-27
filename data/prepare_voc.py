# Code adopted from https://github.com/bingykang/Fewshot_Detection

import os
import sys
import argparse
import logging
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--DATA-ROOT', required=True, help='Directory to store VOC data')

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("VOC-Setup")

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def download_data(args):
    logger.info("Downloading Data")
    os.system('mkdir {}'.format(args.DATA_ROOT))
    os.system('wget -P {} https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'.format(args.DATA_ROOT))
    os.system('wget -P {} https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar'.format(args.DATA_ROOT))
    os.system('wget -P {} https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar'.format(args.DATA_ROOT))

    logger.info("Extracting Data")
    os.system('tar -xf {}/VOCtrainval_11-May-2012.tar -C {}'.format(args.DATA_ROOT, args.DATA_ROOT))
    os.system('tar -xf {}/VOCtrainval_06-Nov-2007.tar -C {}'.format(args.DATA_ROOT, args.DATA_ROOT))
    os.system('tar -xf {}/VOCtest_06-Nov-2007.tar -C {}'.format(args.DATA_ROOT, args.DATA_ROOT))

    logger.info("Data Extracted")
    os.system('rm -f {}/VOCtrainval_11-May-2012.tar'.format(args.DATA_ROOT))
    os.system('rm -f {}/VOCtrainval_06-Nov-2007.tar'.format(args.DATA_ROOT))
    os.system('rm -f {}/VOCtest_06-Nov-2007.tar'.format(args.DATA_ROOT))

def setup_data(args):
    wd = os.path.abspath(args.DATA_ROOT)
    logging.info("Setting Up Data")
    def convert(size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)

    def convert_annotation(year, image_id):
        in_file = open('%s/VOCdevkit/VOC%s/Annotations/%s.xml'%(wd, year, image_id))
        out_file = open('%s/VOCdevkit/VOC%s/labels/%s.txt'%(wd, year, image_id), 'w')
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def convert_annotation_meta(year, image_id, class_name):
        in_file = open('%s/VOCdevkit/VOC%s/Annotations/%s.xml'%(wd, year, image_id))
        out_file = open('%s/VOCdevkit/VOC%s/labels_1c/%s/%s.txt'%(wd, year, class_name, image_id), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls != class_name or int(difficult) == 1:
                continue
            # cls_id = classes.index(cls)
            cls_id = 0
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    
    for year, image_set in sets:
        if not os.path.exists('%s/VOCdevkit/VOC%s/labels/'%(wd, year)):
            os.makedirs('%s/VOCdevkit/VOC%s/labels/'%(wd, year))
        image_ids = open('%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(wd, year, image_set)).read().strip().split()
        list_file = open('%s/%s_%s.txt'%(wd, year, image_set), 'w')
        for image_id in image_ids:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
            convert_annotation(year, image_id)
        list_file.close()
    os.system('cat {}/2007_train.txt {}/2007_val.txt {}/2012_*.txt > {}/voc_train.txt'.format(wd, wd, wd, wd))
    if not os.path.exists('%s/voclist'%(wd)):
        os.mkdir('%s/voclist'%(wd))
    for class_name in classes:
        for year, image_set in sets:
            image_ids = open('%s/VOCdevkit/VOC%s/ImageSets/Main/%s_%s.txt'%(wd, year, class_name, image_set)).read().strip().split()
            ids, flags = image_ids[::2], image_ids[1::2]
            image_ids = list(zip(ids, flags))

            # File to save the image path list
            list_file = open('%s/voclist/%s_%s_%s.txt'%(wd, year, class_name, image_set), 'w')

            # File to save the image labels
            label_dir = 'labels_1c/' + class_name
            if not os.path.exists('%s/VOCdevkit/VOC%s/%s/'%(wd, year, label_dir)):
                os.makedirs('%s/VOCdevkit/VOC%s/%s/'%(wd, year, label_dir))

            # Traverse all images
            for image_id, flag in image_ids:
                if int(flag) == -1:
                    continue
                list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
                convert_annotation_meta(year, image_id, class_name)
            list_file.close()

        files = [
            '{}/voclist/2007_{}_train.txt'.format(wd, class_name),
            '{}/voclist/2007_{}_val.txt'.format(wd, class_name),
            '{}/voclist/2012_{}_*.txt'.format(wd, class_name)
        ]
        files = ' '.join(files)
        cmd = 'cat ' + files + '> {}/voclist/{}_train.txt'.format(wd, class_name)
        os.system(cmd)
    logging.info("Data Setup Completed")

def generate_fewshot(args):
    # Use splits mentioned in https://github.com/bingykang/Fewshot_Detection/
    logging.info("Generating Fewshot Files")
    wd = os.path.abspath(args.DATA_ROOT)
    print ('git clone https://github.com/bingykang/Fewshot_Detection.git {}'.format(wd))
    os.system('git clone https://github.com/bingykang/Fewshot_Detection.git {}/Fewshot_Detection/'.format(wd))
    tgt_folder = os.path.join(wd, 'voclist')
    src_folder = os.path.join(wd, 'Fewshot_Detection/data/vocsplit')
    for name_list in sorted(os.listdir(src_folder)):
        print('  | On ' + name_list)
        # Read from src
        with open(os.path.join(src_folder, name_list), 'r') as f:
            names = f.readlines()
        
        # Replace data root
        names = [name.replace('/scratch/bykang/datasets', wd) for name in names]
        
        with open(os.path.join(wd, 'voclist', name_list), 'w') as f:
            f.writelines(names)

    for fname in ['voc_traindict_full.txt',
              'voc_traindict_bbox_1shot.txt',
              'voc_traindict_bbox_2shot.txt',
              'voc_traindict_bbox_3shot.txt',
              'voc_traindict_bbox_5shot.txt',
              'voc_traindict_bbox_10shot.txt']: 
        full_name = os.path.join(wd, fname)
        print('  | On ' + full_name)
        # Read lines
        with open(os.path.join(wd, 'Fewshot_Detection', 'data', fname), 'r') as f:
            lines = f.readlines()

        # Replace data root
        lines = [line.replace('/scratch/bykang/datasets', wd) 
                for line in lines]
        lines = [line.replace('/home/bykang/voc', wd) 
                for line in lines]

        # Rewrite linea
        with open(full_name, 'w') as f:
            f.writelines(lines)
    os.system('cp {}/Fewshot_Detection/data/voc_novels.txt {}/'.format(wd, wd))
    os.system('cp {}/Fewshot_Detection/data/voc.names {}/'.format(wd, wd))
    
    logging.info("Generated Fewshot Files")
    os.system('rm -rf {}/Fewshot_Detection'.format(wd))

def update_paths(args):
    main_dir = getcwd()
    data_dir = os.path.abspath(args.DATA_ROOT)
    
    base_yaml_path = os.path.join(main_dir, 'data', 'pipelines_adaptor', 'voc', 'config_base_training.yaml')
    ft_yaml_path = os.path.join(main_dir, 'data', 'pipelines_adaptor', 'voc', 'config_fine_tuning.yaml')
    with open(base_yaml_path, 'r') as f:
        base_yaml = yaml.safe_load(f)
    with open(ft_yaml_path, 'r') as f:
        ft_yaml = yaml.safe_load(f)
    for yaml_file in [base_yaml, ft_yaml]:
        yaml_file['code_root'] = yaml_file['code_root'].replace('/cpu008/skhandel/FewshotDetection/UniT', main_dir)
        for key in ['data_root', 'novel', 'meta', 'train', 'valid']:
            yaml_file[key] = yaml_file[key].replace('/cpu008/skhandel/FewshotDetection/UniT/data/VOC', data_dir)
    with open(os.path.join(main_dir, 'data', 'pipelines_adaptor', 'voc', 'config_base_training.yaml'), 'w') as f:
        yaml.dump(base_yaml, f)
    with open(os.path.join(main_dir, 'data', 'pipelines_adaptor', 'voc', 'config_fine_tuning.yaml'), 'w') as f:
        yaml.dump(ft_yaml, f)

if __name__ == '__main__':
    args = parser.parse_args()
    download_data(args)
    setup_data(args)
    generate_fewshot(args)
    update_paths(args)
