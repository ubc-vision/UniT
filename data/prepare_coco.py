import os
import sys
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--DATA-ROOT', required=True, help='Directory to store COCO data')

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger("COCO-Setup")

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
