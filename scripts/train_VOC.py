import sys
import os
import numpy as np
import torch
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets.pascal_voc import register_pascal_voc
from UniT.engine import TrainerNoMeta
from UniT.data import register_datasets
from UniT.configs import add_config
from detectron2.engine import DefaultTrainer

parser = default_argument_parser()


def register_voc_data(args):
    register_pascal_voc("pascal_trainval_2007", args.DATASETS.CLASSIFIER_DATAROOT + 'VOC2007/', 'trainval', 2007)
    register_pascal_voc("pascal_trainval_2012", args.DATASETS.CLASSIFIER_DATAROOT + 'VOC2012/', 'trainval', 2012)
    register_pascal_voc("pascal_test_2007", args.DATASETS.CLASSIFIER_DATAROOT + 'VOC2007/', 'test', 2007)

def setup(args):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    register_voc_data(cfg)
    register_datasets(args_data=cfg)
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="UniT")
    return cfg

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = TrainerNoMeta.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = TrainerNoMeta.test(cfg, model)
        return res
        
    trainer = TrainerNoMeta(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    try:
        # use the last 4 numbers in the job id as the id
        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]

        # all ports should be in the 10k+ range
        default_port = int(default_port) + 15000

    except Exception:
        default_port = 59482
    
    args.dist_url = 'tcp://127.0.0.1:'+str(default_port)
    print (args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )