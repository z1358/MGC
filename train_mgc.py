import numpy as np
import torch
from trainer.mgc_trainer import MGCTrainer
from utils import (
    pprint, ensure_path,
    get_command_line_parser,
    postprocess_args,
    set_gpu,
)
from logger import Mylog, set_log_path, log

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    ensure_path(args.save_path)
    set_log_path(args.save_path)
    log(vars(args), filename='trainfsl.txt')

    num_gpu = set_gpu(args)
    trainer = MGCTrainer(args)
    trainer.train()
    trainer.evaluate_test(name='max_auroc_instance.pth')
    trainer.final_record()
    print(args.save_path)