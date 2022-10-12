import torch
from ignite.engine import Events
from modified_ignite_engine import create_supervised_trainer, create_supervised_evaluator
from IQAdataset import IQADataset
from torch.utils.data import DataLoader
from IQAmodel import QAModel
from QAloss import QALoss
from QAperformance import QAPerformance
import os
import numpy as np
import random
import logger
from argparse import ArgumentParser


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QAModel(arch=args.arch, pretrained=args.pretrained, pool_mode=args.pool_mode).to(device)  #
    test_dataset = IQADataset(args, 'test')
    test_loader = DataLoader(test_dataset, batch_size=2*args.batch_size, num_workers=16)

    try:
        checkpoint = torch.load(args.trained_model_file)
    except:
        checkpoint = torch.load(args.trained_model_file.replace('pmode=mean-', ''))
    model.load_state_dict(checkpoint['model'])
    evaluator = create_supervised_evaluator(model, metrics={'IQA_performance': QAPerformance()}, device=device)
    evaluator.run(test_loader)
    performance = evaluator.state.metrics
    # TODO: PLCC after nonlinear mapping when conducting cross-dataset evaluation
    metrics_printed = ['SROCC', 'KROCC', 'PLCC']
    for metric_print in metrics_printed:
        logger.log.info('{}, {}: {:.5f}'.format(args.dataset, metric_print, performance[metric_print].item()))
    np.save(args.save_result_file, performance)
    logger.destroy_logger()
     
if __name__ == "__main__":
    parser = ArgumentParser(description='')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument("--exp_id", type=int, default=0,
                        help='the exp split idx (default: 0)')
    parser.add_argument("--model", type=str, default='IQA')
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='arch name (default: ... resnext101_32x8d)')
    parser.add_argument('--pool_mode', default='mean', type=str,
                        help='pool mode (default: mean, ..., mean+std, std)')
    parser.add_argument('-pretrained', '--pretrained', type=int, default=1,
                        help='fe network init mode (default: 0 for default random, 1 for ImageNet-pretrained)')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='batch size for training (default: 8)')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('-ft_lr_ratio', '--ft_lr_ratio', type=float, default=0.1,
                        help='ft_lr_ratio for fe (default: 0.1)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('-lrd', '--lr_decay', type=float, default=0.1,
                        help='lr decay (default: 0.1)')
    parser.add_argument('-olrd', '--overall_lr_decay', type=float, default=0.01,
                        help='overall lr decay (default: 0.01)')
    parser.add_argument('-nrz', '--noresize', action='store_true',
                        help='No Resize?')
    parser.add_argument('-rs_h', '--resize_size_h', default=498, type=int,
                        help='resize_size_h (default: 498)')
    parser.add_argument('-rs_w', '--resize_size_w', default=664, type=int,
                        help='resize_size_w (default: 664)')
    parser.add_argument('-random', '--randomness', action='store_true',
                        help='Allow randomness during training?') 
    parser.add_argument('-debug', '--debug', action='store_true',
                        help='Debug?') 
    
    parser.add_argument('--trained_model_file', default=None, type=str,
                        help='trained_model_file')
    
    args = parser.parse_args()

    args.dataset = 'CLIVE'  # ln -s database_path xxx
    args.data_info = './data/CLIVEinfo.mat'
    args.train_ratio = 0.0
    args.train_and_val_ratio = 0.0

    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.trained_model_file is None:
        args.format_str = 'IQA-{}-pretrained={}-pmode={}-resize={}-lr={}-bs={}-e={}-ftlrr={}-exp{}'\
                      .format(args.arch, args.pretrained, args.pool_mode, not args.noresize,
                              args.lr, args.batch_size, args.epochs, args.ft_lr_ratio, args.exp_id)
        args.trained_model_file = 'checkpoints/' + args.format_str

    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/dataset={}-tested_on_{}'.format(args.dataset, os.path.split(args.trained_model_file)[1])
    logger.create_logger('logs', 'dataset={}-tested_on_{}'.format(args.dataset, os.path.split(args.trained_model_file)[1]), False)
    logger.log.info(args)
    run(args)
