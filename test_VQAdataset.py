import torch
from ignite.engine import Events
from modified_ignite_engine import create_supervised_trainer, create_supervised_evaluator
from VQAdataset import VQADataset
from torch.utils.data import DataLoader
from VQAmodel import QAModel
from QAperformance import QAPerformance
import os
import numpy as np
import random
import logger
from argparse import ArgumentParser


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QAModel(arch=args.arch, pretrained=args.pretrained, pool_mode=args.pool_mode).to(device)  #
    features_path = 'LIVEVQC-features-{}-fe_init_mode-{}/'.format(args.arch, args.fe_init_mode)  
    test_dataset = VQADataset(features_path, list(range(args.dataset_len)), feature_extractor=args.arch, groups=args.groups)
    test_loader = DataLoader(test_dataset, batch_size=2*args.batch_size)
    
    try:
        checkpoint = torch.load(args.trained_model_file)
    except:
        checkpoint = torch.load(args.trained_model_file.replace('g=16-', ''))
    model.load_state_dict(checkpoint)
    evaluator = create_supervised_evaluator(model, metrics={'VQA_performance': QAPerformance()}, device=device)
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
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--model", type=str, default='VQA')
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='')
    parser.add_argument('--pool_mode', default='mean', type=str,
                        help='pool mode (default: mean, ..., mean+std, std)')
    parser.add_argument('-fim', '--fe_init_mode', type=int, default=3,
                        help='fe network init mode (default: 0 for default random, 1 for ImageNet-pretrained, 2 for iqa-pretrained, 3 for iqa-finetuned)')
    parser.add_argument('-rim', '--re_init_mode', type=int, default=1,
                        help='re network init mode (default: 0 for default random, 1 for iqa-pretrained)')
    parser.add_argument('-g', '--groups', type=int, default=16,
                        help='number of cube groups in a video (default: 16)')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4,
                        help='learning rate (default: )')
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='input batch size for training (default: )')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='number of epochs to train (default: )')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    
    parser.add_argument('-random', '--randomness', action='store_true',
                        help='Allow randomness during training?') 
    
    parser.add_argument('--trained_model_file', default=None, type=str,
                        help='trained_model_file')
    
    args = parser.parse_args()
    
    if args.fe_init_mode == 0 or args.fe_init_mode == 2:
        args.pretrained = 0
    else:
        args.pretrained = 1

    args.dataset = 'LIVE-VQC'  # ln -s database_path xxx
    args.data_info = './data/LIVE-VQCinfo.mat'
    args.dataset_len = 585

    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.trained_model_file is None:
        args.format_str = 'VQA-{}-fim={}-rim={}-g={}-lr{}-bs{}-e{}-exp{}'.format(args.arch, args.fe_init_mode, args.re_init_mode, 
                                                                                 args.groups, args.lr, args.batch_size, args.epochs, args.exp_id)
        args.trained_model_file = 'checkpoints/' + args.format_str

    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/dataset={}-tested_on_{}'.format(args.dataset, os.path.split(args.trained_model_file)[1])
    logger.create_logger('logs', 'dataset={}-tested_on_{}'.format(args.dataset, os.path.split(args.trained_model_file)[1]), False)
    logger.log.info(args)
    run(args)
