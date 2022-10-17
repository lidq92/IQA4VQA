import os
import torch
import logger
import random
import datetime
import numpy as np
from QAloss import QALoss
from ignite.engine import Events
from argparse import ArgumentParser
from VQAdataset import get_data_loaders
from QAperformance import QAPerformance
from VQAmodel import QAModel, WholeQAModel
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from modified_ignite_engine import create_supervised_evaluator, create_supervised_trainer


metrics_printed = ['SROCC', 'KROCC', 'PLCC']
def writer_add_scalar(writer, status, scalars, iter):
    for metric_print in metrics_printed:
        writer.add_scalar('{}/{}'.format(status, metric_print), scalars[metric_print], iter)

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data_loaders(args)
    model = WholeQAModel(args.arch, args.pretrained, args.pool_mode).to(device) if args.whole else QAModel(args.arch).to(device) # QAModel(args.arch, args.pretrained, args.pool_mode).to(device)
    logger.log.info(model)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint)['model'])
    if args.whole:
        optimizer = Adam([{'params': model.regression.parameters()}, 
                          {'params': model.fp.parameters()},
                          {'params': model.dr.parameters()}],
                         lr=args.lr, weight_decay=args.weight_decay)
    else:
        for param in model.features.parameters():
            param.requires_grad = False
        optimizer = Adam([{'params': model.regression.parameters()}, 
                          {'params': model.fp.parameters()},
                          {'params': model.dr.parameters()},
                         {'params': model.features.parameters(), 'lr': args.lr * args.ft_lr_ratio}],
                         lr=args.lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    loss_fn = QALoss()
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)
    evaluator = create_supervised_evaluator(model, metrics={'VQA_performance':QAPerformance()}, device=device)

    if args.inference:
        model.load_state_dict(torch.load(args.trained_model_file))
        evaluator.run(test_loader)
        performance = evaluator.state.metrics
        logger.log.info('SROCC: {}'.format(performance['SROCC']))
        np.save(args.save_result_file, performance)
        return

    writer = SummaryWriter(log_dir='runs/{}-{}'.format(args.format_str, datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    global best_val_criterion, best_epoch
    best_val_criterion, best_epoch = -100, -1  # larger, better, e.g., SROCC/KROCC/PLCC

    @trainer.on(Events.ITERATION_COMPLETED)
    def iter_event_function(engine):
        writer.add_scalar("train/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_event_function(engine):
        evaluator.run(val_loader)
        performance = evaluator.state.metrics
        writer_add_scalar(writer, 'val', performance, engine.state.epoch)
        val_criterion = performance['SROCC']

        evaluator.run(test_loader)
        performance = evaluator.state.metrics
        writer_add_scalar(writer, 'test', performance, engine.state.epoch)

        evaluator.run(train_loader)
        performance = evaluator.state.metrics
        writer_add_scalar(writer, 'train', performance, engine.state.epoch)
        logger.log.info('Train: {} @epoch: {}'.format(performance['SROCC'], engine.state.epoch))

        global best_val_criterion, best_epoch
        if val_criterion > best_val_criterion:
            torch.save(model.state_dict(), args.trained_model_file)
            best_val_criterion = val_criterion
            best_epoch = engine.state.epoch
            logger.log.info('Save current best model @best_val_criterion: {} @epoch: {}'.format(best_val_criterion, best_epoch))
        else:
            logger.log.info('Best model is not updated: SROCC={} @epoch: {}'.format(val_criterion, engine.state.epoch))
            
        scheduler.step()


    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        logger.log.info('best epoch: {}'.format(best_epoch))
        model.load_state_dict(torch.load(args.trained_model_file))
        evaluator.run(test_loader)
        performance = evaluator.state.metrics
        logger.log.info('SROCC: {}'.format(performance['SROCC']))
        np.save(args.save_result_file, performance)
        logger.destroy_logger()

    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    parser = ArgumentParser(description='')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--model", type=str, default='VQA')
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='')
    parser.add_argument('--pool_mode', default='mean', type=str,
                        help='pool mode (default: mean)')
    parser.add_argument('-fim', '--fe_init_mode', type=int, default=3,
                        help='fe network init mode, 0 for default random, 1 for ImageNet-pretrained, 2 for iqa-pretrained, 3 for iqa-finetuned (default: 3)')
    parser.add_argument('-rim', '--re_init_mode', type=int, default=1,
                        help='re network init mode, 0 for default random, 1 for iqa-pretrained (default: 1)')
    parser.add_argument('-g', '--groups', type=int, default=16,
                        help='number of cube groups in a video (default: 16)')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='input batch size for training (default: 8)')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--inference', action='store_true',
                        help='Inference?')
    parser.add_argument('-ft_lr_ratio', '--ft_lr_ratio', type=float, default=0.0,
                        help='ft_lr_ratio for fe (default: 0.0)')
    args = parser.parse_args()
    args.dataset = 'KoNViD-1k' # ln -s database_path xxx
    args.datainfo = 'data/KoNViD-1kinfo.mat'
    args.resize_size_h = 540//2
    args.resize_size_w = 960//2
    # 10404182556 in portrait others in landscape mode
    
    args.features_dir = 'features-{}-fe_init_mode-{}/'.format(args.arch, args.fe_init_mode)  

    if args.re_init_mode:
        if args.fe_init_mode == 0:
            args.checkpoint = 'checkpoints/IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(args.arch, 0, args.pool_mode, 0.0)
            args.pretrained = 0
        elif args.fe_init_mode == 1:
            args.checkpoint = 'checkpoints/IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(args.arch, 1, args.pool_mode, 0.0)
            args.pretrained = 1
        elif args.fe_init_mode == 2:
            args.checkpoint = 'checkpoints/IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(args.arch, 0, args.pool_mode, 1.0)
            args.pretrained = 0
        elif args.fe_init_mode == 3:
            args.checkpoint = 'checkpoints/IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(args.arch, 1, args.pool_mode, 0.1)
            args.pretrained = 1
        else:
            print('Unknown fe_init_mode!')
    else:
        args.checkpoint = None
        if args.fe_init_mode == 0 or args.fe_init_mode == 2:
            args.pretrained = 0
        else:
            args.pretrained = 1
    
    if args.ft_lr_ratio != 0.0:
        args.whole = True #
    else:
        args.whole = False
        
    args.decay_interval = int(args.epochs/10) #
    if args.decay_interval <= 0:
        args.decay_interval = 1
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    args.format_str = '{}-{}-fim={}-rim={}-g={}-lr{}-bs{}-e{}-ftlrr{}-exp{}'.format(args.model, args.arch, args.fe_init_mode, args.re_init_mode, 
                                                                       args.groups, args.lr, args.batch_size, args.epochs, args.ft_lr_ratio, args.exp_id)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    args.trained_model_file = 'checkpoints/{}'.format(args.format_str)
    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/{}'.format(args.format_str)
    logger.create_logger('logs', args.format_str, False)
    logger.log.info(args)
    run(args)
