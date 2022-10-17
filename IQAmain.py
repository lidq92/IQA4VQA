import os
import torch
import random
import logger
import datetime
import numpy as np
from QAloss import QALoss
from IQAmodel import QAModel
from ignite.engine import Events
from argparse import ArgumentParser
from IQAdataset import get_data_loaders
from QAperformance import QAPerformance
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from modified_ignite_engine import create_supervised_trainer, create_supervised_evaluator


metrics_printed = ['SROCC', 'KROCC', 'PLCC']
def writer_add_scalar(writer, status, dataset, scalars, iter):
    for metric_print in metrics_printed:
        writer.add_scalar('{}/{}/{}'.format(status, dataset, metric_print), scalars[metric_print], iter)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QAModel(arch=args.arch, pretrained=args.pretrained, pool_mode=args.pool_mode).to(device)  #
    logger.log.info(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.log.info('total_params: {}'.format(total_params))

    if args.ft_lr_ratio == 0.0:
        for param in model.features.parameters():
            param.requires_grad = False
        optimizer = Adam([{'params': model.regression.parameters()}, 
                        {'params': model.fp.parameters()},
                        {'params': model.dr.parameters()}],
                        lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam([{'params': model.regression.parameters()}, 
                        {'params': model.fp.parameters()},
                        {'params': model.dr.parameters()},
                        {'params': model.features.parameters(), 'lr': args.lr * args.ft_lr_ratio}],
                        lr=args.lr, weight_decay=args.weight_decay)
    
    train_loader, val_loader, test_loader = get_data_loaders(args)

    evaluator = create_supervised_evaluator(model, metrics={'IQA_performance': QAPerformance()}, device=device)
    if args.evaluate:
        checkpoint = torch.load(args.trained_model_file)
        model.load_state_dict(checkpoint['model'])

        evaluator.run(test_loader)
        performance = evaluator.state.metrics
        for metric_print in metrics_printed:
            logger.log.info('{}, {}: {:.5f}'.format(args.dataset, metric_print, performance[metric_print].item()))
        for metric_print in metrics_printed:
            logger.log.info('{:.5f}'.format(performance[metric_print].item()))
        np.save(args.save_result_file, performance)
        return

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    loss_func = QALoss()
    trainer = create_supervised_trainer(model, optimizer, loss_func, device=device) 
    current_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    writer = SummaryWriter(log_dir='runs/{}-{}'.format(args.format_str, current_time))
    global best_val_criterion, best_epoch
    best_val_criterion, best_epoch = -100, -1
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def iter_event_function(engine):
        writer.add_scalar("train/loss", engine.state.output, engine.state.iteration)
        # if args.debug:
        #     logger.log.info(engine.state.output)

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_event_function(engine):
        evaluator.run(train_loader) 
        performance = evaluator.state.metrics
        writer_add_scalar(writer, 'train', args.dataset, performance, engine.state.epoch)
        logger.log.info('Train {}: {:.5f} @epoch: {}'.format(args.val_criterion, performance[args.val_criterion], engine.state.epoch))

        evaluator.run(val_loader)
        performance = evaluator.state.metrics
        writer_add_scalar(writer, 'val', args.dataset, performance, engine.state.epoch)
        val_criterion = performance[args.val_criterion]
        
        evaluator.run(test_loader)
        performance = evaluator.state.metrics
        writer_add_scalar(writer, 'test', args.dataset, performance, engine.state.epoch)

        global best_val_criterion, best_epoch
        if val_criterion > best_val_criterion: 
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, args.trained_model_file)
            best_val_criterion = val_criterion
            best_epoch = engine.state.epoch
            logger.log.info('Save current best model @best_val_criterion ({}): {:.5f} @epoch: {}'.format(args.val_criterion, best_val_criterion, best_epoch))
        else:
            logger.log.info('Model is not updated @val_criterion ({}): {:.5f} @epoch: {}'.format(args.val_criterion, val_criterion, engine.state.epoch))

        scheduler.step()

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        writer.close ()  # close the Tensorboard writer
        logger.log.info('best epoch: {}'.format(best_epoch))
        checkpoint = torch.load(args.trained_model_file)
        model.load_state_dict(checkpoint['model'])
        
        evaluator.run(test_loader)
        performance = evaluator.state.metrics
        for metric_print in metrics_printed:
            logger.log.info('{}, {}: {:.5f}'.format(args.dataset, metric_print, performance[metric_print].item()))
        np.save(args.save_result_file, performance)
        logger.destroy_logger()

    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    parser = ArgumentParser(description='')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument("--exp_id", type=int, default=0,
                        help='the exp split idx (default: 0)')
    parser.add_argument("--model", type=str, default='IQA')
    parser.add_argument('--arch', default='resnext101_32x8d', type=str,
                        help='arch name (default: resnext101_32x8d)')
    parser.add_argument('--pool_mode', default='mean', type=str,
                        help='pool mode (default: mean)')
    parser.add_argument('-pretrained', '--pretrained', type=int, default=1,
                        help='feature extractor (fe) network init mode， 0 for default random, 1 for ImageNet-pretrained (default: 1)')
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
    parser.add_argument('-eval', '--evaluate', action='store_true',
                        help='Evaluate only?')
    parser.add_argument('-random', '--randomness', action='store_true',
                        help='Allow randomness during training?') 
    parser.add_argument('-debug', '--debug', action='store_true',
                        help='Debug?') 
    args = parser.parse_args()
    args.val_criterion = 'SROCC'
    if args.lr_decay == 1 or args.epochs < 3:  # no lr decay
        args.lr_decay_step = args.epochs
    else:  # 
        args.lr_decay_step = int(args.epochs/(1+np.log(args.overall_lr_decay)/np.log(args.lr_decay)))
    if args.ft_lr_ratio != 0: #
        if args.pretrained:
            args.ft_lr_ratio = 0.1
        else:
            args.ft_lr_ratio = 1.0

    args.dataset = 'KonIQ-10k' # ln -s database_path xxx
    args.data_info = './data/KonIQ-10kinfo.mat'
    args.train_ratio = 7058/10073
    args.train_and_val_ratio = 8058/10073
    if args.noresize:
        args.resize_size_h = 768
        args.resize_size_w = 1024

    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    args.format_str = '{}-{}-pretrained={}-pmode={}-resize={}-lr={}-bs={}-e={}-ftlrr={}-exp{}'\
                      .format(args.model, args.arch, args.pretrained, args.pool_mode, not args.noresize, 
                              args.lr, args.batch_size, args.epochs, args.ft_lr_ratio, args.exp_id)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    args.trained_model_file = 'checkpoints/' + args.format_str
    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/' + args.format_str
    logger.create_logger('logs', args.format_str, False)
    logger.log.info(args)
    run(args)
