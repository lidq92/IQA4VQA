import os
import h5py
import torch
import random
import logger
import skvideo.io
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from scipy import stats
from IQAmodel import QAModel
from torchvision import transforms
from argparse import ArgumentParser


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, videos_path, scores):

        super(VQADataset, self).__init__()
        self.data = videos_path 
        self.scores = scores
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        video_data = skvideo.io.vread(self.data[idx]) 
        video_score = self.scores[idx]

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]

        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            transformed_video[frame_idx] = frame
            
        sample = {'video': transformed_video,
                  'score': video_score}
        
        return sample


def get_features(video_data, extractor, frame_batch_size=64):
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    features = torch.Tensor().to(device)
    qs = torch.Tensor().to(device)
    with torch.no_grad():
        while frame_end <= video_length:
            batch = video_data[frame_start:frame_end].to(device)
            feature, q = extractor(batch)
            features = torch.cat((features, feature), 0)
            qs = torch.cat((qs, q), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

    return features, qs.mean()


if __name__ == "__main__":
    parser = ArgumentParser(description='Feature Extractor')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='')
    parser.add_argument('--pool_mode', default='mean', type=str,
                        help='pool mode (default: mean, ..., mean+std, std)')
    parser.add_argument('-fim', '--fe_init_mode', type=int, default=3,
                        help='fe network init mode (default: 0 for default random, 1 for ImageNet-pretrained, 2 for iqa-pretrained, 3 for iqa-finetuned)')
    parser.add_argument('--dataset', default='KoNViD-1k', type=str,
                        help='')
    args = parser.parse_args()
    
    if args.fe_init_mode == 0:
        args.checkpoint = 'checkpoints/IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(args.arch, 0, args.pool_mode, 0.0)
    elif args.fe_init_mode == 1:
        args.checkpoint = 'checkpoints/IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(args.arch, 1, args.pool_mode, 0.0)
    elif args.fe_init_mode == 2:
        args.checkpoint = 'checkpoints/IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(args.arch, 0, args.pool_mode, 1.0)
    elif args.fe_init_mode == 3:
        args.checkpoint = 'checkpoints/IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(args.arch, 1, args.pool_mode, 0.1)
    else:
        print('Unknown fe_init_mode!')

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.dataset == 'KoNViD-1k':
        logger.create_logger('logs', 'fe-{}-fe_init_mode-{}-pmode={}'.format(args.arch, args.fe_init_mode, args.pool_mode), False)
        videos_dir = 'KoNViD-1k/' # ln -s database_path xxx; for name in `ls *.mp4`;do mv $name ${name%%_*}.mp4;done 
        datainfo = pd.read_csv("KoNViD-1k/KoNViD_1k_attributes.csv")
        datainfo['file_names'] = datainfo['flickr_id'].astype(str) + ".mp4"
        videos_path = [os.path.join(videos_dir, video_name) for video_name in datainfo['file_names']]
        scores = datainfo['MOS'] / (datainfo['MOS'].max() - datainfo['MOS'].min())
        dataset = VQADataset(videos_path, scores)
        features_dir = 'features-{}-fe_init_mode-{}/'.format(args.arch, args.fe_init_mode)  
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
        frame_batch_size = 64
    elif args.dataset == 'LIVE-VQC':
        logger.create_logger('logs', '{}-fe-{}-fe_init_mode-{}-pmode={}'.format(args.dataset, args.arch, args.fe_init_mode, args.pool_mode), False)
        videos_dir = 'LIVE-VQC/' # ln -s database_path xxx
        datainfo = 'data/LIVE-VQCinfo.mat'
        Info = h5py.File(datainfo, 'r')
        video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
        scores = Info['scores'][0, :] / (Info['scores'][0, :].max() - Info['scores'][0, :].min())
        videos_path = [os.path.join(videos_dir, video_name) for video_name in video_names]
        dataset = VQADataset(videos_path, scores)
        features_dir = 'LIVEVQC-features-{}-fe_init_mode-{}/'.format(args.arch, args.fe_init_mode)  
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
        frame_batch_size = 32
    else:
        'Unknown dataset!'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = QAModel(arch=args.arch, returnFeature=True, pool_mode=args.pool_mode).to(device)  #
    try:
        checkpoint = torch.load(args.checkpoint)
    except:
        checkpoint = torch.load(args.checkpoint.replace('pmode=mean-', ''))
    extractor.load_state_dict(checkpoint['model'])
    extractor.eval()
    
    sq = dataset.scores
    pq = len(sq) * [0]
    for i in range(len(dataset)): 
        current_data = dataset[i]
        logger.log.info('Extract features of Video {}'.format(i))
        features, q = get_features(current_data['video'], extractor, frame_batch_size)
        pq[i] = q.to('cpu').numpy()
        np.save(features_dir + str(i) + '_' + args.arch +'_last_conv', features.to('cpu').numpy())
        np.save(features_dir + str(i) + '_score', current_data['score'])
        np.save(features_dir + str(i) + '_iqa_pred', q.to('cpu').numpy())
#     for i in range(len(dataset)): 
#         logger.log.info('Extract features of Video {}'.format(i))
#         pq[i] = np.load(features_dir + str(i) + '_iqa_pred.npy')
        
    sq = np.reshape(sq, (-1,))
    pq = np.reshape(np.asarray(pq), (-1, ))
    SROCC = stats.spearmanr(sq, pq)[0]
    KROCC = stats.stats.kendalltau(sq, pq)[0]
    # TODO: nonlinear mapping
    PLCC = stats.pearsonr(sq, pq)[0]
    logger.log.info('SROCC={}, KROCC={}, PLCC={}'.format(SROCC, KROCC, PLCC))
    logger.destroy_logger()
