import os
import h5py
import torch
import logger
import skvideo.io
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import resize


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, features_path, index, feature_extractor='resnet50', groups=16):
        super(VQADataset, self).__init__()
        # index = index[:8] # debug
        self.features = len(index) * [None]
        self.labels = len(index) * [0]
        for i, idx in enumerate(index):
            feature = np.load(features_path + str(idx) + '_' + feature_extractor +'_last_conv.npy')
            feature = feature[::feature.shape[0]//groups]
            # print(feature.shape)
            self.features[i] = feature
            self.labels[i] = np.load(features_path + str(idx) + '_score.npy')  #

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].reshape(-1)


class WholeVQADataset(torch.utils.data.Dataset):
    def __init__(self, args, videos_path, scores, index):

        super(WholeVQADataset, self).__init__()
#         index = index[:8] # debug
        self.data = [videos_path[idx] for idx in index]
        self.scores = [scores[idx] for idx in index]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.resize_size_h = args.resize_size_h
        self.resize_size_w = args.resize_size_w
        self.groups = args.groups

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        video_data = skvideo.io.vread(self.data[idx]) 
        video_score = self.scores[idx]

        video_length = self.groups # video_data.shape[0] 
        video_channel = video_data.shape[3]
        video_height = self.resize_size_h # video_data.shape[1]
        video_width = self.resize_size_w # video_data.shape[2]

        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        for frame_idx in range(0, video_data.shape[0],video_data.shape[0]//self.groups):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = resize(frame, (self.resize_size_h, self.resize_size_w))  
            frame = self.transform(frame)
            transformed_video[frame_idx//self.groups] = frame
        
        return transformed_video, video_score

    
def get_data_loaders(args):
    Info = h5py.File(args.datainfo, 'r')  # index, ref_ids
    index = Info['index']
    index_rd = index[:, args.exp_id % index.shape[1]].astype(int)  
    # index_rd = np.random.permutation(1200) #
    ref_ids = Info['ref_ids'][0, :]  #
    trainindex = index_rd[:round(.6 * len(index_rd))]
    # valindex = index_rd[round(.6 * len(index_rd)):round(.8 * len(index_rd))]
    testindex = index_rd[round(.8 * len(index_rd)):]
    train_index, val_index, test_index = [], [], []
    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)
    datainfo = pd.read_csv("{}/KoNViD_1k_attributes.csv".format(args.dataset))
    datainfo['file_names'] = datainfo['flickr_id'].astype(str) + ".mp4"
    videos_path = [os.path.join(args.dataset, video_name) for video_name in datainfo['file_names']]
    scores = datainfo['MOS'] / (datainfo['MOS'].max() - datainfo['MOS'].min())
    if args.whole:
        logger.log.info("# of videos in the training set: %d" % len(train_index))
        train_dataset = WholeVQADataset(args, videos_path, scores, train_index)
        logger.log.info("# of videos in the validation set: %d" % len(val_index))
        val_dataset = WholeVQADataset(args, videos_path, scores, val_index)
        logger.log.info("# of videos in the test set: %d" % len(test_index))
        test_dataset = WholeVQADataset(args, videos_path, scores, test_index)
    else:
        logger.log.info("# of videos in the training set: %d" % len(train_index))
        train_dataset = VQADataset(args.features_dir, train_index, args.arch, args.groups)
        logger.log.info("# of videos in the validation set: %d" % len(val_index))
        val_dataset = VQADataset(args.features_dir, val_index, args.arch, args.groups)
        logger.log.info("# of videos in the test set: %d" % len(test_index))
        test_dataset = VQADataset(args.features_dir, test_index, args.arch, args.groups)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2*args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2*args.batch_size)

    return train_loader, val_loader, test_loader
