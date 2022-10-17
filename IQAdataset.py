import os
import h5py
import logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_tensor, normalize


def default_loader(path):
    return Image.open(path).convert('RGB')  


class IQADataset(Dataset):
    def __init__(self, args, status='train', loader=default_loader):
        Info = h5py.File(args.data_info, 'r')
        index = Info['index']
        index_rd = index[:, args.exp_id % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]
        if status == 'train':
            index = index[0:int(args.train_ratio * len(index))]
        elif status == 'val':
            index = index[int(args.train_ratio * len(index)):int(args.train_and_val_ratio * len(index))]
        elif status == 'test':
                index = index[int(args.train_and_val_ratio * len(index)):len(index)]
        self.index = []
        for i in range(len(ref_ids)):
            if ref_ids[i] in index:
                self.index.append(i)
        if args.debug:
            self.index = self.index[:8] # debug
        logger.log.info("# {} images: {}".format(status, len(self.index)))

        self.label = Info['subjective_scores'][0, self.index]
        self.im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        self.ims = len(self.im_names) * [None]
        self.noresize = args.noresize
        for i, im_name in enumerate(self.im_names):
            im = loader(os.path.join(args.dataset, im_name))
            if not args.noresize:  
                im = resize(im, (args.resize_size_h, args.resize_size_w))  
                im = to_tensor(im)
                im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            self.ims[i] = im

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = self.ims[idx]
        if self.noresize:
            im = to_tensor(im)
            im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        label = self.label[idx].reshape(-1)
        return im, label


def get_data_loaders(args):
    train_dataset = IQADataset(args, 'train')
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=True) 
    val_dataset = IQADataset(args, 'val')
    test_dataset = IQADataset(args, 'test')
    val_loader = DataLoader(val_dataset, batch_size=2*args.batch_size)    
    test_loader = DataLoader(test_dataset, batch_size=2*args.batch_size)
    
    return train_loader, val_loader, test_loader
