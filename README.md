# Best Practices for Initializing Image and Video Quality Assessment Models
This repo is largely borrowed from [LinearityIQA](https://github.com/lidq92/LinearityIQA).

**Requirements**:
- python==3.6.9
- torch==1.8.1 (with cuda v10.2, cudnn v7.6)
- torchvision==0.9.1
- pytorch-ignite==0.4.2
- h5py==2.10.0
- matplotlib==3.1.3
- numpy==1.18.1
- pandas==0.25.3
- Pillow==6.2.1
- scikit-learn==0.24.1
- scikit-video==1.1.11
- scipy==1.5.4

## 0. Downloading and Linking the Datasets
```bash
ln -s KonIQ-10k_database_path KonIQ-10k
ln -s CLIVE_database_path CLIVE
ln -s KoNViD-1k_database_path KoNViD-1k
ln -s LIVE-VQC_database_path LIVE-VQC
```

## 1. Training and Evaluating the IQA Networks
```bash
# training and intra-dataset evaluation
python IQAmain.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnet18; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnet18; python IQAmain.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnet18; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnet18
python IQAmain.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnet34; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnet34; python IQAmain.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnet34; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnet34
python IQAmain.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnet50; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnet50; python IQAmain.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnet50; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnet50
python IQAmain.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnext101_32x8d; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnext101_32x8d; python IQAmain.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnext101_32x8d; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnext101_32x8d
python IQAmain.py -pretrained 0 -ft_lr_ratio 0.0 --arch alexnet; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.0 --arch alexnet; python IQAmain.py -pretrained 0 -ft_lr_ratio 1.0 --arch alexnet; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.1 --arch alexnet
python IQAmain.py -pretrained 0 -ft_lr_ratio 0.0 --arch vgg16; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.0 --arch vgg16; python IQAmain.py -pretrained 0 -ft_lr_ratio 1.0 --arch vgg16; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.1 --arch vgg16
python IQAmain.py -pretrained 0 -ft_lr_ratio 0.0 --arch googlenet; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.0 --arch googlenet; python IQAmain.py -pretrained 0 -ft_lr_ratio 1.0 --arch googlenet; python IQAmain.py -pretrained 1 -ft_lr_ratio 0.1 --arch googlenet
# cross-dataset evaluation
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnet18; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnet18; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnet18; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnet18
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnet34; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnet34; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnet34; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnet34
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnet50; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnet50; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnet50; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnet50
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnext101_32x8d; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnext101_32x8d; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnext101_32x8d; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnext101_32x8d
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch alexnet; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch alexnet; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch alexnet; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch alexnet
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch vgg16; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch vgg16; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch vgg16; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch vgg16
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch googlenet; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch googlenet; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch googlenet; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch googlenet
```

## 2. Feature Extraction for VQA
```bash
 for i in $(seq 0 3); do python FeatureExtractor.py --arch resnet50 -fim $i; done
```

## 3. Training and Evaluating the VQA Networks
```bash
# training and intra-dataset evaluation
for i in $(seq 0 9); do python VQAmain.py --arch resnet50 -fim 3 -rim 1 -g 16 --exp_id $i; done
for i in $(seq 0 9); do python VQAmain.py --arch resnet50 -fim 3 -rim 0 -g 16 --exp_id $i; done
for i in $(seq 0 9); do python VQAmain.py --arch resnet50 -fim 2 -rim 1 -g 16 --exp_id $i; done
for i in $(seq 0 9); do python VQAmain.py --arch resnet50 -fim 2 -rim 0 -g 16 --exp_id $i; done
for i in $(seq 0 9); do python VQAmain.py --arch resnet50 -fim 1 -rim 1 -g 16 --exp_id $i; done
for i in $(seq 0 9); do python VQAmain.py --arch resnet50 -fim 1 -rim 0 -g 16 --exp_id $i; done
for i in $(seq 0 9); do python VQAmain.py --arch resnet50 -fim 0 -rim 1 -g 16 --exp_id $i; done
for i in $(seq 0 9); do python VQAmain.py --arch resnet50 -fim 0 -rim 0 -g 16 --exp_id $i; done
# cross-dataset evaluation
for i in $(seq 0 3); do python test_VQAdataset.py --arch resnet50 -fim $i -rim 1 -g 16; python test_VQAdataset.py --arch resnet50 -fim $i -rim 0 -g 16; done
```

## 4. Analyzing the Results

```bash
cd analysis
python results_analysis.py # You need to download and rename the csv files which contain data in the TensorBoard writer.
```

## Cite
Our technical report is provided [here](IQA4VQA.pdf). If you find this useful, please kindly cite it.
```bibtex
@techreport{li2022iqa4vqa,
     title = {Initialize and Train a Unified Quality Assessment Model for Images/Videos in the Wild},
     author = {Dingquan Li and Haiqiang Wang and Wei Gao and Ge Li},
     year = {2022},
     pages = {1--10},
     institution = {Peng Cheng Laboratory},
     url = {https://github.com/lidq92/IQA4VQA}
}
```