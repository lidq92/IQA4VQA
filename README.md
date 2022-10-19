# Best Practices for Initializing Image and Video Quality Assessment Models
This repo is largely borrowed from [LinearityIQA](https://github.com/lidq92/LinearityIQA).

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
# cross-dataset evaluation
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnet18; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnet18; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnet18; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnet18
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnet34; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnet34; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnet34; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnet34
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnet50; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnet50; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnet50; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnet50
python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 0.0 --arch resnext101_32x8d; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.0 --arch resnext101_32x8d; python test_IQAdataset.py -pretrained 0 -ft_lr_ratio 1.0 --arch resnext101_32x8d; python test_IQAdataset.py -pretrained 1 -ft_lr_ratio 0.1 --arch resnext101_32x8d
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
