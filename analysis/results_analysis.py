import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
plt.rcParams['pdf.fonttype'] = 42



# Fig. 2
# IQA test curve
fontsize = 24
# plt.figure(figsize=(14,10))            
archs = ['resnext101_32x8d']
inits = ['random-fixed','ImageNet-fixed','ImageNet']
inits.reverse()
Inits = ['Frozen $\mathbf{w}_\mathrm{random}$','Frozen $\mathbf{w}_\mathrm{ImageNet}$','Learnable $\mathbf{w}_\mathrm{ImageNet}$']
Inits.reverse()
for k, arch in enumerate(archs):
    for i, init in enumerate(inits):
        data = pd.read_csv('IQA-{}-{}.csv'.format(arch, init))
        epoch = np.asarray(data.iloc[:,1])
        performance = np.asarray(data.iloc[:,2])
        plt.plot(epoch, performance, label=Inits[i]) 
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('SROCC', fontsize=fontsize)
    plt.tick_params(axis="x", labelsize=fontsize) 
    plt.tick_params(axis="y", labelsize=fontsize) 
    plt.legend(loc='best', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("IQA_{}_test_curve.pdf".format(arch), bbox_inches="tight")
    plt.clf()
# scatter plots
fontsize = 24
archs = ['resnext101_32x8d']
Archs = ['ResNeXt101']
pool = 'mean'
pretrained = [0, 1, 0, 1]
ftlrr = [0.0, 0.0, 1.0, 0.1]
inits = ['random-fixed','ImageNet-fixed','random','ImageNet']
Inits = ['Frozen $\mathbf{w}_\mathrm{random}$','Frozen $\mathbf{w}_\mathrm{ImageNet}$','Learnable $\mathbf{w}_\mathrm{random}$','Learnable $\mathbf{w}_\mathrm{ImageNet}$']
for k, arch in enumerate(archs):
    SROCC = np.zeros(4)
    for i in range(4):
        format_str = 'IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(arch, pretrained[i], pool, ftlrr[i])
        try:
            data = np.load('../results/{}.npy'.format(format_str), allow_pickle=True)
        except:
            data = np.load('../results/{}.npy'.format(format_str).replace('pmode=mean-', ''), allow_pickle=True)
        y = data.item()['sq']
        x = data.item()['pq'].reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        xl = np.linspace(x.min(), x.max(),1000).reshape(-1,1)
        plt.plot(x, y,'o', markersize=2)
        plt.plot(xl, reg.predict(xl))#     
        plt.xlabel('Predicted Quality', fontsize=fontsize)
        plt.ylabel('MOS', fontsize=fontsize)
        plt.title(Inits[i], fontsize=fontsize)
        plt.tick_params(axis="x", labelsize=fontsize) 
        plt.tick_params(axis="y", labelsize=fontsize) 
        plt.tight_layout()
        plt.savefig("IQA_{}_fitted_curve_{}.pdf".format(Archs[k], inits[i]), bbox_inches="tight")
        plt.clf()


# Table I intra-iqa
archs = ['resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'alexnet', 'vgg16', 'googlenet']
Archs = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNeXt101', 'AlexNet', 'VGG16', 'GoogleNet']
pool = 'mean'
pretrained = [0, 1, 0, 1]
ftlrr = [0.0, 0.0, 1.0, 0.1]
for k, arch in enumerate(archs):
    SROCC = np.zeros(4)
    for i in range(4):
        format_str = 'IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(arch, pretrained[i], pool, ftlrr[i])
        data = np.load('../results/{}.npy'.format(format_str), allow_pickle=True)
        SROCC[i] = data.item()['SROCC']
    print('{} & ${:.3f}$ & ${:.3f}$ & ${:+.3f}$ & ${:.3f}$ & ${:.3f}$ & ${:+.3f}$ \\\\'.format(Archs[k], SROCC[0], SROCC[1], SROCC[1]-SROCC[0], SROCC[2], SROCC[3], SROCC[3]-SROCC[2]))
# Table II cross-iqa   
for k, arch in enumerate(archs):
    SROCC = np.zeros(4)
    for i in range(4):
        format_str = 'dataset=CLIVE-tested_on_IQA-{}-pretrained={}-pmode={}-resize=True-lr=0.0001-bs=8-e=30-ftlrr={}-exp0'.format(arch, pretrained[i], pool, ftlrr[i])
        data = np.load('../results/{}.npy'.format(format_str), allow_pickle=True)
        SROCC[i] = data.item()['SROCC']
    print('{} & ${:.3f}$ & ${:.3f}$ & ${:+.3f}$ & ${:.3f}$ & ${:.3f}$ & ${:+.3f}$ \\\\'.format(Archs[k], SROCC[0], SROCC[1], SROCC[1]-SROCC[0], SROCC[2], SROCC[3], SROCC[3]-SROCC[2]))

# Table III intra-vqa   
archs = ['resnet50']
fims = [0, 1, 3]
rims = [0, 1]
groups = 16
for arch in archs:
    SROCC = np.zeros((10, 2, 3))
    for rim in rims:
        for f, fim in enumerate(fims):
            for exp_id in range(10):
                format_str = 'VQA-{}-fim={}-rim={}-g={}-lr0.0001-bs8-e30-exp{}'.format(arch, fim, rim, groups, exp_id)
                data = np.load('../results/{}.npy'.format(format_str), allow_pickle=True)
                SROCC[exp_id][rim][f] = data.item()['SROCC']
    meanSROCC = SROCC.mean(axis=0)
    stdSROCC = SROCC.std(axis=0)
    for rim in rims:
        print(' & ${:.3f}$ & ${:.3f}$ & ${:.3f}$ & ${:+.3f}$ & ${:+.3f}$ \\\\'.format(meanSROCC[rim][0], meanSROCC[rim][1], meanSROCC[rim][2], meanSROCC[rim][1]-meanSROCC[rim][0], meanSROCC[rim][2]-meanSROCC[rim][1]))
    print(' & ${:+.3f}$ & ${:+.3f}$ & ${:+.3f}$ & ${:+.3f}$ & ${:+.3f}$ \\\\'.format(meanSROCC[1][0]-meanSROCC[0][0], meanSROCC[1][1]-meanSROCC[0][1], meanSROCC[1][2]-meanSROCC[0][2], meanSROCC[1][1]-meanSROCC[0][0], meanSROCC[1][2]-meanSROCC[0][1]))
# Table cross-vqa   
for arch in archs:
    SROCC = np.zeros((2, 3))
    for rim in rims:
        for f, fim in enumerate(fims):
            format_str = 'dataset=LIVE-VQC-tested_on_VQA-{}-fim={}-rim={}-g={}-lr0.0001-bs8-e30-exp0'.format(arch, fim, rim, groups)
            data = np.load('../results/{}.npy'.format(format_str), allow_pickle=True)
            SROCC[rim][f] = data.item()['SROCC']
    for rim in rims:
        print(' & ${:.3f}$ & ${:.3f}$ & ${:.3f}$ & ${:+.3f}$ & ${:+.3f}$ \\\\'.format(SROCC[rim][0], SROCC[rim][1], SROCC[rim][2], SROCC[rim][1]-SROCC[rim][0], SROCC[rim][2]-SROCC[rim][1]))
    print(' & ${:+.3f}$ & ${:+.3f}$ & ${:+.3f}$ & ${:+.3f}$ & ${:+.3f}$ \\\\'.format(SROCC[1][0]-SROCC[0][0], SROCC[1][1]-SROCC[0][1], SROCC[1][2]-SROCC[0][2], SROCC[1][1]-SROCC[0][0], SROCC[1][2]-SROCC[0][1]))

# Fig.3 intra-vqa split-SROCC
plt.figure(figsize=(10,10))
fontsize = 28
linewidth = 3.0
archs = ['resnet50']
fims = [1, 1, 3]
rims = [0, 1, 1]
groups = 16
for arch in archs:
    SROCC = np.zeros((10, 3))
    for i in range(3):
        rim = rims[i]
        fim = fims[i]
        for exp_id in range(10):
            format_str = 'VQA-{}-fim={}-rim={}-g={}-lr0.0001-bs8-e30-exp{}'.format(arch, fim, rim, groups, exp_id)
            data = np.load('../results/{}.npy'.format(format_str), allow_pickle=True)
            SROCC[exp_id][i] = data.item()['SROCC']
    meanSROCC = SROCC.mean(axis=0)
    stdSROCC = SROCC.std(axis=0)
    l2, = plt.plot(range(10), SROCC[:,2], 'D-', zorder=1, linewidth=linewidth, label='$\mathbf{w}_\mathrm{IQA},\mathbf{v}_\mathrm{IQA}$')
    l1, = plt.plot(range(10), SROCC[:,1], 'o-', zorder=1, linewidth=linewidth, label='$\mathbf{w}_\mathrm{ImageNet},\mathbf{v}_\mathrm{IQA}$')
    l0, = plt.plot(range(10), SROCC[:,0], '*-', zorder=1, linewidth=linewidth, label='$\mathbf{w}_\mathrm{ImageNet},\mathbf{v}_\mathrm{random}$')
    plt.xlabel('Split ID', fontsize=fontsize)
    plt.ylabel('SROCC', fontsize=fontsize)
    plt.tick_params(axis="x", labelsize=fontsize) 
    plt.tick_params(axis="y", labelsize=fontsize) 
    plt.xlim((0, 9))
    plt.xticks(range(10))
    plt.ylim((0.7,0.9))
    plt.legend(loc='best', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("split-SROCC.pdf", bbox_inches="tight")
    plt.clf()
