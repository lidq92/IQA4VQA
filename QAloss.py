import torch


# norm-in-norm loss
class QALoss(torch.nn.Module):
    def __init__(self):
        super(QALoss, self).__init__()

    def forward(self, y_pred, y):
        y_pred = y_pred - torch.mean(y_pred) 
        normalization = torch.norm(y_pred, p=2) 
        y_pred = y_pred / (1e-8 + normalization) 
        y = y - torch.mean(y)
        y = y / (1e-8 + torch.norm(y, p=2))
        scale = 2 * pow(y.size(0), .5) 
        err = y_pred - y
        loss = torch.norm(err, p=1) / scale  

        return loss
