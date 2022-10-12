import torch


class QALoss(torch.nn.Module):
    def __init__(self):
        super(QALoss, self).__init__()

    def forward(self, y_pred, y):
        # print(y_pred.shape, y.shape)
        # loss = torch.nn.functional.mse_loss(y_pred, y)
        # loss = torch.nn.functional.l1_loss(y_pred, y)
        # assert y_pred.size(0) > 1  #
        # loss = (1 - torch.cosine_similarity(y_pred.view(1, -1) - torch.mean(y_pred), y.view(1, -1) - torch.mean(y))[0]) / 2
        y_pred = y_pred - torch.mean(y_pred) 
        normalization = torch.norm(y_pred, p=2) 
        y_pred = y_pred / (1e-8 + normalization) 
        y = y - torch.mean(y)
        y = y / (1e-8 + torch.norm(y, p=2))
        scale = 2 * pow(y.size(0), .5) 
        err = y_pred - y
        loss = torch.norm(err, p=1) / scale  

        return loss

