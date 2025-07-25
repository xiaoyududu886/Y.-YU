import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable

# CELoss with class weights
def cross_entropy(y_true, y_pred, class_weights=None, size_average=False):
    loss = -torch.sum(class_weights * y_true * torch.log(y_pred))
    return loss

# Adversarial Loss
def adversarial_loss(y_true, y_pred, dis_output):
    # n, c, h, w, s = input.size()
    loss = 0
    for p in range(y_true.shape[0]):
        for i in range(y_true.shape[1]):
            loss += y_true[p, i] * torch.log(dis_output[p]) + \
                    (1 - y_true[p, i]) * torch.log(1 - dis_output[p])
    return loss

class SoftLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


class CustomSoftLoss(nn.Module):
    def __init__(self, n_classes, class_ids):
        super(CustomSoftLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda()

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


if __name__ == '__main__':
    from torch.autograd import Variable
    depth = 3
    batch_size = 2
    encoder = One_Hot(depth=depth).forward
    y = Variable(torch.LongTensor(batch_size, 1, 1, 2 ,2).random_() % depth).cuda()  # 4 classes,1x3x3 img
    y_onehot = encoder(y)
    x = Variable(torch.randn(y_onehot.size()).float()).cuda()
    dicemetric = SoftLoss(n_classes=depth)
    dicemetric(x,y)