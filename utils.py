import torch
from torch.nn.functional import log_softmax

def cross_entropy_for_onehot(pred, target):
    # Prediction should be logits instead of probs
    return torch.mean(torch.sum(-target * log_softmax(pred, dim=-1), 1))

def multihead_accuracy(output, target):
    prec1 = []
    for j in range(output.size(1)):
        acc = accuracy(output[:, j], target[:, j], topk=(1, ))
        prec1.append(acc[0])
    return torch.mean(torch.Tensor(prec1))

def accuracy(output, target, topk=(1, ), multi_head=False):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if len(target.size()) == 1:  # single-class classification
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        else:  # multi-class classification
            if multi_head:
                res = [multihead_accuracy(output, target)]
            else:
                assert len(topk) == 1
                pred = torch.sigmoid(output).ge(0.5).float()
                correct = torch.count_nonzero(pred == target).float()
                correct *= 100.0 / (batch_size * target.size(1))
                res = [correct]
    return res
