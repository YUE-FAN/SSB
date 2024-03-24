import torch
import torch.nn.functional as F

__all__ = ['ova_loss', 'unlabeled_ova_neg_loss']

def ova_loss(args, logits_open_w, logits_open_s, label, label_s):
    """
        label_sp_neg =
        [[1, 1, 0, 1],
         [0, 1, 1, 1],
         [1, 1, 1, 0]]

    OpenMatch hard classifier mining. For each training sample, we find a hard classifier
    (previously was each training sample trains all classifiers)
    In fact, we want, for each classifier, find a hard negative sample

    How to train a OVA classifier?
    1. Consider all negatives, which will lead to imbalance problem.
    2. Always select the top K negatives for each classifier, where K = bs / C, so that the number of positives
       and negatives the classifier receives per batch (averagely speaking) are equal.
       To do so, compute a topK_mask.
    3. Always select the top 1 negatives for each classifier.
       To do so, compute a topK_mask.
    4. A classifier receives bs / C positive samples per batch, but (bs - bs / C) negative samples per batch.
       So to balance out, we need to divide the negative loss by some value so that they are of the same scale.
       But maybe this is not necessary? Because if you take the mean, they are already of the same scale.
       So maybe start with 1. and 2.

    """
    logits_open_w = logits_open_w.view(logits_open_w.size(0), 2, -1)  # [bs, 2, num_class]
    logits_open_s = logits_open_s.view(logits_open_s.size(0), 2, -1)  # [bs, 2, num_class]

    # negative loss with one-hot label
    label_s_sp = torch.zeros((logits_open_w.size(0), logits_open_w.size(2))).long().to(label.device)  # [bs, num_class]
    label_s_sp.scatter_(1, label.view(-1, 1), 1)
    label_sp_neg = 1 - label_s_sp
    loss_values_w = -F.log_softmax(logits_open_w, dim=1)[:, 0, :] * label_sp_neg  # [bs, num_class]

    label_s_sp_s = torch.zeros((logits_open_s.size(0), logits_open_s.size(2))).long().to(label_s.device)  # [bs, num_class]
    label_s_sp_s.scatter_(1, label_s.view(-1, 1), 1)
    label_sp_neg_s = 1 - label_s_sp_s
    loss_values_s = -F.log_softmax(logits_open_s, dim=1)[:, 0, :] * label_sp_neg_s  # [bs, num_class]

    if args.ova_neg_DA == 's':
        loss_values = loss_values_s
    elif args.ova_neg_DA == 'w':
        loss_values = loss_values_w
    elif args.ova_neg_DA == 'ws':
        loss_values = torch.cat((loss_values_w, loss_values_s), dim=0)
    else:
        raise NotImplementedError

    if args.ova_neg_loss == 'all':
        open_loss_neg = torch.mean(loss_values.mean(dim=0))
    elif args.ova_neg_loss == 'top1':
        loss_values = torch.max(loss_values, dim=0)[0]  # [1, C]
        open_loss_neg = torch.mean(loss_values)
    elif args.ova_neg_loss == 'topk':
        loss_values = torch.topk(loss_values, k=int(logits_open_w.size(0) / logits_open_w.size(2)), dim=0)[0]  # [bs/C, C]
        open_loss_neg = torch.mean(loss_values.mean(dim=0))
    else:
        raise NotImplementedError

    # positive loss
    if args.ova_pos_DA == 's':
        open_loss_pos = torch.mean((-F.log_softmax(logits_open_s, dim=1)[:, 1, :] * label_s_sp_s).sum(dim=1))
    elif args.ova_pos_DA == 'w':
        open_loss_pos = torch.mean((-F.log_softmax(logits_open_w, dim=1)[:, 1, :] * label_s_sp).sum(dim=1))
    elif args.ova_pos_DA == 'ws':
        open_loss_pos_s = (-F.log_softmax(logits_open_s, dim=1)[:, 1, :] * label_s_sp_s).sum(dim=1)
        open_loss_pos_w = (-F.log_softmax(logits_open_w, dim=1)[:, 1, :] * label_s_sp).sum(dim=1)
        open_loss_pos = torch.mean(torch.cat((open_loss_pos_s, open_loss_pos_w), dim=0))
    else:
        raise NotImplementedError


    return open_loss_pos + open_loss_neg


def unlabeled_ova_neg_loss(args, logits_w1, logits_w2, logits_s1, logits_s2, mask):
    logits_w2 = logits_w2.view(logits_w2.size(0), 2, -1)  # [bs, 2, num_class]
    logits_s1 = logits_s1.view(logits_s1.size(0), 2, -1)  # [bs, 2, num_class]

    if args.ova_unlabeled_neg_DA == 's':
        open_loss_neg = torch.mean((-F.log_softmax(logits_s1, dim=1)[:, 0, :] * mask).sum(dim=1))
    elif args.ova_unlabeled_neg_DA == 'w':
        open_loss_neg = torch.mean((-F.log_softmax(logits_w2, dim=1)[:, 0, :] * mask).sum(dim=1))
    elif args.ova_unlabeled_neg_DA == 'ws':
        open_loss_neg = torch.mean((-F.log_softmax(logits_w2, dim=1)[:, 0, :] * mask).sum(dim=1))
        open_loss_neg += torch.mean((-F.log_softmax(logits_s1, dim=1)[:, 0, :] * mask).sum(dim=1))
    else:
        raise NotImplementedError

    return open_loss_neg