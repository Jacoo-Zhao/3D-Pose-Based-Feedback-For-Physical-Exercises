import numpy as np
import torch
import torch.nn as nn
import random
from softdtw import SoftDTW
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm, colors
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def generate_random_one_hot(gt_label, batch_size=32, num_class=12):
    """
    gt_label: ground truth label  type=torch.tensor 
    batch_size: batch size   type=int
    num_class: number of classes  type=int
    """
    # generate a random label
    random_label = torch.tensor([random.randint(0,num_class-1)])
    # check if random label is the same as the gt label
    while random_label.item() == gt_label[0].item() :  
             random_label = torch.tensor([random.randint(0,num_class-1)])

    random_one_hot = torch.nn.functional.one_hot(random_label, num_classes=num_class)
    for i in range(batch_size-1):
        random_label = torch.tensor([random.randint(0,num_class-1)])
        while random_label.item() == gt_label[i+1].item() :  
            random_label = torch.tensor([random.randint(0,num_class-1)])  
        label2one_hot = torch.nn.functional.one_hot(random_label, num_classes=num_class)
        random_one_hot=torch.cat((label2one_hot,random_one_hot),0)
    return random_one_hot

def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_labels(raw_labels, level=0):
    if level == 0:
        mapping = {'SQUAT': 0, 'Lunges': 2, 'Plank': 4}
        labels = np.zeros(len(raw_labels))
        for i, el in enumerate(raw_labels):
            if el[2] == 1:
                labels[i] = mapping[el[0]]
            else:
                labels[i] = mapping[el[0]] + 1
        return torch.from_numpy(labels).long()
    elif level == 1:
        mapping = {'SQUAT': 0, 'Lunges': 6, 'Plank': 9}
        map_label = {'SQUAT': [1, 2, 3, 4, 5, 10], 'Lunges': [1, 4, 6], 'Plank': [1, 7, 8]}
        labels = np.zeros(len(raw_labels))
        for i, el in enumerate(raw_labels):
            labels[i] = mapping[el[0]] + np.where(np.array(map_label[el[0]]) == el[2])[0].item()
        return torch.from_numpy(labels).long()

def dtw_loss(originals, deltas, targets, criterion, attentions=None, is_cuda=False, test=False):
    loss = 0
    preds = []
    dtw_loss_corr = []
    dtw_loss_org = []
    for i, o in enumerate(originals):

        length = o.shape[1]
        org = torch.from_numpy(o).T.unsqueeze(0)
        targ = torch.from_numpy(targets[i]).T.unsqueeze(0)

        if length > deltas[i].shape[1]:
            m = torch.nn.ZeroPad2d((0, length - deltas[i].shape[1], 0, 0))
            # delt = dct.idct_2d(m(deltas[i]).T.unsqueeze(0))
            delt = idct_2d(m(deltas[i]).T.unsqueeze(0)).cuda()
        else:
            # delt = dct.idct_2d(deltas[i, :, :length].T.unsqueeze(0))
            delt = idct_2d(deltas[i, :, :length].T.unsqueeze(0)).cuda()

        if attentions is not None:
            delt = torch.mul(delt, attentions[i].T.unsqueeze(0))

        out = org.cuda() + delt.cuda()

        if is_cuda:
            out = out.cuda()
            targ = targ.cuda()

        crit = criterion(out, targ) - 1 / 2 * (criterion(out, out) + criterion(targ, targ))
        crit_org =  criterion(org.cuda(), targ) - 1 / 2 * (criterion(org.cuda(), org.cuda()) + criterion(targ, targ))
        mse = torch.nn.MSELoss()
        smoothness_loss = mse(out[:,1:], out[:,:-1])

        dtw_loss_corr.append(crit.item())
        dtw_loss_org.append(crit_org.item())
        loss += crit + 1e-3 * smoothness_loss      # dtw_loss + smoothness
        # loss += crit  # without smoothness

        if test:
            preds.append(out[0].detach().cpu().numpy().T)

    if test:
        return loss, preds, dtw_loss_corr, dtw_loss_org
    else:
        return loss

def train_corr_class(train_loader, model, optimizer, beta, fact=None, is_cuda=True, level=0):
    tr_l = AccumLoss()

    criterion_corr = SoftDTW(use_cuda=is_cuda, gamma=0.01)

    if level == 0:
        criterion_class = nn.NLLLoss(weight=torch.tensor([1, 0.3, 1, 0.5, 1, 1]))
    else:
        criterion_class = nn.NLLLoss()

    model.train()

    correct = 0
    total = 0

    for i, (batch_id, inputs) in enumerate(train_loader):

        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        # b = inputs.is_cuda
        # For clssifier
        labels = get_labels([train_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        batch_size = inputs.shape[0]

        # For corrector
        targets = [train_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [train_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]

        deltas, att, outputs = model(inputs)

        # For clssifier
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        # c = predicted.is_cuda
        # d = labels.is_cuda
        correct += (predicted == labels).sum().item()

        # calculate loss for classifier
        loss_class = criterion_class(outputs, labels)

        # calculate loss and backward
        if fact is None:
            dtw = dtw_loss(originals, deltas, targets, criterion_corr, is_cuda=is_cuda)
            loss_corr = dtw / batch_size
        else:
            dtw = dtw_loss(originals, deltas, targets, criterion_corr, attentions=att, is_cuda=is_cuda)
            l1 = fact * torch.sum(torch.abs(att))
            loss_corr = (dtw + l1) / batch_size

        # Calculate the total loss: loss_class + beta * loss_corr (beta =1 by default)
        loss = loss_class + beta * loss_corr
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

    return tr_l.avg, 100 * correct / total, loss_corr

def train_corr_class_v4(train_loader, model, curriculum_learning_rate, optimizer, beta, fact=None, is_cuda=True, level=0):
    x = random.random()
    tr_l = AccumLoss()

    criterion_corr = SoftDTW(use_cuda=is_cuda, gamma=0.01)

    if level == 0:
        criterion_class = nn.NLLLoss(weight=torch.tensor([1, 0.3, 1, 0.5, 1, 1]))
    else:
        criterion_class = nn.NLLLoss()

    model.train()

    correct = 0
    total = 0

    for i, (batch_id, inputs) in enumerate(train_loader):

        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        # b = inputs.is_cuda
        # For clssifier
        labels = get_labels([train_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        batch_size = inputs.shape[0]

        # For corrector
        targets = [train_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [train_loader.dataset.inputs_raw[int(i)] for i in batch_id]

        if x < curriculum_learning_rate:
            Use_label = True
        else:
            Use_label = False

        deltas, att, outputs = model(inputs, labels, Use_label)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        # calculate loss for classifier
        loss_class = criterion_class(outputs, labels)

        # calculate loss and backward
        if fact is None:
            dtw = dtw_loss(originals, deltas, targets, criterion_corr, is_cuda=is_cuda)
            loss_corr = dtw / batch_size
        else:
            dtw = dtw_loss(originals, deltas, targets, criterion_corr, attentions=att, is_cuda=is_cuda)
            l1 = fact * torch.sum(torch.abs(att))
            loss_corr = (dtw + l1) / batch_size

        # Calculate the total loss: loss_class + beta * loss_corr (beta =1 by default)
        loss = loss_class + beta * loss_corr
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

    return tr_l.avg, 100 * correct / total, loss_corr

def train_corr(train_loader, model, optimizer, fact=None, is_cuda=False):
    tr_l = AccumLoss()

    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.train()
    for i, (batch_id, inputs) in enumerate(train_loader):
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()
        targets = [train_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [train_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]
        # import pdb; pdb.set_trace()
        deltas, att = model(inputs)

        # calculate loss and backward
        if fact is None:
            dtw = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda)
            loss = dtw / batch_size
        else:
            dtw = dtw_loss(originals, deltas, targets, criterion, attentions=att, is_cuda=is_cuda)
            l1 = fact * torch.sum(torch.abs(att))
            loss = (dtw + l1) / batch_size
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

    return tr_l.avg

def evaluate_corr(val_loader, model, fact=None, is_cuda=False):
    val_l = AccumLoss()

    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.eval()
    for i, (batch_id, inputs) in enumerate(val_loader):

        if is_cuda:
            inputs = inputs.cuda().float()
        else:
            inputs = inputs.float()
        targets = [val_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [val_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]

        deltas, att = model(inputs)

        # calculate loss and backward
        if fact is None:
            dtw = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda)
            loss = dtw
        else:
            dtw = dtw_loss(originals, deltas, targets, criterion, attentions=att, is_cuda=is_cuda)
            l1 = fact * torch.sum(torch.abs(att))
            loss = (dtw + l1)

        # update the training loss
        val_l.update(loss.cpu().data.numpy(), batch_size)

    return val_l.avg

def test_corr(test_loader, model, fact=None, is_cuda=False):
    test_l = AccumLoss()
    preds = {'in': [], 'out': [], 'targ': [], 'att': []}
    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.eval()
    for i, (batch_id, inputs) in enumerate(test_loader):

        if is_cuda:
            inputs = inputs.cuda().float()
        else:
            inputs = inputs.float()
        targets = [test_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [test_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]

        deltas, att = model(inputs)

        # calculate loss and backward
        if fact is None:
            dtw, out, _, _ = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda, test=True)
            loss = dtw
        else:
            dtw, out, _, _ = dtw_loss(originals, deltas, targets, criterion, attentions=att, is_cuda=is_cuda, test=True)
            l1 = fact * torch.sum(torch.abs(att))
            loss = (dtw + l1)

        preds['in'] = preds['in'] + originals
        preds['out'] = preds['out'] + out
        preds['targ'] = preds['targ'] + targets
        preds['att'] = preds['att'] + [att[j].detach().cpu().numpy() for j in range(att.shape[0])]
        test_l.update(loss.cpu().data.numpy(), batch_size)

    return test_l.avg, preds

def test_corr_v1(test_loader, model, fact=None, is_cuda=False):
    test_l = AccumLoss()
    preds = {'in': [], 'out': [], 'targ': [], 'att': []}
    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.eval()
    for i, (batch_id, inputs) in enumerate(test_loader):

        if is_cuda:
            inputs = inputs.cuda().float()
        else:
            inputs = inputs.float()
        targets = [test_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [test_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]

        deltas, att, y = model(inputs)

        # calculate loss and backward
        if fact is None:
            dtw, out, _, _ = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda, test=True)
            loss = dtw
        else:
            dtw, out, _, _ = dtw_loss(originals, deltas, targets, criterion, attentions=att, is_cuda=is_cuda, test=True)
            l1 = fact * torch.sum(torch.abs(att))
            loss = (dtw + l1)

        preds['in'] = preds['in'] + originals
        preds['out'] = preds['out'] + out
        preds['targ'] = preds['targ'] + targets
        preds['att'] = preds['att'] + [att[j].detach().cpu().numpy() for j in range(att.shape[0])]
        test_l.update(loss.cpu().data.numpy(), batch_size)

    return test_l.avg, preds, 0

def test_corr_v4(test_loader, model, fact=None, is_cuda=False, Use_label=False):
    test_l = AccumLoss()
    preds = {'in': [], 'out': [], 'targ': [], 'att': []}
    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.eval()
    
    for i, (batch_id, inputs) in enumerate(test_loader):
        labels = get_labels([test_loader.dataset.inputs_label[int(i)] for i in batch_id], level=1)

        if is_cuda:
            inputs = inputs.cuda().float()
        else:
            inputs = inputs.float()

        targets = [test_loader.dataset.targets[int(i)] for i in batch_id]

        """ Dataset Fetching Method """
        # # pdb.set_trace()
        # with open('Data/DTW_Method.pickle', "rb") as f:
        #     data = pickle.load(f)
        # targets = data['targets']
        # targets = targets[0:31]+targets[56:88]+targets[31:56]
        # print('Load paired training data.')

        originals = [test_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]

        
        deltas, att, y = model(inputs, None, Use_label= False)

        # calculate loss and backward
        if fact is None:
            dtw, out, dtw_loss_corr, dtw_loss_org = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda, test=True)
            loss = dtw
        else:
            dtw, out, dtw_loss_corr, dtw_loss_org = dtw_loss(originals, deltas, targets, criterion, attentions=att, is_cuda=is_cuda, test=True)
            l1 = fact * torch.sum(torch.abs(att))
            loss = (dtw + l1)
        preds['in'] = preds['in'] + originals
        preds['out'] = preds['out'] + out
        preds['targ'] = preds['targ'] + targets
        preds['att'] = preds['att'] + [att[j].detach().cpu().numpy() for j in range(att.shape[0])]
        test_l.update(loss.cpu().data.numpy(), batch_size)

        label_lossCor_lossOrg = {'labels': labels, 'loss_corr': dtw_loss_corr, 'loss_org':dtw_loss_org}
    return test_l.avg, preds, label_lossCor_lossOrg

def train_class(train_loader, model, optimizer, is_cuda=False, level=0):
    tr_l = AccumLoss()

    if level == 0:
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.3, 1, 0.5, 1, 1]))
    else:
        criterion = nn.NLLLoss()
    model.train()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(train_loader):
        # pdb.set_trace() 
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        if train_loader.dataset.name == 'EC3D':
            labels = get_labels([train_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        elif train_loader.dataset.name == 'NTU60':
            labels = torch.from_numpy(np.array([train_loader.dataset.labels[int(i)] for i in batch_id])).long().cuda()
        
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

    return tr_l.avg, 100 * correct / total

def evaluate_class(val_loader, model, is_cuda=False, level=0):
    val_l = AccumLoss()

    if level == 0:
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.5, 1, 0.5, 1, 0.5]))
    else:
        criterion = nn.NLLLoss()
    model.eval()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(val_loader):
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()
        
        if val_loader.dataset.name == 'EC3D':
            labels = get_labels([val_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        elif val_loader.dataset.name == 'NTU60':
            labels = torch.from_numpy(np.array([val_loader.dataset.labels[int(i)] for i in batch_id])).long().cuda()
        
        batch_size = inputs.shape[0]

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)

        # update the training loss
        val_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

    return val_l.avg, 100 * correct / total

def test_class(test_loader, model, is_cuda=False, level=0):
    te_l = AccumLoss()

    if level == 0:
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.5, 1, 0.5, 1, 0.5]))
    else:
        criterion = nn.NLLLoss()
    model.eval()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(test_loader):
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()
        if test_loader.dataset.name == 'EC3D':
            # import pdb; pdb.set_trace()
            labels = get_labels([test_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        elif test_loader.dataset.name == 'NTU60':
            labels = torch.from_numpy(np.array([test_loader.dataset.labels[int(i)] for i in batch_id])).long().cuda()
        batch_size = inputs.shape[0]

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        # import pdb
        # pdb.set_trace()
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)

        # update the training loss
        te_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

        # summary = np.vstack((labels.numpy(), predicted.numpy()))
        summary = torch.stack((labels, predicted), dim=1)
        if level == 0:
            cmt = torch.zeros(6, 6, dtype=torch.int64)
        elif level == 1:
            cmt = torch.zeros(12, 12, dtype=torch.int64)
        else: 
            cmt = torch.zeros(60, 60, dtype=torch.int64)
        # pdb.set_trace()
        for pp in summary:
            tl, pl = pp.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1

    return te_l.avg, 100 * correct / total, summary, cmt

def test_class_v1(test_loader, model, is_cuda=False, level=0):
    te_l = AccumLoss()

    if level == 0:
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.5, 1, 0.5, 1, 0.5]))
    else:
        criterion = nn.NLLLoss()
    model.eval()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(test_loader):

        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([test_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        batch_size = inputs.shape[0]

        _, _, outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)

        # update the training loss
        te_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

        # summary = np.vstack((labels.numpy(), predicted.numpy()))
        summary = torch.stack((labels, predicted), dim=1)
        if level == 0:
            cmt = torch.zeros(6, 6, dtype=torch.int64)
        else:
            cmt = torch.zeros(12, 12, dtype=torch.int64)
        for p in summary:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1

    return te_l.avg, 100 * correct / total, summary, cmt

def test_class_v4(test_loader, model, is_cuda=False, level=0, Use_label=False):
    te_l = AccumLoss()

    if level == 0:
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.5, 1, 0.5, 1, 0.5]))
    else:
        criterion = nn.NLLLoss()
    model.eval()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(test_loader):

        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([test_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level).cuda()
        batch_size = inputs.shape[0]

        _, _, outputs = model(inputs, None, Use_label= False)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)

        # update the training loss
        te_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

        # summary = np.vstack((labels.numpy(), predicted.numpy()))
        summary = torch.stack((labels, predicted), dim=1)
        if level == 0:
            cmt = torch.zeros(6, 6, dtype=torch.int64)
        else:
            cmt = torch.zeros(12, 12, dtype=torch.int64)
        for p in summary:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1

    return te_l.avg, 100 * correct / total, summary, cmt


############ torch dct stuff

def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    V2 = torch.view_as_complex(V)
    v = torch.view_as_real(torch.fft.ifft(V2))[:,:,0]
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def display_poses(poses_list, save_loc=None, custom_name=None, time=0, custom_title=None, legend_=None, color_list=None):
    fig = plt.figure(figsize=(4,4))
    bone_connections = [[0,1],[1,2],[1,5],[1,8],[2,3],[3,4],[5,6],[6,7],[8,9],[8,12],[9,10],
                        [10,11],[11,17],[11,18],[12,13],[13,14],[14,15]]
    
    #  {0: [0, 1], 1: [1, 2, 5, 8], 2: [2, 3], 3:[3, 4], 4: [4], 5: [5, 6], 6: [6, 7], 7: [7], 
    #      8: [8, 9, 12], 9: [9, 10], 10: [10, 11], 11: [11, 22, 24], 12: [12, 13], 13: [13, 14], 
    #      14: [14, 19, 21], 19: [19], 21: [21], 22: [22], 24: [24]}


    ax = fig.add_subplot(111,  projection='3d')
    plots = []
    for ind, poses in enumerate(poses_list):
        X = poses[2,:]
        Y = poses[1,:]
        Z = poses[0,:]
        for _, bone in enumerate(bone_connections):
            bone = list(bone)
            p, = ax.plot(X[bone], Y[bone], Z[bone], c=color_list[ind], marker="o", markersize=6, linewidth=3)

            if len(plots) <= ind:
                plots.append(p)

        ax.set_xlim(-600,600)
        ax.set_ylim(-600,600)
        ax.set_zlim(-600,600)

        ax.view_init(elev=-90., azim=-90)

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Get rid of the panes                          
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 

    # Get rid of the spines                         
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    if custom_title is not None:
        ax.set_title(custom_title)

    if legend_ is not None:
        ax.legend(plots, legend_)

    if save_loc is not None:
        plt.savefig(save_loc + '/' + custom_name + str(time) + '.png', dpi=100)

    plt.close(fig)
    return fig