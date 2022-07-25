import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Define a Graph convolutional layer with a learnable adjacency matrix
    """

    def __init__(self, in_features, out_features, bias=True, node_n=57):

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.adj = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.adj.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    """
    Define a residual block of GCN
    """

    def __init__(self, in_features, p_dropout, bias=True, node_n=57):

        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()

    def forward(self, x):
        y = self.gc1(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_corr_class_v1_origin(nn.Module):
    def __init__(self, input_feature=25, hidden_feature=128, p_dropout=0.5, \
                 num_stage=2, node_n=57, classes=12):
        """
                :param input_feature: num of input feature
                :param hidden_feature: num of hidden feature
                :param p_dropout: drop out prob.
                :param num_stage: number of residual blocks
                :param node_n: number of nodes in graph
         """
        super(GCN_corr_class_v1_origin, self).__init__()
        self.num_stage = num_stage
        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)

        self.gcin2 = GraphConvolution(input_feature, hidden_feature,
                                      node_n=node_n)  # for the second part of classifier
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)  # The first bn that shared

        # self.gcbs = []
        # self.gcbs = self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n)
        # self.gcbs = nn.ModuleList(self.gcbs)

        self.gcout_corr = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.gcatt = GraphConvolution(hidden_feature, 1, node_n=node_n)  # Attention

        self.gcout_class = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bn2 = nn.BatchNorm1d(node_n * input_feature)  # The second bn that exclusively for classifier
        self.lin = nn.Linear(node_n * input_feature, classes)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.Sigmoid()
        self.act_flin = nn.LogSoftmax(dim=1)

    def forward(self, x):

        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape


        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gcbs(y)  # The shared part stop at here

        # For the corr part
        y_corr = self.gcbs(y)
        out = self.gcout_corr(y_corr)
        att = self.gcatt(y_corr)
        att = self.act_fatt(att)  # These are for attention

        # For the class part
        y_class = self.gcout_class(y)
        if b > 1:
            y_class = self.bn2(y_class.view(b, -1)).view(y_class.shape)
        y_class = self.act_f(y_class)
        y_class = self.do(y_class)

        node_n = 57
        input_feature =25
        # y_class = y_class.view(-1, n * f)
        y_class = y_class.view(-1, node_n * input_feature)

        y_class = self.lin(y_class)
        y_class = self.act_flin(y_class)

        return out, att, y_class


class GCN_corr_class_v1_22Spring(nn.Module):

    def __init__(self, input_feature=25, hidden_feature=128, p_dropout=0.5, \
                 num_stage=2, node_n=57, classes=12):
        """
                :param input_feature: num of input feature
                :param hidden_feature: num of hidden feature
                :param p_dropout: drop out prob.
                :param num_stage: number of residual blocks
                :param node_n: number of nodes in graph
         """
        super(GCN_corr_class_v1_22Spring, self).__init__()
        self.num_stage = num_stage
        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)

        self.gcin2 = GraphConvolution(input_feature, hidden_feature,
                                      node_n=node_n)  # for the second part of classifier
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)  # The first bn that shared

        # self.gcbs = []
        # self.gcbs = self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n)
        # self.gcbs = nn.ModuleList(self.gcbs)

        self.gcout_corr = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.gcatt = GraphConvolution(hidden_feature, 1, node_n=node_n)  # Attention

        self.gcout_class = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bn2 = nn.BatchNorm1d(node_n * input_feature)  # The second bn that exclusively for classifier
        self.lin = nn.Linear(node_n * input_feature, classes)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.Sigmoid()
        self.act_flin = nn.LogSoftmax(dim=1)

        # model reconstruct
        self.avgpool = torch.nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2d = nn.Conv2d(3, 3, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.gcont_class_v2 = GraphConvolution(int(hidden_feature/2), input_feature, node_n=node_n)
        self.maxpool = nn.AdaptiveMaxPool2d((57, 25))


    def forward(self, x):

        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape

        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gcbs(y)  # The shared part stop at here

        # For the corr part
        y_corr = self.gcbs(y)
        out = self.gcout_corr(y_corr)
        att = self.gcatt(y_corr)
        att = self.act_fatt(att)  # These are for attention

        # For the class part
        # model 22Spring
        # y = self.gcbs(y) 
        y = self.avgpool(y) # --> batch*mode_n*features/2
        y = y.view(y.shape[0], 3, 19, y.shape[2]) 
        y = self.conv2d(y)  # --> batch,3,25,features/2
        y = y.view(y.shape[0], 57, y.shape[3]) 
        y_class = self.gcont_class_v2(y)
        if b > 1:
            y_class = self.bn2(y_class.view(b, -1)).view(y_class.shape)
        y_class = self.act_f(y_class)
        y_class = self.do(y_class)
        # y = self.maxpool(y)
        node_n = 57
        input_feature =25
        y_class = y_class.view(-1, node_n * input_feature)
        y_class = self.lin(y_class)
        y_class = self.act_flin(y_class)

        return out, att, y_class


class GCN_corr_class_v2(nn.Module):

    def __init__(self, input_feature=25, hidden_feature=128, p_dropout=0.5, \
                 num_stage=2, node_n=57, classes=12):
        """
                :param input_feature: num of input feature
                :param hidden_feature: num of hidden feature
                :param p_dropout: drop out prob.
                :param num_stage: number of residual blocks
                :param node_n: number of nodes in graph
                """
        super(GCN_corr_class_v2, self).__init__()
        self.num_stage = num_stage
        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)

        self.gcin2 = GraphConvolution(input_feature, hidden_feature,
                                      node_n=node_n)  # for the second part of classifier
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)  # The first bn that shared

        # self.gcbs = []
        # self.gcbs = self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gcbs1 = GC_Block(hidden_feature+1, p_dropout=p_dropout, node_n=node_n)

        # self.gcbs = nn.ModuleList(self.gcbs)

        self.gcout_corr = GraphConvolution(hidden_feature+1, input_feature, node_n=node_n)
        self.gcatt = GraphConvolution(hidden_feature+1, 1, node_n=node_n)  # Attention

        self.gcout_class = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bn2 = nn.BatchNorm1d(node_n * input_feature)  # The second bn that exclusively for classifier
        self.lin = nn.Linear(node_n * input_feature, classes)
        self.linearLayer = nn.Linear(12, 57)


        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.Sigmoid()
        self.act_flin = nn.LogSoftmax(dim=1)

    def forward(self, x, labels):

        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape

        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gcbs(y)  # The shared part stop at here

        # For the corr part
        ''''# Concatenate the labels for the corrector!!!!!!!!!!!!!!!!!'''
        batch_size = x.shape[0]
        one_hot_labels = torch.zeros(batch_size, 12).cuda()
        for ind, elem in enumerate(labels):
            one_hot_labels[ind, elem] = 1

        newFeature = self.linearLayer(one_hot_labels)
        newFeature_3d = newFeature.unsqueeze(2)

        ''' Concatenate groundtruth with the orginal inputs'''
        yWithLabels = torch.cat([newFeature_3d, y], dim=2)

        y_corr = self.gcbs1(yWithLabels)
        out = self.gcout_corr(y_corr)
        att = self.gcatt(y_corr)
        att = self.act_fatt(att)  # These are for attention

        # For the class part
        y_class = self.gcout_class(y)
        if b > 1:
            y_class = self.bn2(y_class.view(b, -1)).view(y_class.shape)
        y_class = self.act_f(y_class)
        y_class = self.do(y_class)

        node_n = 57
        input_feature =25
        # y_class = y_class.view(-1, n * f)
        y_class = y_class.view(-1, node_n * input_feature)

        y_class = self.lin(y_class)
        y_class = self.act_flin(y_class)

        return out, att, y_class
        # return y

class GCN_corr_class_v3(nn.Module):

    def __init__(self, input_feature=25, hidden_feature=128, p_dropout=0.5, \
                 num_stage=2, node_n=57, classes=12):
        """
                :param input_feature: num of input feature
                :param hidden_feature: num of hidden feature
                :param p_dropout: drop out prob.
                :param num_stage: number of residual blocks
                :param node_n: number of nodes in graph
                """
        super(GCN_corr_class_v3, self).__init__()
        self.num_stage = num_stage
        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)

        self.gcin2 = GraphConvolution(input_feature, hidden_feature,
                                      node_n=node_n)  # for the second part of classifier
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)  # The first bn that shared

        # self.gcbs = []
        # self.gcbs = self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gcbs1 = GC_Block(hidden_feature+1, p_dropout=p_dropout, node_n=node_n)

        # self.gcbs = nn.ModuleList(self.gcbs)

        self.gcout_corr = GraphConvolution(hidden_feature+1, input_feature, node_n=node_n)
        self.gcatt = GraphConvolution(hidden_feature+1, 1, node_n=node_n)  # Attention

        self.gcout_class = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bn2 = nn.BatchNorm1d(node_n * input_feature)  # The second bn that exclusively for classifier
        self.lin = nn.Linear(node_n * input_feature, classes)
        self.linearLayer = nn.Linear(12, 57)


        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.Sigmoid()
        self.act_flin = nn.LogSoftmax(dim=1)

    def forward(self, x, labels):

        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape

        # if len(x.shape) == 3:
        #     b, n, f = x.shape
        # else:
        #     b = 1
        #     n, f = x.shape
        #
        # y = self.gcin(x)

        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gcbs(y)  # The shared part stop at here

        # For the corr part
        ''''# Concatenate the labels for the corrector!!!!!!!!!!!!!!!!!'''
        batch_size = x.shape[0]
        one_hot_labels = torch.zeros(batch_size, 12).cuda()

        # zeroMatrix = torch.zeros(batch_size).cuda()
        # if labels != zeroMatrix:

        # if labels != torch.zeros(batch_size).long().cuda():
        if torch.tensor(labels).equal(torch.zeros(batch_size).long().cuda()) == False:
            for ind, elem in enumerate(labels):
                one_hot_labels[ind, elem] = 1

        newFeature = self.linearLayer(one_hot_labels)
        newFeature_3d = newFeature.unsqueeze(2)

        ''' Concatenate groundtruth with the orginal inputs'''
        yWithLabels = torch.cat([newFeature_3d, y], dim=2)

        y_corr = self.gcbs1(yWithLabels)
        out = self.gcout_corr(y_corr)
        att = self.gcatt(y_corr)
        att = self.act_fatt(att)  # These are for attention

        # For the class part
        y_class = self.gcout_class(y)
        if b > 1:
            y_class = self.bn2(y_class.view(b, -1)).view(y_class.shape)
        y_class = self.act_f(y_class)
        y_class = self.do(y_class)

        node_n = 57
        input_feature =25
        # y_class = y_class.view(-1, n * f)
        y_class = y_class.view(-1, node_n * input_feature)

        y_class = self.lin(y_class)
        y_class = self.act_flin(y_class)

        return out, att, y_class
        # return y

class GCN_corr_class_v4(nn.Module):

    def __init__(self, input_feature=25, hidden_feature=128, p_dropout=0.5, \
                 num_stage=2, node_n=57, classes=12):
        """
                :param input_feature: num of input feature
                :param hidden_feature: num of hidden feature
                :param p_dropout: drop out prob.
                :param num_stage: number of residual blocks
                :param node_n: number of nodes in graph
                """
        super(GCN_corr_class_v4, self).__init__()
        self.num_stage = num_stage
        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)

        self.gcin2 = GraphConvolution(input_feature, hidden_feature,
                                      node_n=node_n)  # for the second part of classifier
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)  # The first bn that shared

        # self.gcbs = []
        # self.gcbs = self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gcbs1 = GC_Block(hidden_feature+1, p_dropout=p_dropout, node_n=node_n)

        # self.gcbs = nn.ModuleList(self.gcbs)

        self.gcout_corr = GraphConvolution(hidden_feature+1, input_feature, node_n=node_n)
        self.gcatt = GraphConvolution(hidden_feature+1, 1, node_n=node_n)  # Attention

        self.gcout_class = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bn2 = nn.BatchNorm1d(node_n * input_feature)  # The second bn that exclusively for classifier
        self.lin = nn.Linear(node_n * input_feature, classes)
        self.linearLayer = nn.Linear(12, 57)


        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.Sigmoid()
        self.act_flin = nn.LogSoftmax(dim=1)

    def forward(self, x, labels, Use_label):
        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape


        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gcbs(y)  # The shared part stop at here

        # For the class part
        y_class = self.gcout_class(y)
        if b > 1:
            y_class = self.bn2(y_class.view(b, -1)).view(y_class.shape)
        y_class = self.act_f(y_class)
        y_class = self.do(y_class)

        node_n = 57
        input_feature = 25
        # y_class = y_class.view(-1, n * f)
        y_class = y_class.view(-1, node_n * input_feature)

        y_class = self.lin(y_class)
        y_class = self.act_flin(y_class)


        # For the corr part
        ''''# Concatenate the labels for the corrector!!!!!!!!!!!!!!!!!'''
        batch_size = x.shape[0]
        one_hot_labels = torch.zeros(batch_size, 12).cuda()
        # if labels != torch.zeros(batch_size):
        if Use_label == True:
            for ind, elem in enumerate(labels):
                one_hot_labels[ind, elem] = 1
        else:
            _, predicted = torch.max(y_class.data, 1)
            for ind, elem in enumerate(predicted):
                one_hot_labels[ind, elem] = 1

        newFeature = self.linearLayer(one_hot_labels)
        newFeature_3d = newFeature.unsqueeze(2)

        ''' Concatenate groundtruth with the orginal inputs'''
        yWithLabels = torch.cat([newFeature_3d, y], dim=2)

        y_corr = self.gcbs1(yWithLabels)
        out = self.gcout_corr(y_corr)
        att = self.gcatt(y_corr)
        att = self.act_fatt(att)  # These are for attention

        return out, att, y_class


class GCN_corr_class_v4_22Spring(nn.Module):

    def __init__(self, input_feature=25, hidden_feature=128, p_dropout=0.5, \
                 num_stage=2, node_n=57, classes=12):
        """
            :param input_feature: num of input feature
            :param hidden_feature: num of hidden feature
            :param p_dropout: drop out prob.
            :param num_stage: number of residual blocks
            :param node_n: number of nodes in graph
        """
        super(GCN_corr_class_v4_22Spring, self).__init__()
        self.num_stage = num_stage
        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)

        self.gcin2 = GraphConvolution(input_feature, hidden_feature,
                                      node_n=node_n)  # for the second part of classifier
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)  # The first bn that shared

        
        self.gcbs = GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gcbs1 = GC_Block(hidden_feature+1, p_dropout=p_dropout, node_n=node_n)

        self.gcout_corr = GraphConvolution(hidden_feature+1, input_feature, node_n=node_n)
        self.gcatt = GraphConvolution(hidden_feature+1, 1, node_n=node_n)  # Attention

        self.gcout_class = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bn2 = nn.BatchNorm1d(node_n * input_feature)  # The second bn that exclusively for classifier
        self.lin = nn.Linear(node_n * input_feature, classes)
        self.linearLayer = nn.Linear(12, 57)


        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.Sigmoid()
        self.act_flin = nn.LogSoftmax(dim=1)

        # model reconstruct
        # for classifier
        self.avgpool = torch.nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2d = nn.Conv2d(3, 3, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.gcont_class_v2 = GraphConvolution(int(hidden_feature/2), input_feature, node_n=node_n)
        self.maxpool = nn.AdaptiveMaxPool2d((57, 25))
        # for corrector
        # self.gcout_corr_22S = GraphConvolution(int(hidden_feature/2), input_feature, node_n=node_n)


    def forward(self, x, labels, Use_label):
        # import pdb; pdb.set_trace()

        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape

        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y_shared = self.do(y)

        # For the class part
        y_class = self.avgpool(y_shared) # --> batch*mode_n*features/2
        y_class = y_class.view(y_class.shape[0], 3, 19, y_class.shape[2]) 
        y_class = self.conv2d(y_class)  # --> batch,3,25,features/2
        y_class = y_class.view(y_class.shape[0], 57, y_class.shape[3]) 
        y_class = self.gcont_class_v2(y_class)
        if b > 1:
            y_class = self.bn2(y_class.view(b, -1)).view(y_class.shape)
        y_class = self.act_f(y_class)
        y_class = self.do(y_class)
        # y_class = self.maxpool(y_class)
        node_n = 57
        input_feature =25
        y_class = y_class.view(-1, node_n * input_feature)

        y_class = self.lin(y_class)
        y_class = self.act_flin(y_class)

        # For the corr part
        y_corr = self.gcbs(y_shared)  # --> batch,node,hiden_features
        ''''Apply Feedback: Curriculum Learning strategy '''
        batch_size = x.shape[0]
        one_hot_labels = torch.zeros(batch_size, 12).cuda()
        if Use_label == True:
            for ind, elem in enumerate(labels):
                one_hot_labels[ind, elem] = 1
        else:
            _, predicted = torch.max(y_class.data, 1)
            for ind, elem in enumerate(predicted):
                one_hot_labels[ind, elem] = 1

        newFeature = self.linearLayer(one_hot_labels)
        newFeature_3d = newFeature.unsqueeze(2)

        ''' Concatenate groundtruth with the orginal inputs'''
        yWithLabels = torch.cat([newFeature_3d, y_corr], dim=2) #shape: (batch,node,hiddenFeatures+1) 

        y_corr = self.gcbs1(yWithLabels)
        
        
        """continue with previous designed model"""
        out = self.gcout_corr(y_corr)
        att = self.gcatt(y_corr)
        att = self.act_fatt(att)  # These are for attention

        return out, att, y_class


class GCN_corr(nn.Module):

    def __init__(self, input_feature=25, hidden_feature=128, p_dropout=0.5, num_stage=2, node_n=57):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_corr, self).__init__()
        self.num_stage = num_stage

        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gcout = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.gcatt = GraphConvolution(hidden_feature, 1, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_fatt = nn.Sigmoid()

    def forward(self, x):

        y = self.gcin(x)
        if len(y.shape) == 3:
            b, n, f = y.shape
        else:
            b = 1
            n, f = y.shape

        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        out = self.gcout(y)

        att = self.gcatt(y)
        att = self.act_fatt(att)

        return out, att , y # add y here in order to use "test_corr" with GCN_corr_class at the same time


class GCN_class(nn.Module):

    def __init__(self, input_feature=25, hidden_feature=32, p_dropout=0.5, node_n=57, classes=12):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN_class, self).__init__()

        self.gcin = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.gcout = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bnin = nn.BatchNorm1d(node_n * hidden_feature)
        self.bnout = nn.BatchNorm1d(node_n * input_feature)
        self.lin = nn.Linear(node_n * input_feature, classes)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.ReLU()
        self.act_flin = nn.LogSoftmax(dim=1)

    def forward(self, x):

        if len(x.shape) == 3:
            b, n, f = x.shape
        else:
            b = 1
            n, f = x.shape

        y = self.gcin(x)
        if b > 1:
            y = self.bnin(y.view(b, -1)).view(y.shape)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gcout(y)
        if b > 1:
            y = self.bnout(y.view(b, -1)).view(y.shape)
        y = self.act_f(y)
        y = self.do(y)

        y = y.view(-1, n * f)
        y = self.lin(y)
        y = self.act_flin(y)

        return 1, 2, y # add 1 2 in front of y to get three output: to suit "test_class"