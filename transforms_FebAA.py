import copy

import torch
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.transforms import Compose


class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)

class ImportanceFeatures:
    def __init__(self, data,topfeatures,drop_prob,device):
        self.device=device
        self.drop_prob=drop_prob
        self.topfeatures=topfeatures
        data.to(self.device)
        self.indices = torch.tensor(self.topfeatures, device=self.device)  # sorting them will make the process faster?
        self.candidateCF = torch.index_select(data.x, 1, self.indices)
        print("DropFeatures: Candidate Features: ", len(self.topfeatures), ' out of Total: ', data.x.size(1))
        #print("calling __init__")

    def __call__(self, data):
            #X = data.x
            mask = torch.empty(self.candidateCF.size(), device=self.device).uniform_() >= self.drop_prob
            output = self.candidateCF.mul(mask)
            data.x[:, self.indices] = output
            return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.drop_prob)



def TopFeatureFinder(percentage, dsName,x,LeastOrMost, inforrandom):
            import pandas as pd
            import os
            inforrandom="Features/" +inforrandom
            totalFeatures = x.size()[1]
            if (os.path.isfile(inforrandom+" Features "+dsName+".csv")) == True:
                    resultsAnalysis = pd.read_csv(inforrandom+" Features " + dsName + ".csv")
                    topfeatures = resultsAnalysis['Range']
                    topfeatures = topfeatures.tolist()
                    print("File exists ", inforrandom+" Features "+dsName+".csv")
            else:
                    print("File does not exists..., Exiting...",inforrandom+" Features "+dsName+".csv")
                    exit()

            tenper = (totalFeatures * percentage)
            if (LeastOrMost=="Least"):
                #incase of Least the least important features will be taken
                #By default is least imp
                topfeatures = topfeatures[:int(tenper)] #crop top  features as per given percentage
            elif(LeastOrMost=="Most"):
                #incase of Most the most important features will be taken
                topfeatures.reverse()
                topfeatures = topfeatures[:int(tenper)]
            return topfeatures


def get_TopGraph(drop_edge_p, data, drop_prob, device, percentage, dsName, LeastOrMost,inforrandom):
    transforms = list()
    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if percentage > 0.:
        topfeatures = TopFeatureFinder(percentage,dsName, data.x, LeastOrMost, inforrandom)
        transforms.append(ImportanceFeatures(data, topfeatures,drop_prob, device))

    return Compose(transforms)

def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)


