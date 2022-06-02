from torch_geometric.utils import to_undirected
from torch.utils.data import random_split
import copy
import torch
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import pandas as pd
from GCL.models import BootstrapContrast
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from torch_geometric.nn import GCNConv


class InfFeatureFindClass():
    def __init__(self, data,drop_prob,percentage,device,dbName,infvsRand):#, pf: float):
        self.device = torch.device(device)
        self.drop_prob=drop_prob
        self.percentage=percentage
        self.dbName=dbName
        self.datas=data
        self.Orignaldata = data
        self.edge_index_ = to_undirected(self.datas.edge_index)
        self.edge_index_.to(self.device)
        self.datas.to(self.device)
        self.infvsRand=infvsRand


    def FeatureRandomization(self, datas, ranges):
        #make the particular row 0
        indices = torch.tensor([ranges], device=self.device)
        candidateCF = torch.index_select(datas.x, 1, indices)
        mask = torch.empty(candidateCF.size(), device=self.device).uniform_() >= 1
        output = candidateCF.mul(mask)
        datas.x[:, indices] = output
        return self.datas

    def TopFeaturesFind(self):
        device=self.device
        totalFeatures = len(self.datas.x[0])
        firstflag = 1
        resultsAnalysis = pd.DataFrame(columns=['DataSet', 'TotalFeat', 'Range'])
        for i in range(0, 3):
            print("This is round ", i)
            test_list = []
            for onerange in range(0, totalFeatures):
                print("Before Range ", self.datas.x.sum())
                #print(torch.count_nonzero(self.datas.x, dim=0))
                self.datas.x = self.Orignaldata.x
                self.datas = self.FeatureRandomization(self.datas, onerange)
                print("After ",self.datas.x.sum())
                aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
                aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

                gconv = GConv(input_dim=totalFeatures, hidden_dim=256, num_layers=2).to(device)
                encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=256).to(device)
                contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)
                optimizer = Adam(encoder_model.parameters(), lr=0.01)

                print("Training without the ", onerange)
                r = []
                with tqdm(total=2, desc='(T)') as pbar:
                    for epoch in range(1, 3):
                        loss = train(encoder_model, contrast_model, self.datas, optimizer)
                        pbar.set_postfix({'loss': loss})
                        pbar.update()
                        if epoch%1==0:
                            test_result = test(encoder_model, self.datas)
                            r.append(test_result["micro_f1"])
                            print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, Epoch={epoch}')
                max_value = max(r)
                test_list.append(max_value)
                print(f'Test Accuracy: {max_value:.4f}')
                if firstflag != 0:
                    values_to_add = {'DataSet':  self.dbName, 'TotalFeat': totalFeatures, 'Range': str(onerange)}
                    row_to_add = pd.Series(values_to_add)
                    resultsAnalysis = resultsAnalysis.append(row_to_add, ignore_index=True)

            firstflag = 0
            resultsAnalysis["Round " + str(i + 1)] = test_list
        average = resultsAnalysis
        average = average.drop(['DataSet', 'TotalFeat', 'Range'], axis=1)
        average['mean_rows'] = average.mean(axis=1)
        resultsAnalysis['Average'] = average['mean_rows']
        resultsAnalysis.sort_values(by=['Average'], inplace=True, ascending=True)
        resultsAnalysis.to_csv(self.infvsRand+" Features "+self.dbName+".csv")
        totalFeatures = len(self.datas.x[0])
        resultsAnalysis = pd.read_csv(self.infvsRand+" Features "+self.dbName+".csv")
        topfeatures = resultsAnalysis['Range']
        topfeatures = topfeatures.tolist()
        tenper = totalFeatures / self.percentage
        topfeatures = topfeatures[:int(tenper)]
        return topfeatures


    def generate_split(self, num_samples: int, train_ratio: float, val_ratio: float):
        #print("Spliting the Data...")
        train_len = int(num_samples * train_ratio)
        val_len = int(num_samples * val_ratio)
        test_len = num_samples - train_len - val_len
        train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))
        idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
        train_mask = torch.zeros((num_samples,)).to(torch.bool)
        test_mask = torch.zeros((num_samples,)).to(torch.bool)
        val_mask = torch.zeros((num_samples,)).to(torch.bool)
        train_mask[idx_train] = True
        test_mask[idx_test] = True
        val_mask[idx_val] = True
        return train_mask, test_mask, val_mask

class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)

            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum: float):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())
    loss.backward()
    optimizer.step()
    encoder_model.update_target_encoder(0.99)
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    h1, h2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result
