import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as Aug
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import pandas as pd
torch.cuda.empty_cache()


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    Algo = "GRACE"
    d = "cpu" #cuda
    device = torch.device(d)
    dsName = 'CiteSeer'
    path = osp.join(osp.expanduser('~'), 'datasets', 'Planetoid')
    dataset = Planetoid(path, name=dsName)  # transform
    data = dataset[0].to(device)
    drop_prob = 1
    percentage = 0.3
    pf = str(drop_prob) + 'x' + str(percentage)  # drop_prob x Percentage

    LeastOrMost = "Least"  # Least #Most  #least mean Least imp drop, Most mean most imp. For VR Most mean imp will updated
    infvsRand='Inf' #Rand
    outputFile = "OutputFile.csv"  # File To Save  results
    df = pd.read_csv(outputFile, index_col=0)
    for rounds in range(0, 20):
        print("Round: ", rounds)
        aug1 = Aug.Compose([Aug.EdgeRemoving(pe=0.4), Aug.FeatureMasking(pf=0.4)])  # sending as list
        # aug2 = Aug.Compose([Aug.EdgeRemoving(pe=0.2), Aug.FeatureMasking(pf=0.3)])  # sending as list
        aug2 = Aug.Compose([Aug.EdgeRemoving(pe=0.2),
                            Aug.ImportanceFeatures(data, drop_prob, device, percentage, dsName, LeastOrMost,infvsRand)])

        gconv = GConv(input_dim=dataset.num_features, hidden_dim=256, activation=torch.nn.PReLU, num_layers=2).to(
            device)
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=256, proj_dim=256).to(device)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.9), mode='L2L', intraview_negs=True).to(device)
        optimizer = Adam(encoder_model.parameters(), lr=0.00001)

        epochs = 2001
        r = []
        r2 = []

        with tqdm(total=2000, desc='(T)') as pbar:
            for epoch in range(1, epochs):
                loss = train(encoder_model, contrast_model, data, optimizer)
                pbar.set_postfix({'loss': loss})
                pbar.update()
                if epoch % 50 == 0 and epoch >= 200:
                    test_result = test(encoder_model, data)
                    r.append(test_result["micro_f1"])
                    r2.append(test_result["macro_f1"])
        max_value = max(r)
        max_value2 = max(r2)
        df = df.append(row_to_add, ignore_index=True)
        values_to_add = {'Algo': Algo, 'Dataset': dsName, 'Epochs': epochs - 1, 'TotalDrop': pf, 'MicroF1': max_value,
                         'MacroF1': max_value2, 'Value': 1,
                         'View1': 'Original', 'View2': 'FebAA',
                         'LeastOrMost': LeastOrMost}
        row_to_add = pd.Series(values_to_add)
        df = df.append(row_to_add, ignore_index=True)
        # least mean least masking, most mean most masking
        df.to_csv(outputFile)


if __name__ == '__main__':
    main()

