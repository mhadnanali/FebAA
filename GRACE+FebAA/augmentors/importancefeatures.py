import os
import pandas as pd
from GCL.augmentors import FindInfFeatures
from GCL.augmentors.augmentor import Graph, Augmentor
import torch
class ImportanceFeatures(Augmentor):
    def __init__(self, data,drop_prob,device,percentage,dsName,LeastOrMost,infvsRand):
        super(ImportanceFeatures, self).__init__()
        self.device=device
        self.dsName=dsName
        self.drop_prob=drop_prob
        self.percentage=percentage
        self.data=data
        self.data.to(self.device)
        self.LeastOrMost=LeastOrMost
        self.infvsRand=infvsRand
        self.topfeatures = self.TopFeatureFinder(self.data, self.drop_prob, self.device, self.percentage,
                                            self.dsName, self.data.x,self.LeastOrMost,self.infvsRand)

    def augment(self, g: Graph) -> Graph:
        x, self.edge_index_, edge_weights = g.unfold()
        indices = torch.tensor(self.topfeatures, device=self.device)
        candidateCF = torch.index_select(x, 1, indices)
        XT = self.DropFeatures(x, candidateCF, indices, self.drop_prob, self.device,self.topfeatures)
        return Graph(x=XT, edge_index=self.edge_index_, edge_weights=edge_weights)


    def TopFeatureFinder(self,data,drop_prob,device,percentage,dsName,x,LeastOrMost,infvsRand):
            infvsRand="Features/"+infvsRand
            totalFeatures = x.size()[1]
            totalRows = x.size()[0]
            absolute_path = os.path.abspath(infvsRand+" Features "+dsName+".csv")
            if (os.path.isfile(infvsRand+" Features "+dsName+".csv")) == True:
                    resultsAnalysis = pd.read_csv(infvsRand+" Features "+dsName+".csv")
                    topfeatures = resultsAnalysis['Range']
                    topfeatures = topfeatures.tolist()
                    print(infvsRand+" Features "+dsName+".csv already exists")
            else:
                    print("File does not exists Finding...",infvsRand+" Features "+dsName+".csv")
                    tf = FindInfFeatures.InfFeatureFindClass(data, drop_prob, percentage, device, dsName,infvsRand)
                    topfeatures = tf.TopFeaturesFind()
            tenper = (totalFeatures * percentage)
            if (LeastOrMost=="Least"):
                # incase of Least the least important features will be taken for masking
                topfeatures = topfeatures[:int(tenper)]
            elif(LeastOrMost=="Most"):
                #incase of Most the most important features will be taken for masking
                topfeatures.reverse()
                topfeatures = topfeatures[:int(tenper)]
            return topfeatures

    def DropFeatures(self,X, candidateCF,indices, drop_prob, device,topfeatures):
        mask = torch.empty(candidateCF.size(), device=device).uniform_() >= drop_prob
        output = candidateCF.mul(mask)
        X[:, indices] = output
        return X
