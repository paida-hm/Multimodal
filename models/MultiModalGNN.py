# -*- codeing =utf-8 -*-
# @Time : 2025/3/13 22:14
# @Author :gh
# @File :MultiModalGNN.py
# @Software: PyCharm
import torch
import torch.nn as nn
import dgl

class MultiModalGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, smiles_dim=0):
        """
        node_dim: 节点特征维度 (2D分子图)
        edge_dim: 边特征维度 (2D分子图)
        hidden_dim: 隐藏层维度
        smiles_dim: SMILES输入的维度（如果不使用SMILES，默认为0）
        """
        super(MultiModalGNN, self).__init__()

        # 图神经网络（处理2D分子图和3D结构）
        self.gnn = GNN(node_dim, edge_dim, hidden_dim)

        # 处理SMILES序列的嵌入层
        if smiles_dim > 0:
            self.smiles_encoder = nn.Embedding(num_embeddings=128, embedding_dim=hidden_dim)
            self.use_smiles = True
        else:
            self.use_smiles = False

        # 结合图特征和SMILES特征的全连接层
        fusion_input_dim = hidden_dim * (2 if self.use_smiles else 1)
        self.fc = nn.Linear(fusion_input_dim, hidden_dim)

    def forward(self, graph, smiles=None):
        """
        graph: DGL 图对象 (2D/3D分子图)
        smiles: SMILES 序列化张量 (1D SMILES)，可选
        """
        # 提取图特征
        graph_emb = self.gnn(graph)

        if self.use_smiles:
            smiles_emb = self.smiles_encoder(smiles).mean(dim=1)  # 平均池化
            fusion = torch.cat([graph_emb, smiles_emb], dim=-1)  # 融合
        else:
            fusion = graph_emb

        output = self.fc(fusion)  # 经过融合后的全连接层
        return output
