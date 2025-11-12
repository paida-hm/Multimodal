import os
from itertools import chain
from typing import Dict, Callable

import dgl
import torch
from torch import nn

from commons.losses import MultimodalNTXentLoss
from trainer.trainer import Trainer
import json
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np

def decode_smiles_tensor(smiles_tensor, smiles_vocab):
    smiles_list = []
    for row in smiles_tensor:
        tokens = [smiles_vocab[idx.item()] for idx in row if idx.item() != 0]  # 去除 PAD
        smiles_list.append("".join(tokens))
    return smiles_list

class SelfSupervisedTrainer(Trainer):
    def __init__(self, model1d, model2d, model3d, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True,  **kwargs):

        self.device = device
        self.model1d = model1d.to(device) if model1d is not None else None
        self.model2d = model2d.to(device)
        self.model3d = model3d.to(device)
        # self.smiles_model = kwargs.get("smiles_model", None)
        # self.smiles_tokenizer = kwargs.get("smiles_tokenizer", None)

        # self.projector_1d = nn.Linear(768, 256).to(self.device)
        # self.projector_2d = nn.Linear(600, 256).to(device)
        # self.projector_3d = nn.Identity()
        print(f" SelfSupervisedTrainer initialized.")
        print(f" 1D model: {type(self.model1d).__name__}")
        print(f" 2D model: {type(self.model2d).__name__}")
        print(f" 3D model: {type(self.model3d).__name__}")




        # super(SelfSupervisedTrainer, self).__init__(model1d, model2d, model3d, args, metrics, main_metric, device, tensorboard_functions,
        #                                             optim=optim, main_metric_goal=main_metric_goal,
        #                                             loss_func=loss_func, scheduler_step_per_batch=scheduler_step_per_batch)
        super(SelfSupervisedTrainer, self).__init__(model2d, args, metrics, main_metric, device, tensorboard_functions,
                                                    optim, main_metric_goal, loss_func, scheduler_step_per_batch)

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model3d.load_state_dict(checkpoint['model3d_state_dict'])

    def forward_pass(self, batch):
        if getattr(self.args, "finetune", False):
            # === 微调阶段：仅使用2D模型进行回归 ===
            info2d, targets = batch
            preds = self.model2d(*info2d)
            loss = self.loss_func(preds, targets)
            return loss, preds, targets
        else:
            info2d, info3d, info1d, *snorm_n = tuple(batch)
            view2d = self.model2d(*info2d, *snorm_n)
            view3d = self.model3d(*info3d)
            view1d = self.model1d(info1d)
            loss = self.loss_func(view1d, view2d, view3d)
            return loss, view1d, view2d, view3d


    def evaluate_metrics(self, z1d, z2d, z3d, batch=None, val=False) -> Dict[str, float]:
        metric_results = {}

        # 均值与标准差
        if z1d is not None:
            metric_results["mean_pred_1d"] = torch.mean(z1d).item()
            metric_results["std_pred_1d"] = torch.std(z1d).item()
        if z2d is not None:
            metric_results["mean_pred_2d"] = torch.mean(z2d).item()
            metric_results["std_pred_2d"] = torch.std(z2d).item()
        if z3d is not None:
            metric_results["mean_pred_3d"] = torch.mean(z3d).item()
            metric_results["std_pred_3d"] = torch.std(z3d).item()
        # contrastive metric 的处理
        if 'Local' in type(self.loss_func).__name__ and batch is not None:
            node_indices = torch.cumsum(batch[0].batch_num_nodes(), dim=0)
            pos_mask = torch.zeros((len(z2d), len(z3d)), device=z2d.device)
            for graph_idx in range(1, len(node_indices)):
                pos_mask[node_indices[graph_idx - 1]: node_indices[graph_idx], graph_idx] = 1.
            pos_mask[0:node_indices[0], 0] = 1

            for key, metric in self.metrics.items():
                if not hasattr(metric, 'val_only') or val:
                    try:
                        value = metric(z2d, z3d, pos_mask)
                        metric_results[key] = value.item()
                    except Exception:
                        pass  # 避免因为某个 metric 报错影响所有评估
        # 评估各模态之间的正样本相似性
        for key, metric in self.metrics.items():
            if not hasattr(metric, 'val_only') or val:
                metric_results[f'{key}_1d_2d'] = metric(z1d, z2d).item()
                metric_results[f'{key}_1d_3d'] = metric(z1d, z3d).item()
                metric_results[f'{key}_2d_3d'] = metric(z2d, z3d).item()
        return metric_results

    def evaluate_metrics_finetune(self, z2d, z3d, batch=None, val=False) -> Dict[str, float]:
        metric_results = {}
        metric_results[f'mean_pred'] = torch.mean(z2d).item()
        metric_results[f'std_pred'] = torch.std(z2d).item()
        metric_results[f'mean_targets'] = torch.mean(z3d).item()
        metric_results[f'std_targets'] = torch.std(z3d).item()
        if 'Local' in type(self.loss_func).__name__ and batch != None:
            node_indices = torch.cumsum(batch[0].batch_num_nodes(), dim=0)
            pos_mask = torch.zeros((len(z2d), len(z3d)), device=z2d.device)
            for graph_idx in range(1, len(node_indices)):
                pos_mask[node_indices[graph_idx - 1]: node_indices[graph_idx], graph_idx] = 1.
            pos_mask[0:node_indices[0], 0] = 1
            for key, metric in self.metrics.items():
                if not hasattr(metric, 'val_only') or val:
                    metric_results[key] = metric(z2d, z3d, pos_mask).item()
        else:
            for key, metric in self.metrics.items():
                if not hasattr(metric, 'val_only') or val:
                    metric_results[key] = metric(z2d, z3d).item()
        return metric_results


    def run_per_epoch_evaluations(self, data_loader):
        print('fitting linear probe')
        representations = []
        targets = []
        for batch in data_loader:
            batch = [element.to(self.device) for element in batch]
            loss, view1d, view2d, view3d = self.process_batch(batch, optim=None)
            # loss, view2d, view3d = self.process_batch(batch, optim=None)
            representations.append(view1d)
            representations.append(view2d)
            targets.append(batch[-1])
            if len(representations) * len(view1d) >= self.args.linear_probing_samples:
                break
            if len(representations) * len(view2d) >= self.args.linear_probing_samples:
                break
        representations = torch.cat(representations, dim=0)
        targets = torch.cat(targets, dim=0)
        if len(representations) >= representations.shape[-1]:
            X, _ = torch.lstsq(targets, representations)
            X, _ = torch.lstsq(targets, representations)
            X, _ = torch.lstsq(targets, representations)
            sol = X[:representations.shape[-1]]
            pred = representations @ sol
            mean_absolute_error = (pred - targets).abs().mean()
            self.writer.add_scalar('linear_probe_mae', mean_absolute_error.item(), self.optim_steps)
        else:
            raise ValueError(
                f'We have less linear_probing_samples {len(representations)} than the metric dimension {representations.shape[-1]}. Linear probing cannot be used.')

        print('finish fitting linear probe')

    def initialize_optimizer(self, optim):
        normal_params = [v for k, v in chain(self.model2d.named_parameters(), self.model3d.named_parameters()) if
                         not 'batch_norm' in k]

        batch_norm_params = [v for k, v in chain(self.model2d.named_parameters(), self.model3d.named_parameters()) if
                             'batch_norm' in k]

        if self.model1d is not None:
            normal_params += [v for k, v in self.model1d.named_parameters() if 'batch_norm' not in k]
            batch_norm_params += [v for k, v in self.model1d.named_parameters() if 'batch_norm' in k]
        # # 如果 1D SMILES-BERT 也要训练，加入它的参数
        # if self.smiles_model is not None:
        #     normal_params += list(self.smiles_model.parameters())

        self.optim = optim([{'params': batch_norm_params, 'weight_decay': 0},
                            {'params': normal_params}],
                           **self.args.optimizer_params)

    def save_model_state(self, epoch: int, checkpoint_name: str):
        # torch.save({
        #     'epoch': epoch,
        #     'best_val_score': self.best_val_score,
        #     'optim_steps': self.optim_steps,
        #     'model_state_dict': self.model.state_dict(),
        #     'model3d_state_dict': self.model3d.state_dict(),
        #     'optimizer_state_dict': self.optim.state_dict(),
        #     'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        # }, os.path.join(self.writer.log_dir, checkpoint_name))

        checkpoint = {
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model1d_state_dict': self.model1d.state_dict(),
            'model2d_state_dict': self.model2d.state_dict(),
            'model3d_state_dict': self.model3d.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(self.writer.log_dir, checkpoint_name))

        # def visualize_embeddings(view1d_all, view2d_all, view3d_all, save_dir, prefix="tsne", device="cpu"):
        #     """
        #     传入3个模态的嵌入 (torch.tensor)，执行t-SNE并保存图像
        #     """
        #     print("[t-SNE] Preparing data for visualization...")
        #
        #     # 将数据堆叠（假设形状为 [N, 256]）
        #     all_embeds = torch.cat([view1d_all, view2d_all, view3d_all], dim=0).to(device)
        #     labels = (
        #             ["1D SMILES"] * len(view1d_all) +
        #             ["2D Graph"] * len(view2d_all) +
        #             ["3D Geometry"] * len(view3d_all)
        #     )
        #
        #     # 降维
        #     tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', random_state=42)
        #     reduced = tsne.fit_transform(all_embeds.cpu().detach().numpy())
        #
        #     # 绘图
        #     plt.figure(figsize=(8, 6))
        #     colors = {'1D SMILES': 'r', '2D Graph': 'g', '3D Geometry': 'b'}
        #     for label in set(labels):
        #         idxs = [i for i, l in enumerate(labels) if l == label]
        #         plt.scatter(reduced[idxs, 0], reduced[idxs, 1], c=colors[label], label=label, alpha=0.6)
        #
        #     plt.title('t-SNE of Modal Embeddings')
        #     plt.legend()
        #     plt.grid(True)
        #
        #     os.makedirs(save_dir, exist_ok=True)
        #     path = os.path.join(save_dir, f"{prefix}_tsne.png")
        #     plt.savefig(path)
        #     plt.close()
        #     print(f"[t-SNE] Saved visualization to {path}")
        #
