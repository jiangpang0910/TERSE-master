import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import Classifier, Regressor, Temporal_Imputer
from models.models import masking2, mask_adj_matrices_edges, GraphRecover_new
from models.loss import CrossEntropyLabelSmooth, EntropyLoss, RegressorLoss
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError


class TERSE(Algorithm):
    """
    TERSE: Our proposed method using temporal restoration and spatial rewiring adaptation.
    """
    def __init__(self, backbone, configs, hparams, device):
        super(TERSE, self).__init__(configs)
        # backbone:
        self.feature_extractor = backbone(configs)
        # classifier:
               # Use Regressor or Classifier based on task
        if getattr(configs, 'task', 'classification') == 'regression':
            self.classifier = Regressor(configs)
        else:
            self.classifier = Classifier(configs)
        # entire network:
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # temporal imputation.
        self.temporal_verifier = Temporal_Imputer(configs)
        # graph recover:
        self.graph_recover = GraphRecover_new(configs)

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr = hparams['pre_learning_rate'],
            weight_decay=hparams['weight_decay']
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr = hparams['learning_rate'],
            weight_decay=hparams['weight_decay']
        )

        self.recover_optimizer = torch.optim.Adam([
            {'params': self.temporal_verifier.parameters(), 'lr': hparams['learning_rate'], 'weight_decay': hparams['weight_decay']},
            {'params': self.graph_recover.parameters()}
        ])

        self.hparams = hparams
        self.device = device
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1)
        self.mse_loss = nn.MSELoss()
        self.regression_loss = nn.MSELoss()  # separate MSE for regression task supervised loss

    def pretrain(self, src_dataloader, avg_meter, logger):
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x = src_x.float().to(self.device)
                if getattr(self.configs, 'task', 'classification') == 'regression':
                    src_y = src_y.float().to(self.device)
                else:
                    src_y = src_y.long().to(self.device)

                # optimizer zero_grad:
                self.pre_optimizer.zero_grad()
                self.recover_optimizer.zero_grad()

                ### raw data ###
                src_temp_feat = self.feature_extractor.temporal_cnn(src_x)  # 1. extract temporal features.
                src_adj = self.feature_extractor.graph_learner(src_temp_feat)  # 2. graph structure learning.
                src_feat, src_flat = self.feature_extractor.spatial_gnn(src_temp_feat, src_adj)  # 3. extract spatial features.

                ### masked data ###
                masked_src_x, _ = masking2(src_x, num_splits=8, num_masked=1)  # 4. masking the raw features.
                masked_feat, masked_flat = self.feature_extractor(masked_src_x) # 5. extract masked features.
                masked_adj = mask_adj_matrices_edges(src_adj, mask_ratio=self.hparams['gmask_ratio'])  # 7. masking edges.

                ### 1. masked raw features recover loss. ###
                src_recovered_temp_feat = self.temporal_verifier(masked_feat.detach())
                tov_loss = self.mse_loss(src_recovered_temp_feat, src_feat)

                ### 3. masked graph recover loss. ###
                src_masked_adj_feat, _ = self.feature_extractor.spatial_gnn(src_temp_feat, masked_adj)
                src_recovered_graph = self.graph_recover(src_masked_adj_feat.detach(), masked_adj.detach())  # use clearn feats and masked graphs
                graph_recover_loss = self.mse_loss(src_recovered_graph, src_adj)

                # 7. classifier predictions
                src_pred = self.classifier(src_flat)

                # 8. normal cross entropy
                # 8. supervised loss
                if getattr(self.configs, 'task', 'classification') == 'regression':
                    src_cls_loss = self.regression_loss(src_pred, src_y)
                else:
                    src_cls_loss = self.cross_entropy(src_pred, src_y)

                # 9. total loss.
                total_loss = src_cls_loss + tov_loss + graph_recover_loss

                total_loss.backward()
                self.pre_optimizer.step()
                self.recover_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'tov_loss': tov_loss.detach().item(), 'graph_masking_loss': graph_recover_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and temporal restoration and spatial rewiring.
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        for k, v in self.graph_recover.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (trg_x, trg_y, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()

                ### extract features. ###
                trg_temp_feat = self.feature_extractor.temporal_cnn(trg_x) # 1. extract temporal features.
                trg_adj = self.feature_extractor.graph_learner(trg_temp_feat) # 2. graph structure learner.
                trg_feat, trg_flat = self.feature_extractor.spatial_gnn(trg_temp_feat, trg_adj) # 3. extract spatial features.

                # 4. masking the node features and graph structures.
                masked_trg_x, _ = masking2(trg_x, num_splits=8, num_masked=1)
                masked_feat, masked_flat = self.feature_extractor(masked_trg_x)
                recovered_masked_feats = self.temporal_verifier(masked_feat)
                tov_loss = self.mse_loss(recovered_masked_feats, trg_feat)

                masked_adj = mask_adj_matrices_edges(trg_adj, mask_ratio=self.hparams['gmask_ratio'])

                # 6. masked graph recover loss.
                masked_adj_feats, _ = self.feature_extractor.spatial_gnn(trg_temp_feat, masked_adj)
                trg_recovered_graph = self.graph_recover(masked_adj_feats, masked_adj)
                graph_recover_loss = self.mse_loss(trg_recovered_graph, trg_adj)

                # 7. prediction scores
                trg_pred = self.classifier(trg_flat)

                if getattr(self.configs, 'task', 'classification') == 'regression':
                    # Regression: no entropy/IM loss (those require class probabilities)
                    # Only self-supervised adaptation losses
                    loss = self.hparams['tov_wt'] * tov_loss + self.hparams['graph_recover_wt'] * graph_recover_loss
                else:
                    # Classification: entropy minimization + information maximization
                    trg_prob = nn.Softmax(dim=1)(trg_pred)
                    trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))
                    trg_ent -= self.hparams['im'] * torch.sum(
                        -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))
                    loss = trg_ent + self.hparams['tov_wt'] * tov_loss + self.hparams['graph_recover_wt'] * graph_recover_loss

                loss.backward()
                self.optimizer.step()

                if getattr(self.configs, 'task', 'classification') == 'regression':
                    losses = {'tov_loss': tov_loss.detach().item(), 'graph_masking_loss': graph_recover_loss.detach().item()}
                else:
                    losses = {'entropy_loss': trg_ent.detach().item(), 'tov_loss': tov_loss.detach().item(), 'graph_masking_loss': graph_recover_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model
