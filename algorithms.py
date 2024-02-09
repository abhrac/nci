# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F

import networks


ALGORITHMS = [
    'ERM',
    'NCI'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class NCI(ERM):
    """Non-Commutative Invariance"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(NCI, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.discriminator = networks.Classifier(
            self.featurizer.n_outputs,
            num_domains + 1,
            self.hparams['nonlinear_classifier'])
        
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.src_domains = [1, 2, 3, 4, 5]
        self.tgt_domain = 0

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device

        nll = 0.
        loss_d = 0

        all_x = torch.cat([x for x, y in minibatches])
        all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)

            domain_labels = torch.tensor([self.src_domains[i]] * len(features)).to(device)
            pred_d = self.discriminator(features.detach())
            loss_d += F.cross_entropy(pred_d, domain_labels)
        
        features_tgt = self.featurizer(unlabeled[0])
        tgt_domain_labels = torch.tensor([self.tgt_domain] * len(features_tgt)).to(device)
        pred_d = self.discriminator(features_tgt.detach())
        loss_d += F.cross_entropy(pred_d, tgt_domain_labels)
        
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        adv_domain_labels = torch.tensor([self.tgt_domain] * len(all_features)).to(device)
        adv_pred_d = self.discriminator(all_features)
        loss_adv = F.cross_entropy(adv_pred_d, adv_domain_labels) / len(minibatches)
        nll /= len(minibatches)

        # Compile loss
        loss = nll + loss_adv

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item(),
                'nll': nll.item()}
