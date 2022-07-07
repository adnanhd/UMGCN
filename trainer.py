from torchutils.trainer.handler import TrainingHandler
from torchutils.trainer import Trainer
from torchutils.models import TrainerModel
from torchutils.utils.pydantic.types import ModuleType, NpTorchType
from torchutils.metrics import AverageMeter
from torch_geometric.data.batch import Batch
from torch.nn.functional import cross_entropy
from typing import Callable, Optional
from torch_geometric.data import DataLoader, Dataset
from pydantic import PrivateAttr

# @TODO: TrainerModel must have two forward passes -> train and eval where
# train pushes for _push_for_backward where the second one doesn't do this


def init_ce_loss():
    return AverageMeter("Cross Entropy")


def init_um_loss():
    return AverageMeter("Uncertainty")


def init_ls_loss():
    return AverageMeter("Label Smoothing")


class UMGCN(TrainerModel):
    surrogate: ModuleType
    label_smoothing: Callable[[NpTorchType, NpTorchType], NpTorchType]
    l_ls: float = 1e-3  # 0.001
    l_um: float = 3e-1  # 0.3
    _loss_ce: AverageMeter = PrivateAttr(default_factory=init_ce_loss)
    _loss_um: AverageMeter = PrivateAttr(default_factory=init_um_loss)
    _loss_ls: AverageMeter = PrivateAttr(default_factory=init_ls_loss)

    def forward_pass(self, batch: Batch, batch_idx: int):
        targets = batch.y[batch.train_mask]

        gcn_output = self.model(batch.x, batch.edge_index, batch.edge_weight)
        gcn_output = gcn_output[batch.train_mask]
        fc_output = self.surrogate(batch.x[batch.train_mask])

        L_ce = cross_entropy(gcn_output, targets)
        L_ls = self.label_smoothing(fc_output, targets)

        L_um = self.criterion(gcn_output, fc_output)
        loss = L_ce + self.l_um * L_um + self.l_ls + L_ls

        self._loss_ce.update(L_ce.item())
        self._loss_um.update(self.l_um * L_um.item())
        self._loss_ls.update(self.l_ls * L_ls.item())

        self._loss.update(loss.item())
        self._push_for_backward(loss)
        return gcn_output


# @TODO: Batch dataset in the Trainer not in IterationBatch or IterationHandler
# name the method as create_databatch, similar to create_dataloader, which will
# enough to override in order to use it in a generic method


class UMGCNTrainer(Trainer):
    def _run_training_step(self,
                           batch_idx: int,
                           batch: Batch,
                           handler: TrainingHandler):
        handler.on_training_step_begin(batch_idx)

        y_pred = self._model.forward_pass(batch, batch_idx)
        self._model.backward_pass()

        handler.on_training_step_end(x=batch.x[batch.train_mask],
                                     y=batch.y[batch.train_mask],
                                     y_pred=y_pred)

        return y_pred.detach()

    def _run_evaluating_step(self,
                             batch_idx: int,
                             batch: Batch,
                             handler: TrainingHandler):
        handler.on_validation_step_begin(batch_idx)

        y_pred = self._model.forward_pass(batch, batch_idx)
        self._model.backward_pass()

        handler.on_validation_step_end(x=batch.x[batch.val_mask],
                                       y=batch.y[batch.val_mask],
                                       y_pred=y_pred)

        return y_pred.detach()

    def create_dataloader(
        self,
        dataset: Dataset,
        train_mode: bool = True,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> DataLoader:
        assert isinstance(dataset, Dataset), \
            f"Dataset must be inherited from {Dataset}"

        kwargs.setdefault('shuffle', train_mode)
        if not train_mode or batch_size is None:
            kwargs['batch_size'] = dataset.__len__()
        else:
            kwargs['batch_size'] = batch_size

        return DataLoader(dataset, **kwargs)
