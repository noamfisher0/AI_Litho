import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader

from Dataloaders.load_utils import load_dataset
import wandb
import time

import matplotlib.pyplot as plt

# Define GeneralModel_pl in pytorch_lightning framework.

class GeneralModel_pl(pl.LightningModule):
    def __init__(self,
                in_dim,
                out_dim,
                lr = 0.0001,
                batch_size = 20,
                weight_decay = 1e-6,
                scheduler_step = 15,
                scheduler_gamma = 0.98,
                config = dict(),

                wandb_aggregation = 20 # After this many gradient steps, the metrics are logged
                ):
        super(GeneralModel_pl, self).__init__()

        self.model = None # TO BE DEFINED IN THE CHILD CLASS !!!

        '''
            -- For example, in the child class, there must me smth like this:
            self.model = FNO2d(in_dim = in_dim,
                             out_dim = out_dim,
                             n_layers = n_layers,
                             width = width,
                             modes = modes,
                             hidden_dim = hidden_dim,
                             use_conv = use_conv,
                             conv_filters = conv_filters,
                             padding = padding,
                             include_grid = include_grid,
                             is_time = is_time)
        '''

        self.in_dim = in_dim
        self.out_dim = out_dim

        #--------------------
        # Training parameters
        #--------------------

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.scheduler_step = scheduler_step
        self.scheduler_gamma = scheduler_gamma
        self.config = config # IMPORTANT -- Experiment and Training parameters

        self.best_val_loss = 1000.0
        self.validation_errs  = dict()
        self.validation_times = dict()

        """
          - If we traing the model to predict different physical quantities (velocity + pressure + ...)
          - For example, if the variables are [rho, vx, vy, p], then "separate_dim" should be [1,2,1]
          - 2 means that vx and vy are grouped together!
        """
        # Are the physical quantities separated in the loss function?
        if  ("separate" in self.config) and self.config["separate"]:
            self._is_separate = True
            if "separate_dim" in self.config:
                self._separate_dim = self.config["separate_dim"]
            else:
                self._separate_dim = [out_dim]
        else:
            self._is_separate = False


        self._which_benchmark = self.config["which_example"]

        # Are we interested in all the channels or we want to predict just a few of them and ignore others?
        self._is_masked = "is_masked" in self.config and self.config["is_masked"] is not None

        # Is there a spatial mask, like in the airfoil benchmark?
        self._spatial_mask = "spatial_mask" in self.config and self.config["spatial_mask"] is not None and self.config["spatial_mask"]
        self._spatial_mask = self._spatial_mask or self._which_benchmark == "airfoil"

        # Wandb logs:
        self._wandb_aggregation = wandb_aggregation
        self._curr_epoch = -1
        self._cur_step   = 0

    def forward(self, x, time):
        return self.model(x, time)

    def _get_separation(self):

         # How are the variables separated?
        _diff = [0, self._separate_dim[0]]
        for i in range(1,len(self._separate_dim)):
            _diff.append(_diff[-1]+self._separate_dim[i])
        _num_separate = len(_diff)-1
        return _diff, _num_separate

    def _mask_output(self,
                     masked_dim,
                     num_separate,
                     diff,
                     output_batch,
                     output_pred_batch):

        for i in range(num_separate):
            mask = masked_dim[:,diff[i]:diff[i+1]]
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], output_batch.shape[-2], output_batch.shape[-1])
            output_pred_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0
            output_batch[:,diff[i]:diff[i+1]][mask==0.0] = 1.0
        return output_batch, output_pred_batch

    def training_step(self, batch, batch_idx):

        if batch_idx == 0:
            self._curr_epoch+=1

        if not self._is_masked:
            t_batch, input_batch, output_batch = batch
        else:
            # Relevant dim tells us what channels we need to care about (it's a mask)
            t_batch, input_batch, output_batch, masked_dim = batch

        # Predict:
        output_pred_batch = self(input_batch, t_batch)

        # If spatial mask, as in airfoil, mask it
        if self._spatial_mask:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0

        # Compute relative L1 loss
        if not self._is_separate:
            loss = nn.L1Loss()(output_batch, output_pred_batch) / nn.L1Loss()(torch.zeros_like(output_batch), output_batch)
        else:
            _diff, _num_separate = self._get_separation()
            weight = 1.0/_num_separate
            if self._is_masked:
                output_batch, output_pred_batch= self._mask_output(masked_dim,
                                                                 _num_separate,
                                                                 _diff,
                                                                 output_batch,
                                                                 output_pred_batch)
            loss = 0.0

            # Compute the loss over each block in 'separated' output
            for i in range(_num_separate):
                loss = loss + weight * nn.L1Loss()(output_pred_batch[:,_diff[i]:_diff[i+1]], output_batch[:,_diff[i]:_diff[i+1]])/ (nn.L1Loss()(output_batch[:,_diff[i]:_diff[i+1]],torch.zeros_like(output_batch[:,_diff[i]:_diff[i+1]])) + 1e-10)

        if self._cur_step%self._wandb_aggregation == 0:
            self.log('loss', loss, prog_bar=True)
            wandb.log({'train/loss': loss.item(), 'train/step': self._cur_step, 'train/epoch': self._curr_epoch}, step=self._cur_step)
        self._cur_step+=1

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    def train_dataloader(self):

        if self._which_benchmark == "eul_ns_mix1":
            raise Exception("Not possible to run this dataset at the moment")

        else:
            if self._is_masked:
                if self._which_benchmark[:2] == "ns":
                    mask = [1.0, 1.0, 1.0, 0.0]
                elif self._which_benchmark[:3] == "eul":
                    mask = [1.0, 1.0, 1.0, 1.0]
                else:
                    mask = [1.0, 1.0, 1.0, 1.0]
            else:
                mask = None

            train_dataset = load_dataset(config = self.config,
                                          which = self._which_benchmark,
                                          which_loader = "train",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = mask)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return train_loader

    def validation_step(self, batch, batch_idx):

        if not self._is_masked:
            t_batch, input_batch, output_batch = batch
        else:
            # Relevant dim tells us what channels we need to care about (it's a mask)
            t_batch, input_batch, output_batch, masked_dim = batch

        # Predict:
        output_pred_batch = self(input_batch, t_batch)

        # If spatial mask, as in airfoil, mask it
        if self._spatial_mask:
            output_pred_batch[input_batch==1] = 1.0
            output_batch[input_batch==1] = 1.0

        # Compute relative L1 loss -- concat means over the blocks if is_separate
        if not self._is_separate:
            loss = (torch.mean(abs(output_pred_batch- output_batch), (-3, -2, -1)) / (torch.mean(abs(output_batch), (-3, -2, -1))+ 1e-10))* 100

        else:
            _diff, _num_separate = self._get_separation()
            weight = 1.0 / _num_separate
            if self._is_masked:
                output_batch, output_pred_batch= self._mask_output(masked_dim,
                                                                 _num_separate,
                                                                 _diff,
                                                                 output_batch,
                                                                 output_pred_batch)

            for i in range(_num_separate):
                _loss = (torch.mean(abs(output_pred_batch[:,_diff[i]:_diff[i+1]] - output_batch[:,_diff[i]:_diff[i+1]]), (-3, -2, -1)) / (torch.mean(abs(output_batch[:,_diff[i]:_diff[i+1]]), (-3, -2, -1))+ 1e-10))* 100
                if i == 0:
                    loss = weight * _loss
                else:
                    loss = loss + weight *_loss

        # Save validation errs:
        if batch_idx==0:
            self.validation_times = t_batch
            self.validation_errs = loss
        else:
            self.validation_times = torch.cat((self.validation_times, t_batch))
            self.validation_errs = torch.cat((self.validation_errs, loss))

        return loss

    def val_dataloader(self):

        if self._which_benchmark == "eul_ns_mix1":
            raise Exception("Not possible to run this dataset at the moment")
        else:
            if self._is_masked:
                if self._which_benchmark[:2] == "ns":
                    mask = [1.0, 1.0, 1.0, 0.0]
                elif self._which_benchmark[:3] == "eul":
                    mask = [1.0, 1.0, 1.0, 1.0]
                else:
                    mask = [1.0, 1.0, 1.0, 1.0]
            else:
                mask = None

            val_dataset  =  load_dataset(config = self.config,
                                          which = self._which_benchmark,
                                          which_loader = "val",
                                          in_dim = self.in_dim,
                                          out_dim = self.out_dim,
                                          masked_input = mask)
            val_loaders = [DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)]

        return val_loaders

    def on_validation_epoch_end(self):

        _max_time = torch.max(self.validation_times)
        _validation_errs_last = self.validation_errs[self.validation_times==_max_time]

        median_loss = torch.median(self.validation_errs).item()
        mean_loss = torch.mean(self.validation_errs).item()
        median_loss_last = torch.median(_validation_errs_last).item()
        mean_loss_last = torch.mean(_validation_errs_last).item()

        self.log("med_val_loss", median_loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)
        self.log("mean_val_loss",  mean_loss, prog_bar=True, on_step=False, on_epoch=True,sync_dist=True)

        # Save the best loss
        if mean_loss < self.best_val_loss:
            self.best_val_loss = mean_loss

        self.log("best_val_loss",self.best_val_loss,on_step=False, on_epoch=True,sync_dist=True)

        wandb.log({'val/best_val_loss': self.best_val_loss, 'val/mean_val_all': mean_loss, 'val/med_val_all': median_loss, 'val/med_val_last':median_loss_last, 'val/mean_val_last':mean_loss_last}, step=self._cur_step)

        return {"mean_val_loss": mean_loss,}
