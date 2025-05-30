
import argparse
import time

import torch.optim as optim
import torch
from geotransformer.engine import EpochBasedTrainer

from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator
from dataset import test_data_loader
import logging

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=['3DMatch', '3DLoMatch', 'val'], help='test benchmark')
    return parser

class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, checkpoint_path=None):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()
        model = self.register_model(model)

        # If a checkpoint is provided, load the model and optimizer state
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path)
        print("Checkpoint keys:", checkpoint.keys())

        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            raise KeyError("Checkpoint does not contain model state dict.")

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            self.logger.warning("Optimizer state dict not found in checkpoint. Continuing without optimizer state.")

        start_epoch = checkpoint.get('epoch', 0)  
        iteration = checkpoint.get('iteration', 0)  

        self.logger.info(f"Resuming from checkpoint at epoch {start_epoch}, iteration {iteration}")

        return start_epoch, iteration

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    cfg = make_cfg()

    # Load checkpoint argument
    checkpoint_path = args.checkpoint if args.checkpoint else None

    # Create trainer, optionally passing in the checkpoint path
    trainer = Trainer(cfg, checkpoint_path=checkpoint_path)

    # Set the starting epoch if checkpoint is loaded
    start_epoch = args.start_epoch
    if checkpoint_path:
        start_epoch = trainer.load_checkpoint(checkpoint_path)
    start_epoch = 0
    trainer.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Resume Training from Checkpoint")
    parser.add_argument('--checkpoint', type=str, required=False, help="Path to checkpoint file")
    parser.add_argument('--start_epoch', type=int, required=True, help="Epoch to resume training from")
    parser.add_argument('--num_epochs', type=int, default=10, help="Total number of epochs for training")

    args = parser.parse_args()
    logging.getLogger('PIL').setLevel(logging.WARNING)

    main()
