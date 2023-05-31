import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Adam
from torch.nn import MSELoss
from torch import nn
from torch.cuda.amp import GradScaler, autocast

try:
    # Ros runtime
    from .nets import create_network
    from .data_loader import create_dataset
    from .utils import MixtureSpaceLoss, TrajectoryCostLoss
except:
    # Training time
    from nets import create_network
    from data_loader import create_dataset
    from utils import MixtureSpaceLoss, TrajectoryCostLoss, convert_to_trajectory, \
        save_trajectories, transformToWorldFrame



class PlanLearner(object):
    def __init__(self, settings):
        self.data_interface = None
        self.config = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.min_val_loss = np.inf
        self.network = create_network(self.config).to(self.device)
        self.space_loss = MixtureSpaceLoss(T=self.config.out_seq_len * 0.1, modes=self.config.modes)
        self.cost_loss = TrajectoryCostLoss(ref_frame=self.config.ref_frame, state_dim=self.config.state_dim)
        self.cost_loss_v = TrajectoryCostLoss(ref_frame=self.config.ref_frame, state_dim=self.config.state_dim)

        self.optimizer = Adam(self.network.parameters())
        self.learning_rate_fn = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50000,
            T_mult=2,
            eta_min=0.01)

        

        self.train_space_loss = MSELoss()
        self.val_space_loss = MSELoss()
        self.train_cost_loss = MSELoss()
        self.val_cost_loss = MSELoss()

        self.global_epoch = 0

        self.scaler = GradScaler()

        if self.config.resume_training:
            if os.path.exists(self.config.resume_ckpt_file):
                checkpoint = torch.load(self.config.resume_ckpt_file)
                self.network.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.global_epoch = checkpoint['epoch']
                self.min_val_loss = checkpoint['loss']
                print("------------------------------------------")
                print("Restored from {}".format(self.config.resume_ckpt_file))
                print("------------------------------------------")
                return

        print("------------------------------------------")
        print("Initializing from scratch.")
        print("------------------------------------------")

    def train_step(self, inputs, labels):
        self.network.train()
        self.optimizer.zero_grad()
        with autocast():
            predictions = self.network(inputs)
            space_loss = self.space_loss(labels, predictions)
            cost_loss = self.cost_loss((inputs['roll_id'], inputs['imu'][:, -1, :12]), predictions)
            loss = space_loss + cost_loss
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def val_step(self, inputs, labels):
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(inputs)
            space_loss = self.space_loss(labels, predictions)
            cost_loss = self.cost_loss_v((inputs['roll_id'], inputs['imu'][:, -1, :12]), predictions)
            val_loss = space_loss + cost_loss
        return val_loss.item()

    def adapt_input_data(self, features):
        if self.config.use_rgb and self.config.use_depth:
            inputs = {"rgb": features[1][0],
                      "depth": features[1][1],
                      "roll_id": features[2],
                      "imu": features[0]}
        elif self.config.use_rgb and (not self.config.use_depth):
            inputs = {"rgb": features[1],
                      "roll_id": features[2],
                      "imu": features[0]}
        elif self.config.use_depth and (not self.config.use_rgb):
            inputs = {"depth": features[1],
                      "roll_id": features[2],
                      "imu": features[0]}
        else:
            inputs = {"imu": features[0],
                      "roll_id": features[1]}
        return inputs

    def write_train_summaries(self, epoch, loss):
        writer = SummaryWriter()
        writer.add_scalar('Loss/train', loss, epoch)

    def train(self):
        print("Training Network")
        dataset_train, dataloader_train = create_dataset(self.config.train_dir,
                                       self.config, training=True)
        dataset_val, dataloader_val = create_dataset(self.config.val_dir,
                                     self.config, training=False)

        self.cost_loss.add_pointclouds(dataset_train.pointclouds)
        self.cost_loss_v.add_pointclouds(dataset_val.pointclouds)

        for epoch in range(self.config.max_training_epochs):
            self.global_epoch += 1
            # Train
            total_loss = 0
            for k, (features, label, _) in enumerate(tqdm(dataloader_train)):
                features = self.adapt_input_data(features)
                label = label.to(self.device)
                loss = self.train_step(features, label)
                total_loss += loss

            self.write_train_summaries(epoch, total_loss / len(dataloader_train))
            # Eval
            total_val_loss = 0
            for k, (features, label, _) in enumerate(tqdm(dataloader_val)):
                features = self.adapt_input_data(features)
                label = label.to(self.device)
                val_loss = self.val_step(features, label)
                total_val_loss += val_loss

            validation_loss = total_val_loss / len(dataloader_val)

            print("Epoch: {:2d}, Val Space Loss: {:.4f}, Val Cost Loss: {:.4f}".format(
                epoch, validation_loss, validation_loss))

            
            if validation_loss < self.min_val_loss or ((self.global_epoch + 1) % self.config.save_every_n_epochs) == 0:
                if validation_loss < self.min_val_loss:
                    self.min_val_loss = validation_loss
                if validation_loss < 10.0:  # otherwise training diverged
                    torch.save({
                        'epoch': self.global_epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': validation_loss,
                    }, os.path.join(self.config.log_dir, f'checkpoint_{epoch}.pt'))

        print("------------------------------")
        print("Training finished successfully")
        print("------------------------------")

    def test(self):
        print("Testing Network")
        dataset_val = create_dataset(self.config.test_dir,
                                     self.config, training=False)
        self.cost_loss_v.add_pointclouds(dataset_val.pointclouds)
        total_val_loss = 0
        for k, (features, label, _) in enumerate(tqdm(dataset_val.batched_dataset)):
            features = self.adapt_input_data(features)
            val_loss = self.val_step(features, label)
            total_val_loss += val_loss

        print("Testing Space Loss: {:.4f} Testing Cost Loss: {:.4f}".format(total_val_loss, total_val_loss))

    
    def inference(self, inputs):
        # run time inference
        with torch.no_grad():
            processed_pred = self.full_post_inference(inputs).cpu().numpy()
        # Assume BS = 1
        processed_pred = processed_pred[:, np.abs(processed_pred[0, :, 0]).argsort(), :]
        alphas = np.abs(processed_pred[0, :, 0])
        predictions = processed_pred[0, :, 1:]
        return alphas, predictions

    def full_post_inference(self, inputs):
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(inputs)
        return predictions
