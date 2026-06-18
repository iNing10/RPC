"""
Train/Test helper, based on awesome previous work by https://github.com/amirmk89/gepc
"""

import os
import time
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm
import logging
import json
from datetime import datetime


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def compute_loss(nll, reduction="mean", mean=0):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "logsumexp":
        losses = {"nll": torch.logsumexp(nll, dim=0)}
    elif reduction == "exp":
        losses = {"nll": torch.exp(torch.mean(nll) - mean)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


class Trainer:
    def __init__(self, args, model, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Best model tracking for epoch-wise evaluation
        self.best_auc = -1.0
        self.best_epoch = 0
        
        # Setup logging
        self.setup_logging()
        
        # Loss, Optimizer and Scheduler
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.args.optimizer == 'adamx':
            if self.args.lr:
                return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def setup_logging(self):
        """Setup logging to save training progress to file."""
        # Create logs directory
        log_dir = os.path.join(self.args.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%b%d_%H%M")
        log_filename = f"training_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_path = log_path
        
        # Initialize training metrics log
        self.training_metrics = {
            'start_time': datetime.now().isoformat(),
            'args': vars(self.args),
            'epochs': []
        }
        
        mode = "Evaluation" if getattr(self.args, "checkpoint", None) else "Training"
        self.logger.info("="*60)
        self.logger.info(f"{mode} Started")
        self.logger.info(f"Experiment: {self.args.exp_dir}")
        self.logger.info(f"Dataset: {self.args.dataset}")
        self.logger.info("="*60)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.model_lr, self.args.model_lr_decay, self.scheduler)
    
    def evaluate_epoch(self, dataset, epoch):
        """
        Evaluate model on test set and compute AUC.
        Returns: (normality_scores, auc_score)
        """
        # Import here to avoid circular imports
        from utils.scoring_utils import score_dataset
        
        # Get test predictions
        normality_scores = self.test()
        
        # Compute AUC
        auc, scores = score_dataset(normality_scores, dataset["test"].metadata, args=self.args)
        
        # Log AUC result
        auc_msg = f"Epoch {epoch+1}: AUC = {auc*100:.2f}%"
        print(auc_msg)
        self.logger.info(auc_msg)
        
        return normality_scores, auc

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = self.args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            # Map checkpoint tensors to the current device (CPU-safe)
            map_loc = None
            try:
                dev_str = getattr(self.args, 'device', 'cpu')
            except Exception:
                dev_str = 'cpu'
            if isinstance(dev_str, str) and dev_str.startswith('cuda'):
                map_loc = dev_str if torch.cuda.is_available() else 'cpu'
            else:
                map_loc = 'cpu'

            checkpoint = torch.load(filename, map_location=map_loc, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.set_actnorm_init()
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not load optimizer state: {e}")
                print("Continuing without optimizer state (OK for testing)")
            print(f"Checkpoint loaded: {filename}\n")
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, log_writer=None, clip=100, dataset=None):
        time_str = time.strftime("%b%d_%H%M_")
        checkpoint_filename = time_str + '_checkpoint.pth.tar'
        best_checkpoint_filename = time_str + '_best_checkpoint.pth.tar'
        start_epoch = 0
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to(self.args.device)
        key_break = False
        
        for epoch in range(start_epoch, num_epochs):
            if key_break:
                break
            epoch_start_time = time.time()
            epoch_msg = "Starting Epoch {} / {}".format(epoch + 1, num_epochs)
            print(epoch_msg)
            self.logger.info(epoch_msg)
            
            # Training phase
            self.model.train()
            pbar = tqdm(self.train_loader, disable=getattr(self.args, 'verbose', 1) == 0)
            train_sums = {
                'nll_loss': 0.0,
                'total_loss': 0.0,
                'num_batches': 0,
            }
            for itern, data_arr in enumerate(pbar):
                try:
                    data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
                    full_pose = data[0].float()
                    score_seq = data[-2].float()
                    label = data[-1]
                    
                    # Apply label smoothing if enabled
                    if hasattr(self.args, 'label_smoothing') and self.args.label_smoothing > 0:
                        smoothing = self.args.label_smoothing
                        label = label * (1 - smoothing) + smoothing / 2
                    
                    if not self.args.model_confidence:
                        flow_input = full_pose[:, :2]
                    else:
                        flow_input = full_pose
                    _z, nll = self.model(x=flow_input, label=label)
                    if nll is None:
                        continue
                    if self.args.model_confidence:
                        nll = nll * score_seq.amin(dim=-1)
                    
                    nll_loss = compute_loss(nll, reduction="mean")["total_loss"]
                    losses = nll_loss
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    train_sums['nll_loss'] += float(nll_loss.detach().cpu())
                    train_sums['total_loss'] += float(losses.detach().cpu())
                    train_sums['num_batches'] += 1
                    pbar.set_description("Loss: {}".format(losses.item()))
                    if log_writer is not None:
                        global_step = epoch * len(self.train_loader) + itern
                        log_writer.add_scalar('NLL Loss', nll_loss.item(), global_step)

                except KeyboardInterrupt:
                    print('Keyboard Interrupted. Save results? [yes/no]')
                    choice = input().lower()
                    if choice == "yes":
                        key_break = True
                        break
                    else:
                        exit(1)

            # Save regular checkpoint
            epoch_train_metrics = self.average_train_metrics(train_sums)
            self.save_checkpoint(epoch, filename=checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            lr_msg = 'Checkpoint Saved. New LR: {0:.3e}'.format(new_lr)
            print(lr_msg)
            self.logger.info(lr_msg)
            
            # Evaluate every epoch if enabled
            if hasattr(self.args, 'eval_every_epoch') and self.args.eval_every_epoch and dataset is not None:
                eval_msg = f"Evaluating epoch {epoch+1}..."
                print(eval_msg)
                self.logger.info(eval_msg)
                self.model.eval()
                _, current_auc = self.evaluate_epoch(dataset, epoch)
                
                # Log epoch metrics
                epoch_time = time.time() - epoch_start_time
                epoch_metrics = {
                    'epoch': epoch + 1,
                    'auc': current_auc,
                    'is_best': current_auc > self.best_auc,
                    'training_time': epoch_time,
                    'lr': new_lr,
                    **epoch_train_metrics,
                }
                self.training_metrics['epochs'].append(epoch_metrics)
                
                # Check if this is the best model so far
                if current_auc > self.best_auc:
                    self.best_auc = current_auc
                    self.best_epoch = epoch + 1
                    best_msg = f"New best AUC: {current_auc*100:.2f}% (Epoch {self.best_epoch})"
                    print(best_msg)
                    self.logger.info(best_msg)
                    
                    # Save best model
                    self.save_checkpoint(epoch, is_best=True, filename=best_checkpoint_filename)
                    save_msg = f"Best model saved: {best_checkpoint_filename}"
                    print(save_msg)
                    self.logger.info(save_msg)
                else:
                    current_msg = f"Current AUC: {current_auc*100:.2f}%, Best AUC: {self.best_auc*100:.2f}% (Epoch {self.best_epoch})"
                    print(current_msg)
                    self.logger.info(current_msg)
                
                # Switch back to training mode
                self.model.train()
        
        # Print final best results if epoch evaluation was used
        if hasattr(self.args, 'eval_every_epoch') and self.args.eval_every_epoch:
            final_msg = f"Best model: Epoch {self.best_epoch} with AUC = {self.best_auc*100:.2f}%"
            model_msg = f"Best model saved as: {best_checkpoint_filename}"
            print(f"\n{final_msg}")
            print(model_msg)
            self.logger.info("="*60)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info(final_msg)
            self.logger.info(model_msg)
            self.logger.info("="*60)
            
            # Save training metrics to JSON file
            self.save_training_metrics()

    def save_training_metrics(self):
        """Save detailed training metrics to JSON file."""
        self.training_metrics['end_time'] = datetime.now().isoformat()
        self.training_metrics['best_auc'] = self.best_auc
        self.training_metrics['best_epoch'] = self.best_epoch
        
        # Create metrics filename
        timestamp = datetime.now().strftime("%b%d_%H%M")
        metrics_filename = f"training_metrics_{timestamp}.json"
        metrics_path = os.path.join(self.args.exp_dir, 'logs', metrics_filename)
        
        # Save to JSON
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        self.logger.info(f"Training metrics saved to: {metrics_path}")

    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.test_loader, disable=getattr(self.args, 'verbose', 1) == 0)
        probs = torch.empty(0).to(self.args.device)
        print("Running evaluation...")
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            full_pose = data[0].float()
            score_seq = data[-2].float()
            if not self.args.model_confidence:
                flow_input = full_pose[:, :2]
            else:
                flow_input = full_pose
            with torch.no_grad():
                # Fix device and dtype compatibility for TTA
                label = torch.ones(full_pose.shape[0], dtype=torch.float32, device=self.args.device)
                _z, nll = self.model(x=flow_input, label=label)
            if self.args.model_confidence:
                nll = nll * score_seq.amin(dim=-1)

            normality = -1 * nll

            probs = torch.cat((probs, normality), dim=0)
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        return {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def average_train_metrics(self, train_sums):
        num_batches = max(1, train_sums['num_batches'])
        return {
            'train_nll_loss': train_sums['nll_loss'] / num_batches,
            'train_total_loss': train_sums['total_loss'] / num_batches,
        }
