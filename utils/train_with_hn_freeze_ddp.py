import torch
import time
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import torch.distributed as dist


def train_with_hn_freeze_ddp(model, hn_freeze, train_loader, val_loader, criterion,
                             optimizer, scheduler, config, device, rank, world_size,
                             experiment_dir=None):
    """
    Train a ViT model with HN-Freeze approach using Distributed Data Parallel.

    Args:
        model: DDP-wrapped ViT model
        hn_freeze: HNFreeze instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration object
        device: Device to train on
        rank: Process rank
        world_size: Total number of processes
        experiment_dir: Directory to save results (only used for rank 0)

    Returns:
        Dictionary with training history and results
    """
    # Create experiment directory only on the main process
    if rank == 0 and experiment_dir is None:
        experiment_dir = os.path.join('experiments',
                                      f'hn_freeze_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(experiment_dir, exist_ok=True)

        # Save configuration
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            config_dict = {k: v for k, v in vars(config).items()
                           if not k.startswith('__') and not callable(v)}
            json.dump(config_dict, f, indent=4)

    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': [],
        'epoch_time': []
    }

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(config.num_epochs):
        # Set train sampler's epoch to ensure different shuffling each epoch
        train_loader.sampler.set_epoch(epoch)

        # Record epoch start time
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Only show progress bar on the main process
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')
        else:
            progress_bar = train_loader

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar on the main process
            if rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    'loss': train_loss / total,
                    'acc': 100. * correct / total,
                    'lr': optimizer.param_groups[0]['lr']
                })

            # Learning rate scheduler step (if batch-based)
            if scheduler is not None and hasattr(scheduler, 'step_every_batch') and scheduler.step_every_batch:
                scheduler.step()

        # Gather training statistics from all processes
        train_loss_tensor = torch.tensor([train_loss], device=device)
        correct_tensor = torch.tensor([correct], device=device)
        total_tensor = torch.tensor([total], device=device)

        # Sum across all processes
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        # Calculate global stats
        global_train_loss = train_loss_tensor.item()
        global_correct = correct_tensor.item()
        global_total = total_tensor.item()

        # Learning rate scheduler step (if epoch-based)
        if scheduler is not None and not (hasattr(scheduler, 'step_every_batch') and scheduler.step_every_batch):
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.logits, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # Gather validation statistics from all processes
        val_loss_tensor = torch.tensor([val_loss], device=device)
        val_correct_tensor = torch.tensor([val_correct], device=device)
        val_total_tensor = torch.tensor([val_total], device=device)

        # Sum across all processes
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)

        # Calculate global stats
        global_val_loss = val_loss_tensor.item()
        global_val_correct = val_correct_tensor.item()
        global_val_total = val_total_tensor.item()

        global_val_acc = 100.0 * global_val_correct / global_val_total
        avg_train_loss = global_train_loss / global_total
        avg_val_loss = global_val_loss / global_val_total

        # Calculate epoch duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Update history on the main process
        if rank == 0:
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(global_val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['epoch_time'].append(epoch_duration)

            print(f'Epoch {epoch + 1}/{config.num_epochs}: '
                  f'Train Loss: {avg_train_loss:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f} | '
                  f'Val Acc: {global_val_acc:.2f}% | '
                  f'Time: {epoch_duration:.2f}s')

            # Save best model
            if global_val_acc > best_acc:
                best_acc = global_val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # Save unwrapped model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'frozen_layers': hn_freeze.frozen_layers
                }, os.path.join(experiment_dir, 'best_model.pth'))

                print(f'Best model saved with accuracy: {best_acc:.2f}%')

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # Save unwrapped model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': global_val_acc,
                    'frozen_layers': hn_freeze.frozen_layers
                }, os.path.join(experiment_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

            # Save history
            with open(os.path.join(experiment_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=4)

    # Wait for all processes to finish
    dist.barrier()

    end_time = time.time()
    training_time = end_time - start_time

    # Final results (only on the main process)
    if rank == 0:
        # Calculate average epoch time
        avg_epoch_time = np.mean(history['epoch_time'])

        results = {
            'best_accuracy': best_acc,
            'final_accuracy': global_val_acc,
            'training_time': training_time,
            'average_epoch_time': avg_epoch_time,
            'history': history,
            'frozen_layers': hn_freeze.frozen_layers
        }

        # Save results
        with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        print(f'Training completed in {training_time:.2f} seconds')
        print(f'Average epoch time: {avg_epoch_time:.2f} seconds')
        print(f'Best accuracy: {best_acc:.2f}%')

        return results
    else:
        # Non-main processes don't return anything
        return None