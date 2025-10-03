import torch


class EarlyStopping:
    """
    Early stopping to terminate training when a monitored metric has stopped improving.

    Methods:
    check_stop(current_loss: float, model) -> bool:
        Check if training should be stopped based on the current loss.
    
    restore_best_weights(model) -> None:
        Restore the model weights from the best epoch.
    """
    def __init__(self, min_delta: float = 0.001, patience: int = 5) -> None:
        """
        Early stopping to terminate training when a monitored metric has stopped improving.

        Parameters:
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        patience: Number of epochs with no improvement after which training will be stopped.
        """
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.best_state_dict = None

    def check_stop(self, current_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should be stopped based on the current loss.

        Parameters:
        current_loss: The loss value for the current epoch.
        model: The model whose state_dict will be saved if it is the best so far.
        """
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_state_dict = model.state_dict()  # <- Save initial weights
            return False

        relative_change = abs((self.best_loss - current_loss) / (self.best_loss + 1e-8))

        if relative_change < self.min_delta:
            self.counter += 1
        else:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state_dict = model.state_dict()  # <- Save new best weights

        return self.counter >= self.patience

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        """
        Restore the model weights from the best epoch.

        Parameters:
        model: The model whose weights will be restored.
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)



class LRScheduling:
    """
    Reduce learning rate when a metric has stopped improving.
    
    Methods:
    step(current_loss: float) -> bool:
        Update the learning rate based on the current loss.
    """

    def __init__(
            self, 
            optimizer: torch.optim.Optimizer, 
            mode: str = 'min', 
            factor: float = 0.1, 
            patience: int = 10, 
            min_lr: float = 1e-6, 
            min_delta: float = 1e-4
            ) -> None:
        """
        Reduce learning rate when a metric has stopped improving.
        
        Parameters:
        optimizer: The optimizer whose learning rate will be reduced.
        mode: One of 'min' or 'max'. In 'min' mode, lr will be reduced when the quantity stopped decreasing. In 'max', when it stops increasing.
        factor: Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience: Number of epochs with no improvement after which learning rate will be reduced.
        min_lr: Lower bound on the learning rate.
        min_delta: Minimum change to qualify as an improvement.
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        
        self.best = None
        self.num_bad_epochs = 0
        self.last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, current_loss: float) -> bool:
        """
        Update the learning rate based on the current loss.
        
        Parameters:
        current_loss: The current value of the monitored metric.
        
        Returns:
        bool: True if the learning rate was reduced, False otherwise.
        """
        if self.best is None:
            self.best = current_loss
            return False
            
        if self.mode == 'min':
            is_better = current_loss < (self.best - self.min_delta)
        else:  # mode == 'max'
            is_better = current_loss > (self.best + self.min_delta)
        
        if is_better:
            self.best = current_loss
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            return True
            
        return False
    
    def _reduce_lr(self) -> None:
        """Reduce learning rate for all parameter groups."""
        for _, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr != old_lr:
                param_group['lr'] = new_lr
                print(f"Reduced learning rate from {old_lr:.2e} to {new_lr:.2e}")
                
        self.last_lr = [group['lr'] for group in self.optimizer.param_groups]



class GradientClipping:
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def on_backward_end(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)



class DropOutScheduling:
    """
    Adjust dropout rate based on current epoch to gradually reduce dropout to 0.
    
    Methods:
    step(current_epoch: int, n_total_epochs: int) -> Optional[float]:
        Update dropout rate based on current epoch and return new rate if changed.
    """
    def __init__(
        self,
        dropout_values: list[float] = None,
        initial_rate: float = 0.0
    ) -> None:
        """
        Adjust dropout rate based on training progress.

        Parameters:
        dropout_values: List of dropout values to use throughout training. If provided, initial_rate is ignored.
        initial_rate: Initial dropout rate (used if dropout_values is None)
        """
        if dropout_values is not None:
            self.dropout_values = dropout_values
            self.n_steps = len(dropout_values)
        else:
            self.dropout_values = None
            self.initial_rate = initial_rate
        
        self.current_rate = None
        self.last_epoch = -1

    def step(self, current_epoch: int, n_total_epochs: int) -> float | None:
        """
        Update dropout rate based on current epoch.

        Parameters:
        current_epoch: Current epoch number
        n_total_epochs: Total number of epochs for training

        Returns:
        float: New dropout rate if changed, None otherwise
        """
        if current_epoch <= self.last_epoch:
            return None
            
        self.last_epoch = current_epoch
        
        if self.dropout_values is not None:
            # Use predefined dropout values
            step_size = n_total_epochs // len(self.dropout_values)
            step_index = min(current_epoch // step_size, len(self.dropout_values) - 1)
            new_rate = self.dropout_values[step_index]
        else:
            # Linear decay from initial_rate to 0
            progress = current_epoch / n_total_epochs
            new_rate = self.initial_rate * (1 - progress)
        
        if self.current_rate is None or abs(new_rate - self.current_rate) > 1e-6:
            self.current_rate = new_rate
            return new_rate
        
        return None

    def get_current_rate(self) -> float:
        """Get the current dropout rate."""
        return self.current_rate if self.current_rate is not None else 0.0



class EMA:
    def __init__(self, model, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register initial shadow values
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def on_batch_end(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]