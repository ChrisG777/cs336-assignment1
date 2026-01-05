import json
import time
import os
import matplotlib.pyplot as plt

class ExperimentLogger:
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 1. Save Hyperparameters (Config)
        self.config = config
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            
        self.metrics = []
        self.start_time = time.time()
        
    def log(self, iteration, train_loss=None, val_loss=None, lr=None):
        """
        Logs a single step's metrics.
        """
        current_time = time.time()
        wall_clock_time = current_time - self.start_time
        
        entry = {
            "iteration": iteration,
            "wall_clock_time": wall_clock_time,
            "lr": lr
        }
        if train_loss is not None:
            entry["train_loss"] = train_loss
        if val_loss is not None:
            entry["val_loss"] = val_loss
            
        self.metrics.append(entry)
        
        # Auto-save every log so you don't lose data if job crashes
        self.save_metrics()
        
    def save_metrics(self):
        with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

    def plot(self):
        """
        Generates a standard loss curve plot from the stored metrics.
        """
        iterations = [m["iteration"] for m in self.metrics]
        train_losses = [m.get("train_loss") for m in self.metrics if "train_loss" in m]
        # Align train iterations (assuming log is called every iter)
        train_iters = [m["iteration"] for m in self.metrics if "train_loss" in m]

        val_losses = [m.get("val_loss") for m in self.metrics if "val_loss" in m]
        val_iters = [m["iteration"] for m in self.metrics if "val_loss" in m]
        
        plt.figure(figsize=(10, 6))
        if train_losses:
            plt.plot(train_iters, train_losses, label="Train Loss", alpha=0.6)
        if val_losses:
            plt.plot(val_iters, val_losses, 'r-o', label="Validation Loss")
            
        plt.xlabel("Iterations")
        plt.ylabel("Cross Entropy Loss")
        plt.title(f"Training Run: {self.config.get('exp_name', 'Unnamed')}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.log_dir, "loss_curve.png"))
        plt.close()