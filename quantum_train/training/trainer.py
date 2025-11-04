"""Training loop for Quantum-Train model."""
import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import calculate_metrics

class QuantumTrainTrainer:
    """Trainer class for Quantum-Train models."""
    
    def __init__(self, model, config, train_loader, val_loader):
        """Initialize trainer."""
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (only for quantum + mapping parameters)
        trainable_params = list(model.quantum_circuit.parameters()) + \
                          list(model.mapping_model.parameters())
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=config.learning_rate
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.n_epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Check for numerical instability
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Loss is {loss.item()}, skipping batch")
                continue
            
            # DEBUG: Check gradient flow on first batch WITHOUT calling backward()
            if batch_idx == 0 and epoch == 0:
                print("\n=== Gradient Flow Check ===")
                print(f"Loss: {loss.item():.4f}")
                
                # Use autograd.grad to inspect WITHOUT consuming the graph
                qnn_grad = torch.autograd.grad(
                    loss, 
                    self.model.quantum_circuit.phi, 
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if qnn_grad is not None:
                    print(f"QNN grad norm: {qnn_grad.norm().item():.6f}")
                else:
                    print("WARNING: QNN gradients are None!")
                
                # Check mapping model gradients
                for name, param in self.model.mapping_model.named_parameters():
                    grad = torch.autograd.grad(
                        loss, 
                        param, 
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    if grad is not None:
                        print(f"Mapping {name} grad norm: {grad.norm().item():.6f}")
                    else:
                        print(f"WARNING: Mapping {name} gradients are None!")
                
                print("=========================\n")
            
                # Backward pass
        loss.backward()
        
        # Use simple gradient clipping instead of scaling
        torch.nn.utils.clip_grad_norm_(
            list(self.model.quantum_circuit.parameters()) + 
            list(self.model.mapping_model.parameters()),
            max_norm=10.0
        )
        
        self.optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        if batch_idx % self.config.log_interval == 0:
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self):
        """Complete training loop."""
        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_total_params()}")
        
        for epoch in range(self.config.n_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'quantum_state_dict': self.model.quantum_circuit.state_dict(),
            'mapping_state_dict': self.model.mapping_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        path = f"{self.config.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
