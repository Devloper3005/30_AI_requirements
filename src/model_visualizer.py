"""
Model Layer Visualization Module

This module provides real-time visualization of transformer model layers
during training, showing activation levels, gradients, and layer states.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import os
import time
from threading import Thread
from queue import Queue
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for embedding in tkinter


class LayerVisualizer:
    def __init__(self, model, model_type="bert", num_layers=12):
        """
        Initialize a layer visualizer for transformer models.
        
        Args:
            model: The transformer model being trained
            model_type: Model type (bert or roberta)
            num_layers: Number of layers in the model
        """
        self.model = model
        self.model_type = model_type
        self.num_layers = num_layers
        self.layer_prefix = f"{model_type}.encoder.layer."
        self.activation_data = []
        self.gradient_data = []
        self.layer_status = []  # Frozen or trainable
        self.loss_history = []
        self.accuracy_history = []
        self.epoch_markers = []
        
        self.animation_active = False
        self.data_queue = Queue()
        self.fig = None
        self.window = None
        self.is_setup = False
        
        # Register hooks for capturing activation and gradient information
        self.activation_hooks = []
        self.register_hooks()
        
    def register_hooks(self):
        """Register forward and backward hooks to collect activation and gradient data"""
        # Clear existing hooks if any
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        
        # Setup new hooks for each layer
        for i in range(self.num_layers):
            layer_name = f"{self.layer_prefix}{i}"
            if hasattr(self.model, self.model_type) and hasattr(getattr(self.model, self.model_type), "encoder"):
                layer = getattr(getattr(self.model, self.model_type).encoder.layer, str(i))
                
                # Forward hook for activations
                self.activation_hooks.append(
                    layer.register_forward_hook(
                        lambda module, input_tensor, output_tensor, idx=i: 
                        self.capture_activation(idx, output_tensor)
                    )
                )
                
                # Backward hook for gradients
                self.activation_hooks.append(
                    layer.register_full_backward_hook(
                        lambda module, grad_in, grad_out, idx=i:
                        self.capture_gradient(idx, grad_out[0])
                    )
                )
        
        # Get layer status (frozen or trainable)
        self.update_layer_status()
    
    def update_layer_status(self):
        """Update the frozen/unfrozen status of each layer"""
        self.layer_status = []
        for i in range(self.num_layers):
            is_trainable = False
            layer_name = f"{self.layer_prefix}{i}"
            
            # Check parameters in the layer
            for name, param in self.model.named_parameters():
                if layer_name in name and param.requires_grad:
                    is_trainable = True
                    break
            
            self.layer_status.append(is_trainable)
    
    def capture_activation(self, layer_idx, output_tensor):
        """Capture activation values from forward pass"""
        if self.animation_active:
            # Get mean activation magnitude
            activation = output_tensor[0].detach().abs().mean().cpu().item()
            
            # Put in queue to avoid threading issues
            self.data_queue.put(('activation', layer_idx, activation))
    
    def capture_gradient(self, layer_idx, grad_tensor):
        """Capture gradient values from backward pass"""
        if self.animation_active:
            # Get mean gradient magnitude
            gradient = grad_tensor.detach().abs().mean().cpu().item()
            
            # Put in queue to avoid threading issues
            self.data_queue.put(('gradient', layer_idx, gradient))
    
    def add_training_metrics(self, epoch, loss, accuracy):
        """Add training metrics to history"""
        if self.animation_active:
            self.data_queue.put(('metrics', epoch, loss, accuracy))
    
    def setup_visualization(self, parent_window=None):
        """Setup the visualization window and plots"""
        if parent_window:
            # Create a toplevel window if parent is provided
            self.window = tk.Toplevel(parent_window)
            self.window.title("Model Layer Visualization")
            self.window.geometry("1200x800")
        else:
            # Create a standalone tkinter window
            self.window = tk.Tk()
            self.window.title("Model Layer Visualization")
            self.window.geometry("1200x800")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Layer activation heatmap
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Layer gradients heatmap
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Loss history
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Accuracy history
        
        # Initialize plots
        self.activation_data = np.zeros((self.num_layers, 20))  # Keep 20 time steps
        self.gradient_data = np.zeros((self.num_layers, 20))
        self.loss_history = []
        self.accuracy_history = []
        self.epoch_markers = []
        
        # Create colormaps
        self.cmap1 = LinearSegmentedColormap.from_list("activation", ["#f0f0f0", "#ff9900", "#ff0000"])
        self.cmap2 = LinearSegmentedColormap.from_list("gradients", ["#f0f0f0", "#00aaff", "#0000ff"])
        
        # Initial plots
        self.im1 = self.ax1.imshow(self.activation_data, aspect='auto', cmap=self.cmap1, 
                                  vmin=0, vmax=1, interpolation='none')
        self.im2 = self.ax2.imshow(self.gradient_data, aspect='auto', cmap=self.cmap2, 
                                  vmin=0, vmax=0.1, interpolation='none')
        
        # Set titles and labels
        self.ax1.set_title("Layer Activations (Live)")
        self.ax1.set_xlabel("Time Steps")
        self.ax1.set_ylabel("Layer")
        
        self.ax2.set_title("Layer Gradients (Live)")
        self.ax2.set_xlabel("Time Steps")
        self.ax2.set_ylabel("Layer")
        
        self.ax3.set_title("Training Loss")
        self.ax3.set_xlabel("Batch")
        self.ax3.set_ylabel("Loss")
        
        self.ax4.set_title("Training Accuracy")
        self.ax4.set_xlabel("Batch")
        self.ax4.set_ylabel("Accuracy")
        
        # Add layer status indicators
        self.ax1.set_yticks(range(self.num_layers))
        self.ax2.set_yticks(range(self.num_layers))
        
        # Label layers by trainable status
        self.update_layer_labels()
        
        # Add colorbar
        self.fig.colorbar(self.im1, ax=self.ax1, label="Activation Magnitude")
        self.fig.colorbar(self.im2, ax=self.ax2, label="Gradient Magnitude")
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()
        
        # Add a button to save the visualization
        self.save_button = tk.Button(self.window, text="Save Visualization", 
                                    command=self.save_visualization)
        self.save_button.pack(pady=10)
        
        # Add a status label
        self.status_label = tk.Label(self.window, text="Initializing visualization...", 
                                    font=("Helvetica", 12))
        self.status_label.pack(pady=5)
        
        self.is_setup = True
        self.animation_active = True
        
        # Start animation thread
        self.ani_thread = Thread(target=self.animation_loop)
        self.ani_thread.daemon = True
        self.ani_thread.start()
    
    def update_layer_labels(self):
        """Update layer labels based on frozen/trainable status"""
        layer_labels = []
        for i, is_trainable in enumerate(self.layer_status):
            status = "Trainable" if is_trainable else "Frozen"
            layer_labels.append(f"L{i} ({status})")
        
        self.ax1.set_yticklabels(layer_labels)
        self.ax2.set_yticklabels(layer_labels)
    
    def animation_loop(self):
        """Animation thread to update visualization"""
        while self.animation_active:
            # Process all data in the queue
            self.process_queue_data()
            
            if self.is_setup and self.window.winfo_exists():
                try:
                    # Update the plots
                    self.update_plots()
                    # Update the canvas
                    self.canvas.draw_idle()
                    self.window.update()
                except Exception as e:
                    print(f"Error updating visualization: {e}")
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)
    
    def process_queue_data(self):
        """Process all data in the queue"""
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                data_type = data[0]
                
                if data_type == 'activation':
                    _, layer_idx, activation = data
                    # Roll the data arrays
                    self.activation_data[layer_idx] = np.roll(self.activation_data[layer_idx], -1)
                    # Update the newest value (rightmost)
                    self.activation_data[layer_idx, -1] = activation
                
                elif data_type == 'gradient':
                    _, layer_idx, gradient = data
                    self.gradient_data[layer_idx] = np.roll(self.gradient_data[layer_idx], -1)
                    self.gradient_data[layer_idx, -1] = gradient
                
                elif data_type == 'metrics':
                    _, epoch, loss, accuracy = data
                    self.loss_history.append(loss)
                    self.accuracy_history.append(accuracy)
                    
                    # Mark the start of a new epoch
                    if len(self.loss_history) > 1 and (len(self.epoch_markers) == 0 or 
                                                     self.epoch_markers[-1] < len(self.loss_history) - 1):
                        self.epoch_markers.append(len(self.loss_history) - 1)
                        
                    # Update status label
                    if self.is_setup:
                        self.status_label.config(text=f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.2%}")
            except:
                # Skip errors in data processing
                pass
    
    def update_plots(self):
        """Update all plots with current data"""
        if not self.is_setup:
            return
        
        # Update activation and gradient heatmaps
        self.im1.set_array(self.activation_data)
        self.im2.set_array(self.gradient_data)
        
        # Update loss and accuracy plots
        x = range(len(self.loss_history))
        self.ax3.clear()
        self.ax3.plot(x, self.loss_history, 'b-')
        self.ax3.set_title("Training Loss")
        self.ax3.set_xlabel("Batch")
        self.ax3.set_ylabel("Loss")
        
        # Add epoch markers
        for marker in self.epoch_markers:
            self.ax3.axvline(x=marker, color='r', linestyle='--', alpha=0.5)
        
        self.ax4.clear()
        self.ax4.plot(x, self.accuracy_history, 'g-')
        self.ax4.set_title("Training Accuracy")
        self.ax4.set_xlabel("Batch")
        self.ax4.set_ylabel("Accuracy")
        
        # Add epoch markers
        for marker in self.epoch_markers:
            self.ax4.axvline(x=marker, color='r', linestyle='--', alpha=0.5)
    
    def save_visualization(self):
        """Save the current visualization to a file"""
        if not self.is_setup:
            return
            
        save_path = os.path.join(os.getcwd(), "model_visualization.png")
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Visualization saved to {save_path}")
    
    def stop_visualization(self):
        """Stop the visualization and clean up"""
        self.animation_active = False
        
        # Remove hooks
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        
        # Close window if it exists
        if self.window and self.window.winfo_exists():
            self.window.destroy()
            
        self.is_setup = False


# Helper function to start visualization
def start_model_visualization(model, parent_window=None, model_type="bert"):
    """
    Start visualizing a model's layers in a new window
    
    Args:
        model: The model to visualize
        parent_window: Optional parent tkinter window
        model_type: 'bert' or 'roberta'
    
    Returns:
        LayerVisualizer instance
    """
    visualizer = LayerVisualizer(model, model_type=model_type)
    visualizer.setup_visualization(parent_window)
    return visualizer


if __name__ == "__main__":
    # Test code
    import torch
    from transformers import BertForSequenceClassification
    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    
    # Create a tkinter window
    root = tk.Tk()
    root.title("Test Window")
    root.geometry("200x100")
    
    # Add button to launch visualizer
    button = tk.Button(root, text="Launch Visualizer", 
                     command=lambda: start_model_visualization(model, root))
    button.pack(pady=20)
    
    root.mainloop()
