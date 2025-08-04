import sys
import json
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import nlpaug.augmenter.word as naw

# Import functions from other files
from train_req_model import RequirementsDataset
from test_req_model import predict_supplier_status
from req_to_jsonl import parse_csv, parse_excel, write_jsonl

# Set proxy with authentication only if needed
# Uncomment and set your actual proxy if required
# os.environ['HTTP_PROXY'] = 'http://username:password@proxy.company.com:port'
# os.environ['HTTPS_PROXY'] = 'http://username:password@proxy.company.com:port'

def contextual_augment(text, model_name="bert-base-uncased"):
    # You can switch model_name to "roberta-base" if preferred
    aug = naw.ContextualWordEmbsAug(
        model_path=model_name, action="substitute", device="cpu"
    )
    return aug.augment(text)

class ReqEvalApp:
    def __init__(self, master):
        self.master = master
        master.title("Requirement Evaluation Tool")
        master.geometry("800x700")
        master.configure(bg="#f0f0f0")
        
        self.label_map = {"agreed": 0, "partly agreed": 1, "not agreed": 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.model = None
        self.tokenizer = None
        self.history = []
        
        # Create notebook with tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.train_tab = ttk.Frame(self.notebook)
        self.predict_tab = ttk.Frame(self.notebook)
        self.convert_tab = ttk.Frame(self.notebook)
        self.history_tab = ttk.Frame(self.notebook)
        self.model_info_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.train_tab, text="Train Model")
        self.notebook.add(self.predict_tab, text="Predict Status")
        self.notebook.add(self.convert_tab, text="Convert Data")
        self.notebook.add(self.history_tab, text="History")
        self.notebook.add(self.model_info_tab, text="Model Info")
        
        # Setup Training Tab
        self.setup_train_tab()
        
        # Setup Prediction Tab
        self.setup_predict_tab()
        
        # Setup Convert Tab
        self.setup_convert_tab()
        
        # Setup History Tab
        self.setup_history_tab()
        
        # Setup Model Info Tab
        self.setup_model_info_tab()
        
        # Check if model exists and load it
        try:
            if os.path.exists("bert_req_eval_model") and os.path.exists(os.path.join("bert_req_eval_model", "config.json")):
                # Use the safer loading from test_req_model
                from test_req_model import load_model
                self.model, self.tokenizer = load_model()
                self.status_label.config(text="Model loaded successfully")
            else:
                self.status_label.config(text="No model found. Please train a model first.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")  # Print to console for debugging
            self.status_label.config(text=f"Error loading model: {str(e)}")
    def augment_training_data(self):
        data_file = self.data_path.get()
        if not os.path.exists(data_file):
            messagebox.showerror("Error", f"Training data file not found: {data_file}")
            return

        try:
            augmented_lines = []
            with open(data_file, encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    orig_text = item.get("text", "")
                    aug_text = contextual_augment(orig_text, model_name="bert-base-uncased")
                    augmented_lines.append(json.dumps({"text": aug_text, "supplier_status": item.get("supplier_status", "")}))
            
            aug_file = os.path.splitext(data_file)[0] + "_augmented.jsonl"
            with open(aug_file, "w", encoding="utf-8") as f:
                with open(data_file, encoding='utf-8') as orig_f:
                    for line in orig_f:
                        f.write(line)
                for line in augmented_lines:
                    f.write(line + "\n")
            
            messagebox.showinfo("Augmentation Complete", f"Augmented data saved to {aug_file}")
            self.data_path.set(aug_file)
            self.log_text.insert(tk.END, f"Augmented data created: {aug_file}\n")
            self.log_text.see(tk.END)
        except Exception as e:
            messagebox.showerror("Augmentation Error", f"Failed to augment data: {str(e)}")
    def setup_train_tab(self):
        frame = ttk.LabelFrame(self.train_tab, text="Model Training")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Data file selection
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(file_frame, text="Training Data:").pack(side="left")
        self.data_path = tk.StringVar(value="requirements_for_llm.jsonl")
        ttk.Entry(file_frame, textvariable=self.data_path, width=40).pack(side="left", padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_data).pack(side="left")
        
        # Training parameters
        param_frame = ttk.LabelFrame(frame, text="Training Parameters")
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Epochs
        epoch_frame = ttk.Frame(param_frame)
        epoch_frame.pack(fill="x", pady=5)
        ttk.Label(epoch_frame, text="Epochs:").pack(side="left")
        self.epochs = tk.IntVar(value=3)
        ttk.Spinbox(epoch_frame, from_=1, to=10, textvariable=self.epochs, width=5).pack(side="left", padx=5)
        
        # Batch size
        batch_frame = ttk.Frame(param_frame)
        batch_frame.pack(fill="x", pady=5)
        ttk.Label(batch_frame, text="Batch Size:").pack(side="left")
        self.batch_size = tk.IntVar(value=8)
        ttk.Spinbox(batch_frame, from_=1, to=32, textvariable=self.batch_size, width=5).pack(side="left", padx=5)
        
        # Layer selection
        layer_frame = ttk.Frame(param_frame)
        layer_frame.pack(fill="x", pady=5)
        ttk.Label(layer_frame, text="Layers to Train:").pack(side="left")
        
        self.train_all_layers = tk.BooleanVar(value=False)
        ttk.Radiobutton(layer_frame, text="Last 4 layers only", variable=self.train_all_layers, value=False).pack(side="left", padx=5)
        ttk.Radiobutton(layer_frame, text="All layers", variable=self.train_all_layers, value=True).pack(side="left", padx=5)
        
        # Progress display
        self.progress = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=600, mode='determinate')
        self.progress.pack(fill="x", padx=10, pady=10)
        
        self.status_label = ttk.Label(frame, text="Ready to train")
        self.status_label.pack(pady=5)
        
        # Training logs
        log_frame = ttk.LabelFrame(frame, text="Training Logs")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Training button
        self.train_button = ttk.Button(frame, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)
        
        # Augment button
        self.augment_button = ttk.Button(frame, text="Augment Training Data", command=self.augment_training_data)
        self.augment_button.pack(pady=5)
        
        # Plot area for loss curve
        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.plot.set_title("Training Loss")
        self.plot.set_xlabel("Epoch")
        self.plot.set_ylabel("Loss")
        
        self.canvas = FigureCanvasTkAgg(self.figure, frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def setup_predict_tab(self):
        frame = ttk.LabelFrame(self.predict_tab, text="Requirement Evaluation")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Enter requirement text:").pack(anchor="w", padx=10, pady=5)
        
        self.req_text = scrolledtext.ScrolledText(frame, height=5)
        self.req_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(button_frame, text="Predict", command=self.predict_requirement).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Clear", command=self.clear_prediction).pack(side="left")
        
        result_frame = ttk.LabelFrame(frame, text="Prediction Result")
        result_frame.pack(fill="x", padx=10, pady=10)
        
        self.result_label = ttk.Label(result_frame, text="No prediction yet", font=("Arial", 12))
        self.result_label.pack(pady=10)
        
        # Confidence scores
        self.confidence_frame = ttk.Frame(result_frame)
        self.confidence_frame.pack(fill="x", padx=10, pady=5)
        
        # Add prediction result visualization
        self.confidence_bars = {}
        for status in ["agreed", "partly agreed", "not agreed"]:
            label_frame = ttk.Frame(self.confidence_frame)
            label_frame.pack(fill="x", pady=2)
            
            ttk.Label(label_frame, text=f"{status.capitalize()}: ", width=15).pack(side="left")
            
            self.confidence_bars[status] = ttk.Progressbar(label_frame, orient="horizontal", length=300, mode="determinate")
            self.confidence_bars[status].pack(side="left", padx=5)
            
            self.confidence_bars[status+"_label"] = ttk.Label(label_frame, text="0%", width=5)
            self.confidence_bars[status+"_label"].pack(side="left")
    
    def setup_convert_tab(self):
        frame = ttk.LabelFrame(self.convert_tab, text="Convert CSV/Excel to JSONL")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Input file selection
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(input_frame, text="Input File:").pack(side="left")
        self.input_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path, width=40).pack(side="left", padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_input_file).pack(side="left")
        
        # Output file selection
        output_frame = ttk.Frame(frame)
        output_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(output_frame, text="Output JSONL:").pack(side="left")
        self.output_path = tk.StringVar(value="requirements_for_llm.jsonl")
        ttk.Entry(output_frame, textvariable=self.output_path, width=40).pack(side="left", padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_file).pack(side="left")
        
        # Convert button
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", padx=10, pady=20)
        
        ttk.Button(button_frame, text="Convert to JSONL", command=self.convert_to_jsonl).pack(side="left", padx=10)
        
        # Status label
        self.convert_status = ttk.Label(frame, text="")
        self.convert_status.pack(pady=10)
        
        # Preview area
        preview_frame = ttk.LabelFrame(frame, text="Data Preview")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.preview_text = scrolledtext.ScrolledText(preview_frame, height=10)
        self.preview_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def setup_history_tab(self):
        frame = ttk.Frame(self.history_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Prediction History", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
        
        # Create treeview for history
        columns = ("Requirement", "Prediction")
        self.history_tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        # Set column headings
        self.history_tree.heading("Requirement", text="Requirement")
        self.history_tree.heading("Prediction", text="Prediction")
        
        # Set column widths
        self.history_tree.column("Requirement", width=500)
        self.history_tree.column("Prediction", width=100)
        
        self.history_tree.pack(fill="both", expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.history_tree.yview)
        scrollbar.pack(side="right", fill="y")
        
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", pady=10)
        
        ttk.Button(button_frame, text="Export History", command=self.export_history).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear History", command=self.clear_history).pack(side="left")
    
    def setup_model_info_tab(self):
        frame = ttk.Frame(self.model_info_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Model Features & Parameters", font=("Arial", 14, "bold")).pack(anchor="w", pady=10)

        self.model_info_text = scrolledtext.ScrolledText(frame, height=20)
        self.model_info_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.update_model_info()
    
    def update_model_info(self):
        info = []
        info.append("Model Directory: bert_req_eval_model")
        info.append(f"Model Type: {'RoBERTa' if getattr(self, 'model', None) and 'roberta' in str(type(self.model)).lower() else 'BERT'}")
        info.append(f"Loss Function: CrossEntropyLoss")
        info.append(f"Optimizer: AdamW")
        info.append(f"Learning Rate: {getattr(self, 'lr', 3e-5)}")
        info.append(f"Batch Size: {self.batch_size.get() if hasattr(self, 'batch_size') else 'N/A'}")
        info.append(f"Epochs: {self.epochs.get() if hasattr(self, 'epochs') else 'N/A'}")
        info.append(f"Layers Trained: {'All layers' if self.train_all_layers.get() else 'Last 4 layers only'}")
        info.append(f"Label Map: {self.label_map}")
        info.append(f"Reverse Label Map: {self.reverse_label_map}")
        info.append(f"Tokenizer: {type(self.tokenizer).__name__ if self.tokenizer else 'N/A'}")
        info.append(f"Model Loaded: {'Yes' if self.model else 'No'}")
        info.append(f"Training Data File: {self.data_path.get() if hasattr(self, 'data_path') else 'N/A'}")
        info.append(f"Augmentation: ContextualWordEmbsAug (BERT/RoBERTa)")

        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(tk.END, "\n".join(info))

    def browse_data(self):
        filename = filedialog.askopenfilename(
            title="Select Training Data",
            filetypes=(("JSONL files", "*.jsonl"), ("All files", "*.*"))
        )
        if filename:
            self.data_path.set(filename)
    
    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=(("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.input_path.set(filename)
            self.preview_input_file(filename)
    
    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Save JSONL File",
            defaultextension=".jsonl",
            filetypes=(("JSONL files", "*.jsonl"), ("All files", "*.*"))
        )
        if filename:
            self.output_path.set(filename)
    
    def preview_input_file(self, filename):
        try:
            self.preview_text.delete(1.0, tk.END)
            if filename.lower().endswith('.xlsx'):
                import openpyxl
                wb = openpyxl.load_workbook(filename)
                ws = wb.active
                headers = [str(cell.value) for cell in next(ws.iter_rows(min_row=1, max_row=1))]
                self.preview_text.insert(tk.END, "Headers: " + ", ".join(headers) + "\n\n")
                
                # Preview first 5 rows
                for i, row in enumerate(ws.iter_rows(min_row=2, max_row=6, values_only=True)):
                    self.preview_text.insert(tk.END, f"Row {i+1}: {str(row)}\n")
            
            elif filename.lower().endswith('.csv'):
                with open(filename, newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    headers = next(reader)
                    self.preview_text.insert(tk.END, "Headers: " + ", ".join(headers) + "\n\n")
                    
                    # Preview first 5 rows
                    for i, row in enumerate(reader):
                        if i >= 5:
                            break
                        self.preview_text.insert(tk.END, f"Row {i+1}: {str(row)}\n")
        
        except Exception as e:
            self.preview_text.insert(tk.END, f"Error previewing file: {str(e)}")
    
    def convert_to_jsonl(self):
        try:
            input_file = self.input_path.get()
            output_file = self.output_path.get()
            
            if not input_file:
                messagebox.showerror("Input Error", "Please select an input file.")
                return
                
            if not os.path.exists(input_file):
                messagebox.showerror("File Error", f"File not found: {input_file}")
                return
            
            self.convert_status.config(text="Converting...")
            self.master.update()
            
            # Use functions from req_to_jsonl.py
            if input_file.lower().endswith('.xlsx'):
                requirements = parse_excel(input_file)
            elif input_file.lower().endswith('.csv'):
                requirements = parse_csv(input_file)
            else:
                messagebox.showerror("File Error", "Unsupported file type. Use .xlsx or .csv")
                return
            
            write_jsonl(requirements, output_file)
            
            self.convert_status.config(text=f"Converted {len(requirements)} requirements to {output_file}")
            
            # Preview output
            self.preview_text.delete(1.0, tk.END)
            with open(output_file, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    self.preview_text.insert(tk.END, f"{line}\n")
                    
            messagebox.showinfo("Conversion Complete", f"Converted {len(requirements)} requirements to {output_file}")
            
        except Exception as e:
            self.convert_status.config(text=f"Error: {str(e)}")
            messagebox.showerror("Conversion Error", f"Failed to convert file: {str(e)}")
    
    def train_model(self):
        try:
            data_file = self.data_path.get()
            if not os.path.exists(data_file):
                messagebox.showerror("Error", f"Training data file not found: {data_file}")
                return
                
            # Reset status message at the beginning of training
            self.status_label.config(text="Training in progress...")
            self.log_text.delete(1.0, tk.END)
        
            # Define callbacks for the training process
            def on_log(msg):
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.master.update_idletasks()
            
            def on_progress(step, total_steps):
                self.progress["value"] = step
                self.master.update_idletasks()
            
            def on_epoch_end(epoch, loss):
                # Nothing extra needed here - handled by on_log
                pass
            
            callbacks = {
                'on_log': on_log,
                'on_progress': on_progress,
                'on_epoch_end': on_epoch_end
            }
            
            # Use train_model from train_req_model.py
            from train_req_model import train_model as train_requirement_model
            
            # Clear existing models
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            
            # Prepare UI for training
            num_epochs = self.epochs.get()
            batch_size = self.batch_size.get()
            self.progress["maximum"] = 100  # Will be updated by the callback
            
            # Run the training process
            model, tokenizer, history = train_requirement_model(
                data_file,
                model_dir="bert_req_eval_model",
                use_roberta=True,
                batch_size=batch_size,
                epochs=num_epochs,
                train_all_layers=self.train_all_layers.get(),
                callbacks=callbacks
            )
            
            # Check if model was saved to a different directory
            actual_model_dir = history.get('model_dir', "bert_req_eval_model")
            if actual_model_dir != "bert_req_eval_model":
                self.log_text.insert(tk.END, f"Note: Model was saved to alternate location: {actual_model_dir}\n")

            # Update plot with training history
            self.plot.clear()
            self.plot.set_title("Training Loss")
            self.plot.set_xlabel("Epoch")
            self.plot.set_ylabel("Loss")
            epochs = list(range(1, num_epochs + 1))
            self.plot.plot(epochs, history['epoch_losses'], 'b-')
            self.canvas.draw()
            
            # Load the trained model for use in the app
            self.model = model
            self.tokenizer = tokenizer
            
            self.status_label.config(text="Training complete!")
            messagebox.showinfo("Success", "Model trained and saved successfully!")
            
        except Exception as e:
            self.log_text.insert(tk.END, f"Error during training: {str(e)}\n")
            self.log_text.see(tk.END)
            self.status_label.config(text="Training error")
            messagebox.showerror("Error", f"Training failed: {str(e)}")
    
    def predict_requirement(self):
        try:
            req_text = self.req_text.get("1.0", tk.END).strip()
            if not req_text:
                messagebox.showwarning("Input Error", "Please enter a requirement text.")
                return
            
            # Check if model exists
            if not os.path.exists("bert_req_eval_model"):
                messagebox.showerror("Model Error", "Model not found. Please train a model first.")
                return
            
            # Use the predict_with_confidence function from test_req_model.py
            from test_req_model import predict_with_confidence
            
            # Get prediction and confidence scores
            result, confidence_scores = predict_with_confidence(req_text)
            
            self.result_label.config(text=f"Predicted: {result.upper()}")
            
            # Update confidence bars
            for status in ["agreed", "partly agreed", "not agreed"]:
                prob_value = confidence_scores[status]
                self.confidence_bars[status]["value"] = prob_value
                self.confidence_bars[status+"_label"].config(text=f"{prob_value:.1f}%")
            
            # Add to history
            self.history.append((req_text, result))
            self.history_tree.insert("", "end", values=(req_text[:80] + "..." if len(req_text) > 80 else req_text, result))
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")
            # Try to reload the model
            try:
                from test_req_model import load_model
                self.model, self.tokenizer = load_model()
                messagebox.showinfo("Model Reloaded", "The model was reloaded. Please try prediction again.")
            except Exception as reload_error:
                messagebox.showerror("Model Error", f"Could not load model: {str(reload_error)}")
    
    def clear_prediction(self):
        self.req_text.delete("1.0", tk.END)
        self.result_label.config(text="No prediction yet")
        for status in ["agreed", "partly agreed", "not agreed"]:
            self.confidence_bars[status]["value"] = 0
            self.confidence_bars[status+"_label"].config(text="0%")
    
    def export_history(self):
        if not self.history:
            messagebox.showinfo("Export", "No prediction history to export.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export History",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Requirement,Prediction\n")
                    for req, pred in self.history:
                        # Escape quotes and remove newlines
                        req = req.replace('"', '""').replace('\n', ' ').replace('\r', '')
                        f.write(f'"{req}",{pred}\n')
                messagebox.showinfo("Export", f"History exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export history: {str(e)}")
    
    def clear_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear the prediction history?"):
            self.history = []
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
    
    def reset_gui(self):
        # Clear status label
        self.status_label.config(text="")
        # Clear log text
        self.log_text.delete(1.0, tk.END)
        # Clear requirement text input
        if hasattr(self, 'req_text'):
            self.req_text.delete("1.0", tk.END)
        # Clear prediction result label
        if hasattr(self, 'result_label'):
            self.result_label.config(text="")
        # Reset confidence bars
        if hasattr(self, 'confidence_bars'):
            for status in self.confidence_bars:
                self.confidence_bars[status]["value"] = 0
                self.confidence_bars[status+"_label"].config(text="0.0%")
        # Clear history
        if hasattr(self, 'history_tree'):
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
        # Clear convert status
        if hasattr(self, 'convert_status'):
            self.convert_status.config(text="")
        # Optionally reset other fields as needed

if __name__ == "__main__":
    root = tk.Tk()
    app = ReqEvalApp(root)
    root.mainloop()