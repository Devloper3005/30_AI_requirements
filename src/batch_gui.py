    def setup_batch_tab(self):
        """Setup the batch prediction tab for model verification"""
        frame = ttk.LabelFrame(self.batch_tab, text="Batch Model Verification")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Input file selection
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(input_frame, text="Input File:").pack(side="left")
        self.batch_input_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.batch_input_path, width=40).pack(side="left", padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_batch_input).pack(side="left")
        
        # Output file selection
        output_frame = ttk.Frame(frame)
        output_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(output_frame, text="Output File:").pack(side="left")
        self.batch_output_path = tk.StringVar(value="batch_predictions.xlsx")
        ttk.Entry(output_frame, textvariable=self.batch_output_path, width=40).pack(side="left", padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_batch_output).pack(side="left")
        
        # Options frame
        options_frame = ttk.LabelFrame(frame, text="Options")
        options_frame.pack(fill="x", padx=10, pady=10)
        
        # Format selection
        format_frame = ttk.Frame(options_frame)
        format_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(format_frame, text="Output Format:").pack(side="left")
        self.batch_format = tk.StringVar(value="excel")
        format_menu = ttk.Combobox(format_frame, textvariable=self.batch_format, 
                                 values=["excel", "csv", "jsonl"], width=10)
        format_menu.pack(side="left", padx=5)
        format_menu.current(0)
        
        # Confidence threshold
        threshold_frame = ttk.Frame(options_frame)
        threshold_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(threshold_frame, text="Min. Confidence:").pack(side="left")
        self.confidence_threshold = tk.DoubleVar(value=0.0)
        threshold_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                  length=200, variable=self.confidence_threshold)
        threshold_scale.pack(side="left", padx=5)
        
        threshold_value_label = ttk.Label(threshold_frame, textvariable=tk.StringVar())
        threshold_value_label.pack(side="left", padx=5)
        
        # Update threshold label when scale changes
        def update_threshold_label(*args):
            threshold_value_label.config(text=f"{self.confidence_threshold.get():.2f}")
        
        self.confidence_threshold.trace_add("write", update_threshold_label)
        update_threshold_label()  # Initial update
        
        # Report option
        report_frame = ttk.Frame(options_frame)
        report_frame.pack(fill="x", padx=10, pady=5)
        
        self.generate_report = tk.BooleanVar(value=True)
        ttk.Checkbutton(report_frame, text="Generate evaluation report (if input has actual statuses)",
                      variable=self.generate_report).pack(side="left", padx=5)
        
        # Run button
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", padx=10, pady=20)
        
        self.batch_progress = ttk.Progressbar(button_frame, orient=tk.HORIZONTAL, length=400, mode='indeterminate')
        self.batch_progress.pack(side="bottom", pady=10)
        
        ttk.Button(button_frame, text="Run Batch Verification", command=self.run_batch_prediction).pack(side="left", padx=10)
        
        # Status label
        self.batch_status = ttk.Label(frame, text="")
        self.batch_status.pack(pady=10)
        
        # Results frame with notebook for different views
        results_frame = ttk.LabelFrame(frame, text="Results")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create notebook for results
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill="both", expand=True)
        
        # Summary tab
        self.summary_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_tab, text="Summary")
        
        # Summary text widget
        self.summary_text = scrolledtext.ScrolledText(self.summary_tab, height=10)
        self.summary_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Details tab
        self.details_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.details_tab, text="Details")
        
        # Create treeview for details
        columns = ("ID", "Text", "Predicted", "Confidence", "Actual")
        self.results_tree = ttk.Treeview(self.details_tab, columns=columns, show="headings")
        
        # Set column headings
        self.results_tree.heading("ID", text="ID")
        self.results_tree.heading("Text", text="Requirement")
        self.results_tree.heading("Predicted", text="Predicted Status")
        self.results_tree.heading("Confidence", text="Confidence")
        self.results_tree.heading("Actual", text="Actual Status")
        
        # Set column widths
        self.results_tree.column("ID", width=50)
        self.results_tree.column("Text", width=400)
        self.results_tree.column("Predicted", width=100)
        self.results_tree.column("Confidence", width=80)
        self.results_tree.column("Actual", width=100)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(self.details_tab, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Pack tree and scrollbar
        self.results_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
    
    def browse_batch_input(self):
        """Browse for input file for batch prediction"""
        filetypes = [
            ("All Supported Files", "*.xlsx *.xls *.csv *.jsonl"),
            ("Excel Files", "*.xlsx *.xls"),
            ("CSV Files", "*.csv"),
            ("JSONL Files", "*.jsonl"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(title="Select Input File", filetypes=filetypes)
        if filename:
            self.batch_input_path.set(filename)
            
            # Set default output filename based on input
            base = os.path.basename(filename)
            name, _ = os.path.splitext(base)
            output_ext = ".xlsx" if self.batch_format.get() == "excel" else f".{self.batch_format.get()}"
            self.batch_output_path.set(f"{name}_predictions{output_ext}")
    
    def browse_batch_output(self):
        """Browse for output file for batch prediction"""
        format_ext = self.batch_format.get()
        filetypes = []
        
        if format_ext == "excel":
            filetypes = [("Excel Files", "*.xlsx"), ("All Files", "*.*")]
            default_ext = ".xlsx"
        elif format_ext == "csv":
            filetypes = [("CSV Files", "*.csv"), ("All Files", "*.*")]
            default_ext = ".csv"
        elif format_ext == "jsonl":
            filetypes = [("JSONL Files", "*.jsonl"), ("All Files", "*.*")]
            default_ext = ".jsonl"
        
        filename = filedialog.asksaveasfilename(
            title="Save Results As", 
            filetypes=filetypes,
            defaultextension=default_ext
        )
        if filename:
            self.batch_output_path.set(filename)
    
    def run_batch_prediction(self):
        """Run batch prediction on the selected input file"""
        try:
            input_file = self.batch_input_path.get()
            output_file = self.batch_output_path.get()
            
            if not input_file:
                messagebox.showerror("Input Error", "Please select an input file.")
                return
                
            if not os.path.exists(input_file):
                messagebox.showerror("File Error", f"File not found: {input_file}")
                return
            
            # Update status
            self.batch_status.config(text="Running batch prediction...")
            self.batch_progress.start()
            self.master.update_idletasks()
            
            # Import batch_predict here to avoid circular imports
            import batch_predict
            
            # Run in a separate thread
            import threading
            
            def prediction_thread():
                try:
                    # Get options from UI
                    format_type = self.batch_format.get()
                    threshold = self.confidence_threshold.get()
                    
                    # Run batch prediction
                    stats, results = batch_predict.batch_predict(
                        input_file, 
                        output_file, 
                        format_type,
                        threshold
                    )
                    
                    # Generate report if requested
                    report_file = None
                    metrics = None
                    if self.generate_report.get() and output_file:
                        report_file = os.path.splitext(output_file)[0] + "_report.txt"
                        metrics = batch_predict.generate_comparison_report(output_file, report_file)
                    
                    # Update UI in the main thread
                    self.master.after(0, lambda: self.update_batch_results(stats, results, metrics, report_file))
                    
                except Exception as e:
                    # Update UI with error in the main thread
                    self.master.after(0, lambda: self.handle_batch_error(str(e)))
            
            # Start thread
            thread = threading.Thread(target=prediction_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.batch_progress.stop()
            self.batch_status.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to run batch prediction: {str(e)}")
    
    def update_batch_results(self, stats, results, metrics=None, report_file=None):
        """Update UI with batch prediction results"""
        # Stop progress bar
        self.batch_progress.stop()
        
        # Update status
        self.batch_status.config(text=f"Completed batch prediction of {stats['total']} requirements")
        
        # Update summary
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, f"Batch Prediction Summary\n")
        self.summary_text.insert(tk.END, f"=======================\n\n")
        self.summary_text.insert(tk.END, f"Total requirements: {stats['total']}\n\n")
        self.summary_text.insert(tk.END, f"Results by category:\n")
        self.summary_text.insert(tk.END, f"  • Agreed: {stats['agreed']} ({100 * stats['agreed'] / stats['total']:.1f}%)\n")
        self.summary_text.insert(tk.END, f"  • Partly Agreed: {stats['partly_agreed']} ({100 * stats['partly_agreed'] / stats['total']:.1f}%)\n")
        self.summary_text.insert(tk.END, f"  • Not Agreed: {stats['not_agreed']} ({100 * stats['not_agreed'] / stats['total']:.1f}%)\n")
        
        if stats['low_confidence'] > 0:
            self.summary_text.insert(tk.END, f"  • Low Confidence: {stats['low_confidence']} ({100 * stats['low_confidence'] / stats['total']:.1f}%)\n")
        
        self.summary_text.insert(tk.END, f"\nResults saved to: {self.batch_output_path.get()}\n")
        
        # Add evaluation metrics if available
        if metrics:
            self.summary_text.insert(tk.END, f"\nEvaluation Metrics:\n")
            self.summary_text.insert(tk.END, f"  • Overall Accuracy: {metrics['accuracy']:.4f}\n\n")
            self.summary_text.insert(tk.END, f"Per-class Metrics:\n")
            
            for category, class_metrics in metrics['class_metrics'].items():
                self.summary_text.insert(tk.END, f"  • {category}:\n")
                self.summary_text.insert(tk.END, f"    - Precision: {class_metrics['precision']:.4f}\n")
                self.summary_text.insert(tk.END, f"    - Recall: {class_metrics['recall']:.4f}\n")
                self.summary_text.insert(tk.END, f"    - F1 Score: {class_metrics['f1']:.4f}\n")
                
            if report_file:
                self.summary_text.insert(tk.END, f"\nDetailed report saved to: {report_file}\n")
        
        # Update details tree
        self.results_tree.delete(*self.results_tree.get_children())
        
        for i, result in enumerate(results):
            # Get values for columns
            req_id = result.get('id', str(i+1))
            req_text = result.get('text', '')
            if len(req_text) > 100:
                req_text = req_text[:97] + '...'
            
            predicted = result.get('predicted_status', '')
            confidence = f"{result.get('prediction_confidence', 0):.2f}"
            
            # Get actual status if available
            actual = result.get('supplier_status', '')
            
            # Add to tree
            item_id = self.results_tree.insert('', 'end', values=(req_id, req_text, predicted, confidence, actual))
            
            # Color code based on match between predicted and actual
            if actual and actual.lower() == predicted.lower():
                self.results_tree.item(item_id, tags=('match',))
            elif actual:
                self.results_tree.item(item_id, tags=('mismatch',))
        
        # Configure tag colors
        self.results_tree.tag_configure('match', background='#d0ffd0')
        self.results_tree.tag_configure('mismatch', background='#ffd0d0')
    
    def handle_batch_error(self, error_message):
        """Handle errors in batch prediction"""
        self.batch_progress.stop()
        self.batch_status.config(text=f"Error: {error_message}")
        messagebox.showerror("Batch Prediction Error", error_message)
