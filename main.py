import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as fd
from options.base_options import BaseOptions
import os
import sys
import pandas as pd
# Experiment scripts
from scripts_experiments.train_GNN import train_network_nested_cv
from scripts_experiments.train_TML import train_tml_model_nested_cv
from scripts_experiments.predict_test import predict_final_test
from scripts_experiments.compare_gnn_tml import plot_results
from scripts_experiments.explain_gnn import denoise_graphs, GNNExplainer_node_feats, shapley_analysis

from icecream import ic



class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Run All Experiments")
        self.geometry("500x400")
        self.configure(bg='#f0f0f0')

        # Create a style
        self.style = ttk.Style(self)
        self.style.configure('TCheckbutton', font=('Avenir Next', 14))
        self.style.configure('TButton', font=('Avenir Next', 14), padding=10)
        self.style.configure('TLabel', font=('Avenir Next', 16), background='#f0f0f0')

        self.opt = BaseOptions().parse()

        # Create the variables
        self.train_GNN_var = tk.BooleanVar(value=False)
        self.train_tml_var = tk.BooleanVar(value=False)
        self.predict_unseen_var = tk.BooleanVar(value=False)
        self.compare_models_var = tk.BooleanVar(value=False)
        self.denoise_graph_var = tk.BooleanVar(value=False)
        self.GNNExplainer_var = tk.BooleanVar(value=False)
        self.shapley_analysis_var = tk.BooleanVar(value=False)

        # train_GNN variables
        self.data_entry = None
        self.log_dir_entry = None
        self.log_dir_name_entry = None

        self.smiles_cols_GNN = {}

        # train_tml variables
        self.data_entry_tml = None
        self.log_dir_entry_tml = None
        self.log_dir_name_entry_tml = None
        self.train_algorithm_tml = None

        # Denoise variables
        self.denoise_reactions = None  # To store the denoise reactions value
        self.denoise_mol = None  # To store the denoise molecule value
        self.denoise_based_on = None  # To store the denoise based on value
        self.denoise_outer = None  # To store the outer value
        self.denoise_inner = None # To store the inner value
        self.denoise_norm = None  # To store the normalize value

        # GNNExplainer variables
        self.explain_reactions = None
        self.explain_mol = None
        self.explain_based_on = None
        self.explain_outer = None
        self.explain_inner = None
        self.explain_norm = None

        # Shap variables
        self.shap_reactions = None  # To store the shap reactions value
        self.shap_mol = None
        self.shap_based_on = None
        self.shap_outer = None
        self.shap_inner = None
        self.shap_norm = None

        self.tml_algorithm_dict = {'Random Forest': 'rf',
                                   'Gradient Boosting': 'gb',
                                   'Linear Regression': 'lr'}
        

        self.denoise_mol_dict = {'Ligand': 'ligand',
                                    'Substrate': 'substrate',
                                    'Organoboron Reagent': 'boron',
                                    'All': None}
        
        self.denoise_based_on_dict = {'All': None,
                                      'Atom Identity': 'atom_identity',
                                        'Degree': 'degree',
                                        'Hybridization': 'hyb',
                                        'Aromaticity': 'aromatic',
                                        'Atom in Ring': 'ring',
                                        'Chirality': 'chiral',
                                        'Ligand Configuration': 'conf'}


        self.create_widgets()

    def create_widgets(self):
        # Title label
        title_label = ttk.Label(self, text="Run Experiments", style='TLabel')
        title_label.pack(pady=10)

        # Description label
        description_label = ttk.Label(self, text="Select the operations to run:", style='TLabel')
        description_label.pack(pady=5)

        # Checkbuttons
        options_frame = ttk.Frame(self)
        options_frame.pack(pady=10)

        train_GNN_checkbutton = ttk.Checkbutton(options_frame, text="Train GNN", variable=self.train_GNN_var, command=self.open_train_GNN_window)
        train_GNN_checkbutton.grid(column=0, row=0, sticky=tk.W, padx=10, pady=5)

        train_TML_checkbutton = ttk.Checkbutton(options_frame, text="Train TML Model", variable=self.train_tml_var, command=self.open_train_tml_window)
        train_TML_checkbutton.grid(column=0, row=1, sticky=tk.W, padx=10, pady=5)

        ttk.Checkbutton(options_frame, text="Predict Unseen Data", variable=self.predict_unseen_var).grid(column=0, row=2, sticky=tk.W, padx=10, pady=5)
        ttk.Checkbutton(options_frame, text="Compare Models", variable=self.compare_models_var).grid(column=0, row=3, sticky=tk.W, padx=10, pady=5)

        denoise_checkbutton = ttk.Checkbutton(options_frame, text="Denoise Graph", variable=self.denoise_graph_var, command=self.open_denoise_window)
        denoise_checkbutton.grid(column=0, row=4, sticky=tk.W, padx=10, pady=5)

        GNNExp_checkbutton = ttk.Checkbutton(options_frame, text="Run GNNExplainer", variable=self.GNNExplainer_var, command=self.open_GNNEx_window)
        GNNExp_checkbutton.grid(column=0, row=5, sticky=tk.W, padx=10, pady=5)

        shap_checkbutton = ttk.Checkbutton(options_frame, text="Shapley Analysis", variable=self.shapley_analysis_var, command=self.open_shap_window)
        shap_checkbutton.grid(column=0, row=6, sticky=tk.W, padx=10, pady=5)

        # Run button
        run_button = ttk.Button(self, text="Run", command=self.run_experiments)
        run_button.pack(pady=20)


    def browse_file(self):
        file_path = fd.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.data_entry.delete(0, tk.END)
            self.data_entry.insert(0, file_path)


    def populate_columns_listbox(self, file_path):
        df = pd.read_csv(file_path)
        self.columns_listbox.delete(0, tk.END)
        for col in df.columns:
            self.columns_listbox.insert(tk.END, col)

    def show_column_selection(self,):
        selected_file = self.data_entry.get()
        if selected_file:
            df = pd.read_csv(selected_file)
            column_names = list(df.columns)

            # Create a scrollable frame for checkboxes
            column_selection_window = tk.Toplevel(self)
            column_selection_window.title("Select Columns")
            column_selection_window.geometry("300x400")
            column_selection_window.configure(bg='#f0f0f0')

            # Create a frame to hold checkboxes
            scrollbar_frame = tk.Frame(column_selection_window)
            scrollbar_frame.pack(fill="both", expand=True)

            # Create a vertical scrollbar
            scrollbar = tk.Scrollbar(scrollbar_frame, orient="vertical")
            scrollbar.pack(side="right", fill="y")

            # Create the canvas for scrollable content
            scrollable_canvas = tk.Canvas(scrollbar_frame, yscrollcommand=scrollbar.set)
            scrollable_canvas.pack(side="left", fill="both", expand=True)

            # Inner frame to hold checkboxes (placed on the canvas)
            inner_frame = tk.Frame(scrollable_canvas, bg='#f0f0f0')
            scrollable_canvas.create_window((0, 0), window=inner_frame, anchor='nw')

            self.col_vars_dict = {}
            for col in column_names:
                var = tk.BooleanVar()
                chk = tk.Checkbutton(inner_frame, text=col, variable=var, font=('Avenir Next', 12), bg='#f0f0f0')
                chk.pack(anchor='w')
                self.col_vars_dict[col] = var

            # Apply button (remains outside the scrollable area)
            apply_button = ttk.Button(column_selection_window, text="Apply", command=column_selection_window.destroy)
            apply_button.pack(pady=10)

            # Configure scrollbar with canvas content
            scrollbar.config(command=scrollable_canvas.yview)
            inner_frame.bind("<Configure>", lambda e: scrollable_canvas.configure(scrollregion=scrollable_canvas.bbox("all")))

            # Bind mouse wheel event for scrolling
            scrollable_canvas.bind("<MouseWheel>", self._on_mouse_wheel)

        def _on_mouse_wheel(self, event):
            # Adjust scroll amount based on event.delta
            scrollable_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


    def browse_directory(self):
        dir_path = fd.askdirectory()
        if dir_path:
            self.log_dir_entry.delete(0, tk.END)
            self.log_dir_entry.insert(0, dir_path)


    def clear_placeholder(self, event, entry, placeholder):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg='black')

    def add_placeholder(self, event, entry, placeholder):
        if entry.get() == "":
            entry.insert(0, placeholder)
            entry.config(fg='grey')

    def open_train_GNN_window(self):
        if self.train_GNN_var.get():
            # Create a new window
            train_GNN_window = tk.Toplevel(self)
            train_GNN_window.title("Train GNN Options")
            train_GNN_window.geometry("600x650")
            train_GNN_window.configure(bg='#f0f0f0')

            # Select csv file with the data
            label_data = ttk.Label(train_GNN_window, text="Select the csv file with the data:", style='TLabel')
            label_data.pack(pady=10)

            self.data_entry = ttk.Entry(train_GNN_window, font=('Avenir Next', 14), width=50)
            self.data_entry.pack(pady=5)

            browse_button = ttk.Button(train_GNN_window, text="Browse files", command=self.browse_file)
            browse_button.pack(pady=5)

            # Placeholder for columns selection dropdown
            self.columns_button = ttk.Button(train_GNN_window, text="Select SMILES Columns", command=self.show_column_selection)
            self.columns_button.pack(pady=10)

            # Log directory
            log_dir_label = ttk.Label(train_GNN_window, text="Log directory:", style='TLabel')
            log_dir_label.pack(pady=10)

            self.log_dir_entry = ttk.Entry(train_GNN_window, font=('Avenir Next', 14), width=50)
            self.log_dir_entry.pack(pady=5)

            log_dir_button = ttk.Button(train_GNN_window, text="Browse", command=self.browse_directory)
            log_dir_button.pack(pady=5)

            # Log directory name
            log_dir_name_label = ttk.Label(train_GNN_window, text="Log directory name:", style='TLabel')
            log_dir_name_label.pack(pady=10)

            self.log_dir_name_entry = ttk.Entry(train_GNN_window, font=('Avenir Next', 14), width=50)
            self.log_dir_name_entry.pack(pady=5)

            # Training Options button
            training_options_button = ttk.Button(train_GNN_window, text="Customize training Options", command=self.open_training_options_window)
            training_options_button.pack(pady=10)

            # Apply button
            apply_button = ttk.Button(train_GNN_window, text="Apply", command=lambda: self.apply_GNN_train_options(train_GNN_window))
            apply_button.pack(pady=10)


    def open_training_options_window(self):

        # Create a second window for training options
        training_options_window = tk.Toplevel(self)
        training_options_window.title("Training Options")
        training_options_window.geometry("400x400")
        training_options_window.configure(bg='#f0f0f0')

        # Example content for the training options window
        label_options = ttk.Label(training_options_window, text="Training Options:", style='TLabel')
        label_options.pack(pady=10)

        # emb size entry 
        emb_label = ttk.Label(training_options_window, text="Embedding size:", style='TLabel')
        emb_label.pack(pady=5)
        default_emb_size = str(self.opt.embedding_dim)
        self.emb_size_entry = tk.Entry(training_options_window, font=('Avenir Next', 14), fg='grey')
        self.emb_size_entry.pack(pady=5)
        self.emb_size_entry.insert(0, default_emb_size)
        self.emb_size_entry.bind("<FocusIn>", lambda event: self.clear_placeholder(event, self.emb_size_entry, default_emb_size))
        self.emb_size_entry.bind("<FocusOut>", lambda event: self.add_placeholder(event, self.emb_size_entry, default_emb_size))

        # num_convs entry 
        convs_label = ttk.Label(training_options_window, text="Number of convolutions:", style='TLabel')
        convs_label.pack(pady=5)
        default_n_conv = str(self.opt.n_convolutions)
        self.num_conv_entry = tk.Entry(training_options_window, font=('Avenir Next', 14), fg='grey')
        self.num_conv_entry.pack(pady=5)
        self.num_conv_entry.insert(0, default_n_conv)
        self.num_conv_entry.bind("<FocusIn>", lambda event: self.clear_placeholder(event, self.num_conv_entry, default_n_conv))
        self.num_conv_entry.bind("<FocusOut>", lambda event: self.add_placeholder(event, self.num_conv_entry, default_n_conv))

        # num redaout entry 
        readout_label = ttk.Label(training_options_window, text="Number of readout layers:", style='TLabel')
        readout_label.pack(pady=5)
        default_readout = str(self.opt.readout_layers)
        self.readout_entry = tk.Entry(training_options_window, font=('Avenir Next', 14), fg='grey')
        self.readout_entry.pack(pady=5)
        self.readout_entry.insert(0, default_readout)
        self.readout_entry.bind("<FocusIn>", lambda event: self.clear_placeholder(event, self.readout_entry, default_readout))
        self.readout_entry.bind("<FocusOut>", lambda event: self.add_placeholder(event, self.readout_entry, default_readout))

        # epochs entry 
        epochs_label = ttk.Label(training_options_window, text="Number of epochs:", style='TLabel')
        epochs_label.pack(pady=5)
        default_epochs = str(self.opt.epochs)
        self.epochs_entry = tk.Entry(training_options_window, font=('Avenir Next', 14), fg='grey')
        self.epochs_entry.pack(pady=5)
        self.epochs_entry.insert(0, default_epochs)
        self.epochs_entry.bind("<FocusIn>", lambda event: self.clear_placeholder(event, self.epochs_entry, default_epochs))
        self.epochs_entry.bind("<FocusOut>", lambda event: self.add_placeholder(event, self.epochs_entry, default_epochs))

        # batch size entry 
        batch_label = ttk.Label(training_options_window, text="Training Batch Size:", style='TLabel')
        batch_label.pack(pady=5)
        default_batch = str(self.opt.batch_size)
        self.batch_entry = tk.Entry(training_options_window, font=('Avenir Next', 14), fg='grey')
        self.batch_entry.pack(pady=5)
        self.batch_entry.insert(0, default_batch)
        self.batch_entry.bind("<FocusIn>", lambda event: self.clear_placeholder(event, self.batch_entry, default_batch))
        self.batch_entry.bind("<FocusOut>", lambda event: self.add_placeholder(event, self.batch_entry, default_batch))


        # Example button to apply options
        apply_options_button = ttk.Button(training_options_window, text="Apply Options", command=lambda: self.apply_training_options(training_options_window))
        apply_options_button.pack(pady=10)

    def apply_training_options(self, window):
        # Implement logic to apply the training options here
        try:
            self.opt.embedding_dim = self.emb_size_entry.get()
            self.opt.n_convolutions = self.num_conv_entry.get()
            self.opt.readout_layers = self.readout_entry.get()
            self.opt.epochs = self.epochs_entry.get()
            self.opt.batch_size = self.batch_entry.get()
        except ValueError:
            self.num_epochs = None
        window.destroy()


    def apply_GNN_train_options(self, window):
        try:
            self.data_entry = self.data_entry.get()
            self.log_dir_entry = self.log_dir_entry.get()
            self.log_dir_name_entry = self.log_dir_name_entry.get()
            self.opt.mol_cols = [col for col, var in self.col_vars_dict.items() if var.get()]

        except ValueError:
            self.denoise_reactions = None
        window.destroy()


    def open_train_tml_window(self):
        if self.train_tml_var.get():
            # Create a new window
            train_tml_window = tk.Toplevel(self)
            train_tml_window.title("Train TML Options")
            train_tml_window.geometry("600x650")
            train_tml_window.configure(bg='#f0f0f0')

            # Select csv file with the data
            label_data = ttk.Label(train_tml_window, text="Select the csv file with the data:", style='TLabel')
            label_data.pack(pady=10)

            self.data_entry = ttk.Entry(train_tml_window, font=('Avenir Next', 14), width=50)
            self.data_entry.pack(pady=5)

            browse_button = ttk.Button(train_tml_window, text="Browse", command=self.browse_file)
            browse_button.pack(pady=5)

            # Log directory
            log_dir_label = ttk.Label(train_tml_window, text="Log directory:", style='TLabel')
            log_dir_label.pack(pady=10)

            self.log_dir_entry = ttk.Entry(train_tml_window, font=('Avenir Next', 14), width=50)
            self.log_dir_entry.pack(pady=5)

            log_dir_button = ttk.Button(train_tml_window, text="Browse", command=self.browse_directory)
            log_dir_button.pack(pady=5)

            # Log directory name
            log_dir_name_label = ttk.Label(train_tml_window, text="Log directory name:", style='TLabel')
            log_dir_name_label.pack(pady=10)

            self.log_dir_name_entry = ttk.Entry(train_tml_window, font=('Avenir Next', 14), width=50)
            self.log_dir_name_entry.pack(pady=5)

            # Denoise molecule label
            label_choose_tml = ttk.Label(train_tml_window, text="Choose the tml algorithm to train:", style='TLabel')
            label_choose_tml.pack(pady=10)

            # Radio button options for denoise molecule
            self.train_tml_algorithm = tk.StringVar()

            options = ['Gradient Boosting', 'Random Forest', 'Linear Regression']

            for option in options:
                rb = ttk.Radiobutton(train_tml_window, text=option, variable=self.train_tml_algorithm, value=option)
                rb.pack(anchor='center', padx=20)

            # Apply button
            apply_button = ttk.Button(train_tml_window, text="Apply", command=lambda: self.apply_tml_train_options(train_tml_window))
            apply_button.pack(pady=10)

    def apply_tml_train_options(self, window):
        try:
            self.data_entry_tml = self.data_entry.get()
            self.log_dir_entry_tml = self.log_dir_entry.get()
            self.log_dir_name_entry_tml = self.log_dir_name_entry.get()
            self.train_algorithm_tml = self.tml_algorithm_dict[self.train_tml_algorithm.get()]

        except ValueError:
            self.denoise_reactions = None
        window.destroy()


    def open_denoise_window(self):

        if self.denoise_graph_var.get():

            # Create a new window
            denoise_window = tk.Toplevel(self)
            denoise_window.title("Denoise Graph Options")
            denoise_window.geometry("400x650")
            denoise_window.configure(bg='#f0f0f0')

            # Denoise molecule label
            label_denoise_molecule = ttk.Label(denoise_window, text="Denoise molecule:", style='TLabel')
            label_denoise_molecule.pack(pady=10)

            # Radio button options for denoise molecule
            self.denoise_molecule_var = tk.StringVar()

            options = ['Ligand', 'Substrate', 'Organoboron Reagent']

            for option in options:
                rb = ttk.Radiobutton(denoise_window, text=option, variable=self.denoise_molecule_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)

            # Radio button options for denoise molecule
            self.denoise_based_on_var = tk.StringVar()

            options = ['All', 'Atom Identity', 'Degree', 'Hybridization', 'Aromaticity', 'Atom in Ring', 'Chirality', 'Ligand Configuration']

            for option in options:
                rb = ttk.Radiobutton(denoise_window, text=option, variable=self.denoise_based_on_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)

            # Denoise reaction number entry
            label_denoise_reaction = ttk.Label(denoise_window, text="Denoise reaction number:", style='TLabel')
            label_denoise_reaction.pack()

            self.denoise_reaction_entry = ttk.Entry(denoise_window, font=('Avenir Next', 14))
            self.denoise_reaction_entry.pack(pady=5)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)

            label_explain_outer = ttk.Label(denoise_window, text="Explain outer:", style='TLabel')
            label_explain_outer.pack()

            self.denoise_outer = ttk.Entry(denoise_window, font=('Avenir Next', 14))
            self.denoise_outer.pack(pady=5)

            label_explain_inner = ttk.Label(denoise_window, text="Explain inner:", style='TLabel')
            label_explain_inner.pack()

            self.denoise_inner = ttk.Entry(denoise_window, font=('Avenir Next', 14))
            self.denoise_inner.pack(pady=5)

            # Spacer
            ttk.Separator(denoise_window, orient='horizontal').pack(fill='x', pady=10)


            # Checkbox for True/False
            self.den_checkbox_norm = tk.BooleanVar()
            self.den_checkbox_norm.set(False)  # Initial value

            # Normalize checkbox
            norm_checkbox = ttk.Checkbutton(denoise_window, text="Normalize attribution scores", variable=self.den_checkbox_norm)
            
            norm_checkbox.pack(pady=10)

            # Apply button
            apply_button = ttk.Button(denoise_window, text="Apply", command=lambda: self.apply_denoise_options(denoise_window))
            apply_button.pack(pady=10)

    def apply_denoise_options(self, window):
        try:
            self.denoise_reactions = int(self.denoise_reaction_entry.get())
            self.denoise_mol = self.denoise_mol_dict[self.denoise_molecule_var.get()]
            self.denoise_based_on = self.denoise_based_on_dict[self.denoise_based_on_var.get()]
            self.outer = int(self.denoise_outer.get())
            self.inner = int(self.denoise_inner.get())
            self.norm = self.den_checkbox_norm.get()

        except ValueError:
            self.denoise_reactions = None
        window.destroy()


    def open_GNNEx_window(self):
        if self.GNNExplainer_var.get():
            gnnex_window = tk.Toplevel(self)
            gnnex_window.title("GNNExplainer Options")
            gnnex_window.geometry("400x650")
            gnnex_window.configure(bg='#f0f0f0')

            # Denoise molecule label
            label_explain_molecule = ttk.Label(gnnex_window, text="Analyze Molecule:", style='TLabel')
            label_explain_molecule.pack(pady=10)

            # Radio button options for denoise molecule
            self.explain_molecule_var = tk.StringVar()

            options = ['All', 'Ligand', 'Substrate', 'Organoboron Reagent']

            for option in options:
                rb = ttk.Radiobutton(gnnex_window, text=option, variable=self.explain_molecule_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(gnnex_window, orient='horizontal').pack(fill='x', pady=10)

            # Radio button options for denoise molecule
            self.explain_feature = tk.StringVar()

            options = ['All', 'Atom Identity', 'Degree', 'Hybridization', 'Aromaticity', 'Atom in Ring', 'Chirality', 'Ligand Configuration']

            for option in options:
                rb = ttk.Radiobutton(gnnex_window, text=option, variable=self.explain_feature, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(gnnex_window, orient='horizontal').pack(fill='x', pady=10)

            # Denoise reaction number entry
            label_explain_reaction = ttk.Label(gnnex_window, text="Analyze reaction number:", style='TLabel')
            label_explain_reaction.pack()

            self.explain_reaction_entry = ttk.Entry(gnnex_window, font=('Avenir Next', 14))
            self.explain_reaction_entry.pack(pady=5)


            # Spacer
            ttk.Separator(gnnex_window, orient='horizontal').pack(fill='x', pady=10)

            label_explain_outer = ttk.Label(gnnex_window, text="Explain outer:", style='TLabel')
            label_explain_outer.pack()

            self.explain_outer_entry = ttk.Entry(gnnex_window, font=('Avenir Next', 14))
            self.explain_outer_entry.pack(pady=5)

            label_explain_inner = ttk.Label(gnnex_window, text="Explain inner:", style='TLabel')
            label_explain_inner.pack()

            self.explain_inner_entry = ttk.Entry(gnnex_window, font=('Avenir Next', 14))
            self.explain_inner_entry.pack(pady=5)

            # Spacer
            ttk.Separator(gnnex_window, orient='horizontal').pack(fill='x', pady=10)


            # Checkbox for True/False
            self.exp_checkbox_norm = tk.BooleanVar()
            self.exp_checkbox_norm.set(False)  # Initial value

            # Normalize checkbox
            norm_checkbox = ttk.Checkbutton(gnnex_window, text="Normalize attribution scores", variable=self.exp_checkbox_norm)
            
            norm_checkbox.pack(pady=10)

            # Apply button
            apply_button = ttk.Button(gnnex_window, text="Apply", command=lambda: self.apply_GNNExp_options(gnnex_window))
            apply_button.pack(pady=10)


    def apply_GNNExp_options(self, window):
        try:
            self.explain_reactions = int(self.explain_reaction_entry.get())
            self.explain_mol = self.denoise_mol_dict[self.explain_molecule_var.get()]
            self.explain_based_on = self.denoise_based_on_dict[self.explain_feature.get()]
            self.explain_outer = int(self.explain_outer_entry.get())
            self.explain_inner = int(self.explain_inner_entry.get())
            self.explain_norm = self.exp_checkbox_norm.get()

        except ValueError:
            self.denoise_reactions = None
        window.destroy()

    def open_shap_window(self):
        if self.shapley_analysis_var.get():
            shap_window = tk.Toplevel(self)
            shap_window.title("Shap Value Sampling Options")
            shap_window.geometry("400x650")
            shap_window.configure(bg='#f0f0f0')

            # Denoise molecule label
            label_denoise_molecule = ttk.Label(shap_window, text="Analyze Molecule:", style='TLabel')
            label_denoise_molecule.pack(pady=10)

            # Radio button options for denoise molecule
            self.shap_molecule_var = tk.StringVar()

            options = ['Ligand', 'Substrate', 'Organoboron Reagent']

            for option in options:
                rb = ttk.Radiobutton(shap_window, text=option, variable=self.shap_molecule_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(shap_window, orient='horizontal').pack(fill='x', pady=10)

            # Radio button options for denoise molecule
            self.shap_based_on_var = tk.StringVar()

            options = ['All', 'Atom Identity', 'Degree', 'Hybridization', 'Aromaticity', 'Atom in Ring', 'Chirality', 'Ligand Configuration']

            for option in options:
                rb = ttk.Radiobutton(shap_window, text=option, variable=self.shap_based_on_var, value=option)
                rb.pack(anchor='center', padx=20)

            # Spacer
            ttk.Separator(shap_window, orient='horizontal').pack(fill='x', pady=10)

            # Denoise reaction number entry
            label_denoise_reaction = ttk.Label(shap_window, text="Analyze reaction number:", style='TLabel')
            label_denoise_reaction.pack()

            self.shap_reaction_entry = ttk.Entry(shap_window, font=('Avenir Next', 14))
            self.shap_reaction_entry.pack(pady=5)

            # Spacer
            ttk.Separator(shap_window, orient='horizontal').pack(fill='x', pady=10)

            label_explain_outer = ttk.Label(shap_window, text="Explain outer:", style='TLabel')
            label_explain_outer.pack()

            self.shap_outer_entry = ttk.Entry(shap_window, font=('Avenir Next', 14))
            self.shap_outer_entry.pack(pady=5)

            label_explain_inner = ttk.Label(shap_window, text="Explain inner:", style='TLabel')
            label_explain_inner.pack()

            self.shap_inner_entry = ttk.Entry(shap_window, font=('Avenir Next', 14))
            self.shap_inner_entry.pack(pady=5)

            # Spacer
            ttk.Separator(shap_window, orient='horizontal').pack(fill='x', pady=10)

            # Checkbox for True/False
            self.shap_checkbox_norm = tk.BooleanVar()
            self.shap_checkbox_norm.set(False)  # Initial value

            # Normalize checkbox
            norm_checkbox = ttk.Checkbutton(shap_window, text="Normalize attribution scores", variable=self.shap_checkbox_norm)
            
            norm_checkbox.pack(pady=10)

            # Apply button
            apply_button = ttk.Button(shap_window, text="Apply", command=lambda: self.apply_shap_options(shap_window))
            apply_button.pack(pady=10)


    def apply_shap_options(self, window):
        try:
            self.shap_reactions = int(self.shap_reaction_entry.get())
            self.shap_mol = self.denoise_mol_dict[self.shap_molecule_var.get()]
            self.shap_based_on = self.denoise_based_on_dict[self.shap_based_on_var.get()]
            self.shap_outer = int(self.shap_outer_entry.get())
            self.shap_inner = int(self.shap_inner_entry.get())
            self.shap_norm = self.shap_checkbox_norm.get()

        except ValueError:
            self.denoise_reactions = None
        window.destroy()

    def run_experiments(self):

        self.show_terminal_output()

        if self.train_GNN_var.get():
            path, filename = os.path.split(self.data_entry)
            self.opt.root = os.path.dirname(path)
            self.opt.filename = filename
            self.opt.log_dir_results = os.path.join(self.log_dir_entry, self.log_dir_name_entry)
            train_network_nested_cv(self.opt)

        if self.train_tml_var.get():
            path, filename = os.path.split(self.data_entry_tml)
            self.opt.root = os.path.dirname(path)
            self.opt.filename = filename
            self.opt.log_dir_results = os.path.join(self.log_dir_entry_tml, self.log_dir_name_entry_tml)
            self.opt.tml_algorithm = self.train_algorithm_tml
            train_tml_model_nested_cv(self.opt)

        if self.predict_unseen_var.get():
            predict_final_test()

        if self.compare_models_var.get():
            plot_results(exp_dir=os.path.join(os.getcwd(), self.opt.log_dir_results, self.opt.filename[:-4]))
            plot_results(exp_dir=os.path.join(os.getcwd(), self.opt.log_dir_results, 'final_test'))

        
        # Update opt with user-provided denoise_reactions value
        if self.denoise_graph_var.get():
            self.opt.denoise_reactions = self.denoise_reactions
            self.opt.denoise_mol = self.denoise_mol
            self.opt.denoise_based_on = self.denoise_based_on
            self.opt.explain_model = [self.outer, self.inner]
            self.opt.norm = self.norm

            # Run denoise_graphs
            denoise_graphs(self.opt, exp_path=os.path.join(os.getcwd(), self.opt.log_dir_results, self.opt.filename[:-4], 'results_GNN'))


        if self.GNNExplainer_var.get():
            # Update opt with user-provided explain_reactions value
            self.opt.explain_model = [self.explain_outer, self.explain_inner]
            # Run GNNExplainer_node_feats
            GNNExplainer_node_feats(self.opt, exp_path=os.path.join(os.getcwd(), self.opt.log_dir_results, self.opt.filename[:-4], 'results_GNN'))


        if self.shapley_analysis_var.get():
            # Update opt with user-provided shap_reactions value
            self.opt.shap_index = self.shap_reactions
            self.opt.explain_model = [self.shap_outer, self.shap_inner]
            # Run shapley_analysis
            shapley_analysis(self.opt, exp_path=os.path.join(os.getcwd(), self.opt.log_dir_results, self.opt.filename[:-4], 'results_GNN'))




    def show_terminal_output(self):
        terminal_window = tk.Toplevel(self)
        terminal_window.title("Terminal Output")
        terminal_window.geometry("600x400")

        text_widget = tk.Text(terminal_window, wrap='word', font=('Helvetica', 12))
        text_widget.pack(expand=True, fill='both')

        # Redirect stdout and stderr
        sys.stdout = RedirectText(text_widget)
        sys.stderr = RedirectText(text_widget)



class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass  # No need to implement flush for this example


if __name__ == "__main__":
    app = Application()
    app.mainloop()






