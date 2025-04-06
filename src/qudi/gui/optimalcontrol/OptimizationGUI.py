"""
GUI for the Optimal Control plugin with QUOCS integration.
"""
from PySide2.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QLabel, QLineEdit, QTextEdit, 
                              QGroupBox, QFileDialog, QComboBox, QTabWidget,
                              QDoubleSpinBox, QSpinBox, QFormLayout, QCheckBox,
                              QSplitter, QScrollArea, QToolButton, QColorDialog)
from PySide2.QtCore import Signal, Slot, Qt, QSize
from PySide2.QtGui import QColor, QPalette
import numpy as np
from qudi.core.module import GuiBase
from qudi.core.connector import Connector
import pyqtgraph as pg
from qudi.util.widgets.plotting.plot_widget import RubberbandZoomPlotWidget

class OptimizationGUI(GuiBase, QMainWindow):
    """
    GUI class for the Optimal Control plugin with QUOCS integration,
    integrating Qudi's GuiBase and a QMainWindow.
    """

    # Define connectors to the Logic modules
    optimization_logic = Connector(interface="OptimizationLogic")

    # No signals defined here to avoid Qt/PySide issues
    # We'll use direct method calls instead
    
    def __init__(self, config=None, **kwargs):
        # Initialize the parent classes with a single super() call
        super().__init__(config=config, **kwargs)
        self.setWindowTitle("QUOCS Pulse Optimization")
        
        # Flag to prevent infinite recursion between GUI and logic
        self._send_to_logic = True
        self._updating_from_signal = False
        
        # Initialize state variables
        self.optimization_params = {
            'algorithm': 'GRAPE',
            'iterations': 100,
            'pulse_count': 1,
            'sample_count': 100,
            'convergence_threshold': 1e-5,
            'target_fidelity': 0.99,
            'max_amplitude': 1.0,
            'min_amplitude': -1.0,
            'pulse_duration': 1.0,
            'smooth_penalty': 0.1
        }
        
        self.pulse_colors = [QColor(255, 0, 0), QColor(0, 0, 255), QColor(0, 128, 0)]
        
        # Create the main window
        self._create_main_window()
        
        # Connect internal signals
        self._connect_internal_signals()
    
    def _create_main_window(self):
        """Create all the UI elements."""
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Top section - Title and info
        title_label = QLabel("QUOCS Pulse Optimization")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        info_label = QLabel("Interface for optimizing pulse sequences using QUOCS")
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)
        
        # Main content splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter, 1)  # stretch factor 1
        
        # Left panel - Configuration tabs
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # Create the tabs
        tabs = QTabWidget()
        
        # Algorithm settings tab
        algorithm_tab = self._create_algorithm_tab()
        tabs.addTab(algorithm_tab, "Algorithm")
        
        # Pulse Configuration tab
        pulse_tab = self._create_pulse_tab()
        tabs.addTab(pulse_tab, "Pulse Settings")
        
        # Constraints tab
        constraints_tab = self._create_constraints_tab()
        tabs.addTab(constraints_tab, "Constraints")
        
        # Add tabs to left layout
        left_layout.addWidget(tabs)
        
        # Status display
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status:")
        self.status_value = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_value)
        left_layout.addLayout(status_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Optimization")
        self.stop_button = QPushButton("Stop Optimization")
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("Save Configuration")
        self.load_button = QPushButton("Load Configuration")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        left_layout.addLayout(button_layout)
        
        # Right panel - Visualization and Results
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # Results tabs
        result_tabs = QTabWidget()
        
        # Pulse visualization tab
        pulse_viz_widget = QWidget()
        pulse_viz_layout = QVBoxLayout()
        self.pulse_plot = pg.PlotWidget()
        self.pulse_plot.setLabel('bottom', 'Time', 's')
        self.pulse_plot.setLabel('left', 'Amplitude', '')
        pulse_viz_layout.addWidget(self.pulse_plot)
        pulse_viz_widget.setLayout(pulse_viz_layout)
        result_tabs.addTab(pulse_viz_widget, "Pulse Visualization")
        
        # Convergence plot tab
        convergence_widget = QWidget()
        convergence_layout = QVBoxLayout()
        self.convergence_plot = pg.PlotWidget()
        self.convergence_plot.setLabel('bottom', 'Iteration', '#')
        self.convergence_plot.setLabel('left', 'Figure of Merit', '')
        convergence_layout.addWidget(self.convergence_plot)
        convergence_widget.setLayout(convergence_layout)
        result_tabs.addTab(convergence_widget, "Convergence")
        
        # Log tab
        log_widget = QWidget()
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_widget.setLayout(log_layout)
        result_tabs.addTab(log_widget, "Log")
        
        right_layout.addWidget(result_tabs)
        
        # Add widgets to splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        
        # Set the default sizes (40% left, 60% right)
        main_splitter.setSizes([40, 60])
        
        # Set size
        self.resize(1200, 800)
    
    def _create_algorithm_tab(self):
        """Create the algorithm configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        # Algorithm selection
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["GRAPE", "CRAB", "dCRAB", "GROUP"])
        self.algorithm_combo.setCurrentText(self.optimization_params['algorithm'])
        form_layout.addRow("Algorithm:", self.algorithm_combo)
        
        # Iterations
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(self.optimization_params['iterations'])
        form_layout.addRow("Max Iterations:", self.iterations_spin)
        
        # Convergence threshold
        self.conv_threshold_spin = QDoubleSpinBox()
        self.conv_threshold_spin.setRange(1e-10, 1.0)
        self.conv_threshold_spin.setDecimals(10)
        self.conv_threshold_spin.setSingleStep(1e-5)
        self.conv_threshold_spin.setValue(self.optimization_params['convergence_threshold'])
        form_layout.addRow("Convergence Threshold:", self.conv_threshold_spin)
        
        # Target Fidelity
        self.target_fidelity_spin = QDoubleSpinBox()
        self.target_fidelity_spin.setRange(0.0, 1.0)
        self.target_fidelity_spin.setDecimals(4)
        self.target_fidelity_spin.setSingleStep(0.01)
        self.target_fidelity_spin.setValue(self.optimization_params['target_fidelity'])
        form_layout.addRow("Target Fidelity:", self.target_fidelity_spin)
        
        # Smoothness penalty
        self.smooth_penalty_spin = QDoubleSpinBox()
        self.smooth_penalty_spin.setRange(0.0, 1.0)
        self.smooth_penalty_spin.setDecimals(3)
        self.smooth_penalty_spin.setSingleStep(0.05)
        self.smooth_penalty_spin.setValue(self.optimization_params['smooth_penalty'])
        form_layout.addRow("Smoothness Penalty:", self.smooth_penalty_spin)
        
        layout.addLayout(form_layout)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()
        
        # Placeholder for algorithm-specific settings
        self.advanced_form = QFormLayout()
        self.update_advanced_options()
        
        advanced_layout.addLayout(self.advanced_form)
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
        
        tab.setLayout(layout)
        return tab
    
    def update_advanced_options(self):
        """Update the advanced options form based on the selected algorithm."""
        # Clear existing form items
        while self.advanced_form.rowCount() > 0:
            self.advanced_form.removeRow(0)
            
        algorithm = self.algorithm_combo.currentText()
        
        # Add algorithm-specific options
        if algorithm == "GRAPE":
            # GRAPE specific options
            self.grape_learning_rate = QDoubleSpinBox()
            self.grape_learning_rate.setRange(0.001, 10.0)
            self.grape_learning_rate.setDecimals(4)
            self.grape_learning_rate.setValue(0.5)
            self.advanced_form.addRow("Learning Rate:", self.grape_learning_rate)
            
            self.grape_momentum = QDoubleSpinBox()
            self.grape_momentum.setRange(0.0, 1.0)
            self.grape_momentum.setDecimals(2)
            self.grape_momentum.setValue(0.0)
            self.advanced_form.addRow("Momentum:", self.grape_momentum)
            
        elif algorithm == "CRAB" or algorithm == "dCRAB":
            # CRAB specific options
            self.crab_basis_combo = QComboBox()
            self.crab_basis_combo.addItems(["Fourier", "Chebyshev", "Legendre"])
            self.advanced_form.addRow("Basis:", self.crab_basis_combo)
            
            self.crab_terms_spin = QSpinBox()
            self.crab_terms_spin.setRange(1, 50)
            self.crab_terms_spin.setValue(5)
            self.advanced_form.addRow("Basis Terms:", self.crab_terms_spin)
            
            if algorithm == "dCRAB":
                self.dcrab_update_freq = QSpinBox()
                self.dcrab_update_freq.setRange(5, 100)
                self.dcrab_update_freq.setValue(20)
                self.advanced_form.addRow("Update Frequency:", self.dcrab_update_freq)
                
        elif algorithm == "GROUP":
            # GROUP specific options
            self.group_krotov_checkbox = QCheckBox("Use Krotov")
            self.group_krotov_checkbox.setChecked(True)
            self.advanced_form.addRow("", self.group_krotov_checkbox)
            
            self.group_update_spin = QSpinBox()
            self.group_update_spin.setRange(1, 20)
            self.group_update_spin.setValue(5)
            self.advanced_form.addRow("Update Steps:", self.group_update_spin)
    
    def _create_pulse_tab(self):
        """Create the pulse configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        # Number of pulses
        self.pulse_count_spin = QSpinBox()
        self.pulse_count_spin.setRange(1, 10)
        self.pulse_count_spin.setValue(self.optimization_params['pulse_count'])
        form_layout.addRow("Number of Pulses:", self.pulse_count_spin)
        
        # Sample count
        self.sample_count_spin = QSpinBox()
        self.sample_count_spin.setRange(10, 10000)
        self.sample_count_spin.setValue(self.optimization_params['sample_count'])
        form_layout.addRow("Sample Points:", self.sample_count_spin)
        
        # Pulse duration
        self.pulse_duration_spin = QDoubleSpinBox()
        self.pulse_duration_spin.setRange(0.001, 1000.0)
        self.pulse_duration_spin.setDecimals(3)
        self.pulse_duration_spin.setValue(self.optimization_params['pulse_duration'])
        form_layout.addRow("Pulse Duration (s):", self.pulse_duration_spin)
        
        layout.addLayout(form_layout)
        
        # Pulse parameters section
        pulse_group = QGroupBox("Pulse Parameters")
        pulse_layout = QVBoxLayout()
        
        # Scrollable area for multiple pulse settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.pulse_params_layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        
        # Add initial pulse parameter widget
        self._update_pulse_params_widgets()
        
        pulse_layout.addWidget(scroll)
        pulse_group.setLayout(pulse_layout)
        layout.addWidget(pulse_group)
        
        # Set stretch factor to make pulse group take up remaining space
        layout.addStretch(1)
        
        tab.setLayout(layout)
        return tab
    
    def _update_pulse_params_widgets(self):
        """Update the pulse parameter widgets based on the pulse count."""
        # Clear existing widgets
        for i in reversed(range(self.pulse_params_layout.count())):
            widget = self.pulse_params_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        
        # Create widgets for each pulse
        for i in range(self.pulse_count_spin.value()):
            pulse_group = QGroupBox(f"Pulse {i+1}")
            pulse_form = QFormLayout()
            
            # Initial shape selection
            shape_combo = QComboBox()
            shape_combo.addItems(["Flat", "Gaussian", "Sine", "Square", "Custom"])
            pulse_form.addRow("Initial Shape:", shape_combo)
            
            # Color selection
            color_layout = QHBoxLayout()
            color_label = QLabel()
            color_label.setFixedSize(24, 24)
            color_label.setAutoFillBackground(True)
            
            # Set color for the label
            color = self.pulse_colors[i % len(self.pulse_colors)]
            palette = QPalette()
            palette.setColor(QPalette.Window, color)
            color_label.setPalette(palette)
            
            color_button = QToolButton()
            color_button.setText("...")
            color_button.setFixedSize(24, 24)
            color_button.clicked.connect(lambda checked, idx=i: self._on_color_select(idx))
            
            color_layout.addWidget(color_label)
            color_layout.addWidget(color_button)
            color_layout.addStretch(1)
            
            pulse_form.addRow("Color:", color_layout)
            
            # Add the pulse parameters widget
            pulse_group.setLayout(pulse_form)
            self.pulse_params_layout.addWidget(pulse_group)
        
        # Add stretch to push everything to the top
        self.pulse_params_layout.addStretch(1)
    
    def _on_color_select(self, pulse_index):
        """Handle color selection for a pulse."""
        color = QColorDialog.getColor(self.pulse_colors[pulse_index % len(self.pulse_colors)], self, f"Select Color for Pulse {pulse_index+1}")
        
        if color.isValid():
            self.pulse_colors[pulse_index % len(self.pulse_colors)] = color
            self._update_pulse_params_widgets()  # Refresh the UI
    
    def _create_constraints_tab(self):
        """Create the constraints configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        # Max amplitude
        self.max_amp_spin = QDoubleSpinBox()
        self.max_amp_spin.setRange(0.0, 100.0)
        self.max_amp_spin.setDecimals(3)
        self.max_amp_spin.setValue(self.optimization_params['max_amplitude'])
        form_layout.addRow("Maximum Amplitude:", self.max_amp_spin)
        
        # Min amplitude
        self.min_amp_spin = QDoubleSpinBox()
        self.min_amp_spin.setRange(-100.0, 0.0)
        self.min_amp_spin.setDecimals(3)
        self.min_amp_spin.setValue(self.optimization_params['min_amplitude'])
        form_layout.addRow("Minimum Amplitude:", self.min_amp_spin)
        
        # Smoothness constraints
        self.apply_smoothing = QCheckBox("Apply Smoothness Constraint")
        self.apply_smoothing.setChecked(True)
        form_layout.addRow("", self.apply_smoothing)
        
        # Rise time constraints
        self.apply_rise_time = QCheckBox("Apply Rise Time Constraints")
        self.apply_rise_time.setChecked(False)
        form_layout.addRow("", self.apply_rise_time)
        
        self.rise_time_spin = QDoubleSpinBox()
        self.rise_time_spin.setRange(0.001, 1.0)
        self.rise_time_spin.setDecimals(3)
        self.rise_time_spin.setValue(0.1)
        self.rise_time_spin.setEnabled(False)
        form_layout.addRow("Rise Time (s):", self.rise_time_spin)
        
        # Connect rise time constraint checkbox
        self.apply_rise_time.toggled.connect(self.rise_time_spin.setEnabled)
        
        layout.addLayout(form_layout)
        
        # Additional constraints section
        additional_group = QGroupBox("Additional Constraints")
        additional_layout = QVBoxLayout()
        
        # Zero boundary conditions
        self.zero_boundaries = QCheckBox("Zero Amplitude at Boundaries")
        self.zero_boundaries.setChecked(True)
        additional_layout.addWidget(self.zero_boundaries)
        
        # Spectral constraints
        self.spectral_constraints = QCheckBox("Apply Spectral Constraints")
        self.spectral_constraints.setChecked(False)
        additional_layout.addWidget(self.spectral_constraints)
        
        # Frequency range (enabled only when spectral constraints are checked)
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Frequency Range:"))
        
        self.min_freq_spin = QDoubleSpinBox()
        self.min_freq_spin.setRange(0.0, 1000.0)
        self.min_freq_spin.setSuffix(" MHz")
        self.min_freq_spin.setValue(0.0)
        self.min_freq_spin.setEnabled(False)
        
        self.max_freq_spin = QDoubleSpinBox()
        self.max_freq_spin.setRange(0.0, 1000.0)
        self.max_freq_spin.setSuffix(" MHz")
        self.max_freq_spin.setValue(100.0)
        self.max_freq_spin.setEnabled(False)
        
        freq_layout.addWidget(self.min_freq_spin)
        freq_layout.addWidget(QLabel("-"))
        freq_layout.addWidget(self.max_freq_spin)
        
        additional_layout.addLayout(freq_layout)
        
        # Connect spectral constraints checkbox
        self.spectral_constraints.toggled.connect(self.min_freq_spin.setEnabled)
        self.spectral_constraints.toggled.connect(self.max_freq_spin.setEnabled)
        
        additional_group.setLayout(additional_layout)
        layout.addWidget(additional_group)
        
        # Add stretch to push everything to the top
        layout.addStretch(1)
        
        tab.setLayout(layout)
        return tab
    
    def _connect_internal_signals(self):
        """Connect button signals to slots."""
        # Control buttons
        self.browse_button = QPushButton()  # Define for compatibility with old code
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.load_button.clicked.connect(self._on_load_clicked)
        
        # No signal connections needed - using direct method calls
        
        # Algorithm selection change
        self.algorithm_combo.currentTextChanged.connect(self.update_advanced_options)
        
        # Pulse count change
        self.pulse_count_spin.valueChanged.connect(self._update_pulse_params_widgets)
        
        # Connect all parameter changes to the update function
        self._connect_parameter_signals()
        
        # Note: Additional connections to logic module are made in on_activate()
    
    def _connect_parameter_signals(self):
        """Connect all parameter input widgets to the parameter update function."""
        # Algorithm tab
        self.algorithm_combo.currentTextChanged.connect(self._update_optimization_params)
        self.iterations_spin.valueChanged.connect(self._update_optimization_params)
        self.conv_threshold_spin.valueChanged.connect(self._update_optimization_params)
        self.target_fidelity_spin.valueChanged.connect(self._update_optimization_params)
        self.smooth_penalty_spin.valueChanged.connect(self._update_optimization_params)
        
        # Pulse tab
        self.pulse_count_spin.valueChanged.connect(self._update_optimization_params)
        self.sample_count_spin.valueChanged.connect(self._update_optimization_params)
        self.pulse_duration_spin.valueChanged.connect(self._update_optimization_params)
        
        # Constraints tab
        self.max_amp_spin.valueChanged.connect(self._update_optimization_params)
        self.min_amp_spin.valueChanged.connect(self._update_optimization_params)
        self.apply_smoothing.toggled.connect(self._update_optimization_params)
        self.apply_rise_time.toggled.connect(self._update_optimization_params)
        self.rise_time_spin.valueChanged.connect(self._update_optimization_params)
        self.zero_boundaries.toggled.connect(self._update_optimization_params)
        self.spectral_constraints.toggled.connect(self._update_optimization_params)
        self.min_freq_spin.valueChanged.connect(self._update_optimization_params)
        self.max_freq_spin.valueChanged.connect(self._update_optimization_params)
    
    def _update_optimization_params(self):
        """Update the optimization parameters based on UI inputs."""
        try:
            # Build parameter dictionary from UI elements
            params = {
                'algorithm': self.algorithm_combo.currentText(),
                'iterations': self.iterations_spin.value(),
                'pulse_count': self.pulse_count_spin.value(),
                'sample_count': self.sample_count_spin.value(),
                'convergence_threshold': self.conv_threshold_spin.value(),
                'target_fidelity': self.target_fidelity_spin.value(),
                'max_amplitude': self.max_amp_spin.value(),
                'min_amplitude': self.min_amp_spin.value(),
                'pulse_duration': self.pulse_duration_spin.value(),
                'smooth_penalty': self.smooth_penalty_spin.value(),
                'apply_smoothing': self.apply_smoothing.isChecked(),
                'apply_rise_time': self.apply_rise_time.isChecked(),
                'rise_time': self.rise_time_spin.value(),
                'zero_boundaries': self.zero_boundaries.isChecked(),
                'apply_spectral_constraints': self.spectral_constraints.isChecked(),
                'min_frequency': self.min_freq_spin.value(),
                'max_frequency': self.max_freq_spin.value()
            }
            
            # Add algorithm-specific parameters
            algorithm = params['algorithm']
            
            if algorithm == 'GRAPE' and hasattr(self, 'grape_learning_rate'):
                params.update({
                    'learning_rate': self.grape_learning_rate.value(),
                    'momentum': self.grape_momentum.value()
                })
            elif algorithm in ['CRAB', 'dCRAB'] and hasattr(self, 'crab_basis_combo'):
                params.update({
                    'basis': self.crab_basis_combo.currentText(),
                    'basis_terms': self.crab_terms_spin.value()
                })
                
                if algorithm == 'dCRAB' and hasattr(self, 'dcrab_update_freq'):
                    params.update({
                        'update_frequency': self.dcrab_update_freq.value()
                    })
            elif algorithm == 'GROUP' and hasattr(self, 'group_krotov_checkbox'):
                params.update({
                    'use_krotov': self.group_krotov_checkbox.isChecked(),
                    'update_steps': self.group_update_spin.value()
                })
            
            # Update internal parameters
            self.optimization_params = params
            
            # Emit signal for logic module if we have a logic instance
            # Only send to logic if explicitly requested (to avoid infinite recursion)
            if hasattr(self, '_send_to_logic') and self._send_to_logic and hasattr(self, 'optimizationlogic') and self.optimizationlogic is not None:
                try:
                    # Direct method call instead of signal
                    self._send_to_logic = False  # Prevent recursion
                    self.optimizationlogic.load_opti_comm_dict(self.optimization_params)
                    self.log_message(f"Updated parameters: algorithm={params['algorithm']}, iterations={params['iterations']}")
                except Exception as e:
                    self.log_message(f"Error sending parameters update: {str(e)}")
                finally:
                    self._send_to_logic = True  # Re-enable sending
            else:
                self.log_message("Parameter update complete (not sent to logic)")
                
            return params
            
        except Exception as e:
            self.log_message(f"Error updating parameters: {str(e)}")
            raise
    
    def _on_save_clicked(self):
        """Handle Save button click to save optimization configuration."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Optimization Configuration", "", "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            try:
                # In the real implementation, this would call a method in the logic to save the configuration
                self.log_message(f"Saving configuration to {file_path}")
                # self.optimizationlogic.save_configuration(file_path, self.optimization_params)
            except Exception as e:
                self.log_message(f"Error saving configuration: {str(e)}")
    
    def _on_load_clicked(self):
        """Handle Load button click to load optimization configuration."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Optimization Configuration", "", "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            try:
                # In the real implementation, this would call a method in the logic to load the configuration
                self.log_message(f"Loading configuration from {file_path}")
                # params = self.optimizationlogic.load_configuration(file_path)
                # self._update_ui_from_params(params)
            except Exception as e:
                self.log_message(f"Error loading configuration: {str(e)}")
    
    def send_params_to_logic(self):
        """Explicitly send current parameters to the logic module."""
        if hasattr(self, 'optimizationlogic') and self.optimizationlogic is not None:
            try:
                self.optimizationlogic.load_opti_comm_dict(self.optimization_params)
                self.log_message(f"Sent parameters to logic: algorithm={self.optimization_params['algorithm']}")
                return True
            except Exception as e:
                self.log_message(f"Error sending parameters to logic: {str(e)}")
                return False
        return False
    
    def _on_start_clicked(self):
        """Handle Start button click."""
        try:
            # First update UI state
            self.log_message(f"Starting {self.algorithm_combo.currentText()} optimization with {self.iterations_spin.value()} iterations...")
            
            # Update parameters
            try:
                self._update_optimization_params()  # Make sure we have the latest settings
                # Explicitly send parameters to logic
                self.send_params_to_logic()
            except Exception as e:
                self.log_message(f"Error updating parameters: {str(e)}")
                return
                
            # Update plots with initial values
            try:
                self._initialize_plots()
            except Exception as e:
                self.log_message(f"Error initializing plots: {str(e)}")
                # Continue anyway as this is non-critical
            
            # Directly update UI state
            self._on_optimization_status_changed(True)
            
            # Start the optimization in the logic module
            if hasattr(self, 'optimizationlogic') and self.optimizationlogic is not None:
                self.optimizationlogic.start_optimization(self.optimization_params)
            else:
                self.log_message("Error: Optimization logic not available")
                self._on_optimization_status_changed(False)
                
        except Exception as e:
            self.log_message(f"Error starting optimization: {str(e)}")
            # Make sure UI is reset
            self._on_optimization_status_changed(False)
    
    def _initialize_plots(self):
        """Initialize the visualization plots with empty data."""
        # Clear existing plots
        self.pulse_plot.clear()
        self.convergence_plot.clear()
        
        # Create time grid for pulse visualization
        t = np.linspace(0, self.optimization_params['pulse_duration'], 
                         self.optimization_params['sample_count'])
        
        # Add initial pulse shapes (flat lines at 0)
        for i in range(self.optimization_params['pulse_count']):
            color = self.pulse_colors[i % len(self.pulse_colors)]
            pen = pg.mkPen(color=color)
            
            # Add a flat line at 0 for each pulse
            zeros = np.zeros_like(t)
            self.pulse_plot.plot(t, zeros, pen=pen, name=f"Pulse {i+1}")
        
        # Initialize convergence plot with a single data point at 0
        self.convergence_plot.plot([0], [0], pen=pg.mkPen('r'), symbolPen='r', symbol='o', symbolSize=5, name="FoM")
    
    def _on_stop_clicked(self):
        """Handle Stop button click."""
        self.log_message("Stopping optimization process...")
        # Update UI directly
        self._on_optimization_status_changed(False)
        
        # If connected to logic, tell it to stop as well
        if hasattr(self, 'optimizationlogic') and self.optimizationlogic is not None:
            try:
                self.optimizationlogic.stop_optimization()
            except Exception as e:
                self.log_message(f"Error stopping optimization: {str(e)}")
    
    def _on_optimization_status_changed(self, is_running):
        """Update UI state when optimization status changes.
        
        Parameters
        ----------
        is_running : bool
            Whether the optimization is running or not
        """
        if is_running:
            # Optimization started
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.save_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.status_value.setText("Running")
            
            # Disable parameter input while optimization is running
            self._set_inputs_enabled(False)
        else:
            # Optimization stopped
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.save_button.setEnabled(True)
            self.load_button.setEnabled(True)
            self.status_value.setText("Ready")
            
            # Re-enable parameter input
            self._set_inputs_enabled(True)
    
    def _set_inputs_enabled(self, enabled):
        """Enable or disable all parameter input widgets."""
        # Algorithm tab
        self.algorithm_combo.setEnabled(enabled)
        self.iterations_spin.setEnabled(enabled)
        self.conv_threshold_spin.setEnabled(enabled)
        self.target_fidelity_spin.setEnabled(enabled)
        self.smooth_penalty_spin.setEnabled(enabled)
        
        # Advanced options (algorithm specific)
        if self.optimization_params['algorithm'] == 'GRAPE':
            self.grape_learning_rate.setEnabled(enabled)
            self.grape_momentum.setEnabled(enabled)
        elif self.optimization_params['algorithm'] in ['CRAB', 'dCRAB']:
            self.crab_basis_combo.setEnabled(enabled)
            self.crab_terms_spin.setEnabled(enabled)
            if self.optimization_params['algorithm'] == 'dCRAB':
                self.dcrab_update_freq.setEnabled(enabled)
        elif self.optimization_params['algorithm'] == 'GROUP':
            self.group_krotov_checkbox.setEnabled(enabled)
            self.group_update_spin.setEnabled(enabled)
        
        # Pulse tab
        self.pulse_count_spin.setEnabled(enabled)
        self.sample_count_spin.setEnabled(enabled)
        self.pulse_duration_spin.setEnabled(enabled)
        
        # Constraints tab
        self.max_amp_spin.setEnabled(enabled)
        self.min_amp_spin.setEnabled(enabled)
        self.apply_smoothing.setEnabled(enabled)
        self.apply_rise_time.setEnabled(enabled)
        self.rise_time_spin.setEnabled(enabled and self.apply_rise_time.isChecked())
        self.zero_boundaries.setEnabled(enabled)
        self.spectral_constraints.setEnabled(enabled)
        self.min_freq_spin.setEnabled(enabled and self.spectral_constraints.isChecked())
        self.max_freq_spin.setEnabled(enabled and self.spectral_constraints.isChecked())
    
    def log_message(self, message):
        """Add a message to the log text area."""
        self.log_text.append(message)
    
    def update_pulse_plot(self, pulses, timegrids):
        """Update the pulse visualization plot with new pulse data.
        
        Parameters
        ----------
        pulses : list
            List of pulse data arrays
        timegrids : list
            List of time grids for each pulse
        """
        self.pulse_plot.clear()
        
        for i, (pulse, tgrid) in enumerate(zip(pulses, timegrids)):
            color = self.pulse_colors[i % len(self.pulse_colors)]
            pen = pg.mkPen(color=color)
            
            # Plot the pulse
            self.pulse_plot.plot(tgrid, pulse, pen=pen, name=f"Pulse {i+1}")
        
        self.log_message(f"Updated pulse visualization with {len(pulses)} pulses")
    
    def update_convergence_plot(self, iterations, fom_values):
        """Update the convergence plot with optimization progress.
        
        Parameters
        ----------
        iterations : list
            List of iteration numbers
        fom_values : list
            List of figure of merit values for each iteration
        """
        self.convergence_plot.clear()
        self.convergence_plot.plot(iterations, fom_values, pen=pg.mkPen('r'), symbolPen='r', symbol='o', 
                                  symbolSize=5, name="FoM")
        
        # Log the latest value
        if len(fom_values) > 0:
            self.log_message(f"Iteration {iterations[-1]}: FoM = {fom_values[-1]:.6f}")
    
    def update_optimization_status(self, status_dict):
        """Update the optimization status display.
        
        Parameters
        ----------
        status_dict : dict
            Dictionary containing status information (iteration, fom, etc.)
        """
        # Update status text
        if 'status' in status_dict:
            self.status_value.setText(status_dict['status'])
        
        # Log detailed information
        if 'message' in status_dict:
            self.log_message(status_dict['message'])
        
        # Update convergence data if available
        if 'iterations' in status_dict and 'fom_values' in status_dict:
            self.update_convergence_plot(status_dict['iterations'], status_dict['fom_values'])
        
        # Update pulse visualization if available
        if 'pulses' in status_dict and 'timegrids' in status_dict:
            self.update_pulse_plot(status_dict['pulses'], status_dict['timegrids'])
    
    def update_optimization_dictionary(self, optimization_dict):
        """Update the optimization dictionary.
        
        Parameters
        ----------
        optimization_dict : dict
            Dictionary containing optimization configuration
        """
        # Guard against recursion
        if self._updating_from_signal:
            return
            
        self.log_message(f"Received optimization dictionary with {len(optimization_dict)} entries")
        
        # Update UI elements based on the received dictionary
        try:
            self._updating_from_signal = True  # Set the flag to prevent recursion
            self._send_to_logic = False  # Don't send changes back to logic
            self._update_ui_from_params(optimization_dict)
        except Exception as e:
            self.log_message(f"Error updating UI from dictionary: {str(e)}")
        finally:
            self._updating_from_signal = False  # Reset flag
            self._send_to_logic = True  # Re-enable sending
    
    def _update_ui_from_params(self, params):
        """Update UI elements based on parameter dictionary.
        
        Parameters
        ----------
        params : dict
            Dictionary containing parameter values
        """
        # Update algorithm tab
        if 'algorithm' in params:
            self.algorithm_combo.setCurrentText(params['algorithm'])
        
        if 'iterations' in params:
            self.iterations_spin.setValue(params['iterations'])
            
        if 'convergence_threshold' in params:
            self.conv_threshold_spin.setValue(params['convergence_threshold'])
            
        if 'target_fidelity' in params:
            self.target_fidelity_spin.setValue(params['target_fidelity'])
            
        if 'smooth_penalty' in params:
            self.smooth_penalty_spin.setValue(params['smooth_penalty'])
        
        # Update pulse tab
        if 'pulse_count' in params:
            self.pulse_count_spin.setValue(params['pulse_count'])
            
        if 'sample_count' in params:
            self.sample_count_spin.setValue(params['sample_count'])
            
        if 'pulse_duration' in params:
            self.pulse_duration_spin.setValue(params['pulse_duration'])
        
        # Update constraints tab
        if 'max_amplitude' in params:
            self.max_amp_spin.setValue(params['max_amplitude'])
            
        if 'min_amplitude' in params:
            self.min_amp_spin.setValue(params['min_amplitude'])
            
        if 'apply_smoothing' in params:
            self.apply_smoothing.setChecked(params['apply_smoothing'])
            
        if 'apply_rise_time' in params:
            self.apply_rise_time.setChecked(params['apply_rise_time'])
            
        if 'rise_time' in params:
            self.rise_time_spin.setValue(params['rise_time'])
            
        if 'zero_boundaries' in params:
            self.zero_boundaries.setChecked(params['zero_boundaries'])
            
        if 'apply_spectral_constraints' in params:
            self.spectral_constraints.setChecked(params['apply_spectral_constraints'])
            
        if 'min_frequency' in params:
            self.min_freq_spin.setValue(params['min_frequency'])
            
        if 'max_frequency' in params:
            self.max_freq_spin.setValue(params['max_frequency'])
        
        # Update algorithm-specific parameters
        if params.get('algorithm') == 'GRAPE':
            if 'learning_rate' in params:
                self.grape_learning_rate.setValue(params['learning_rate'])
                
            if 'momentum' in params:
                self.grape_momentum.setValue(params['momentum'])
                
        elif params.get('algorithm') in ['CRAB', 'dCRAB']:
            if 'basis' in params:
                self.crab_basis_combo.setCurrentText(params['basis'])
                
            if 'basis_terms' in params:
                self.crab_terms_spin.setValue(params['basis_terms'])
                
            if params.get('algorithm') == 'dCRAB' and 'update_frequency' in params:
                self.dcrab_update_freq.setValue(params['update_frequency'])
                
        elif params.get('algorithm') == 'GROUP':
            if 'use_krotov' in params:
                self.group_krotov_checkbox.setChecked(params['use_krotov'])
                
            if 'update_steps' in params:
                self.group_update_spin.setValue(params['update_steps'])
                
        # Make sure advanced options are displayed correctly
        self.update_advanced_options()
        
        # Update internal parameter dictionary
        self._update_optimization_params()
    
    def handle_ui_elements(self):
        """Handle UI elements."""
        # Update advanced options based on algorithm
        self.update_advanced_options()
        
        # Initialize plots
        self._initialize_plots()
    
    def on_activate(self):
        """Module activation in Qudi."""
        # Get connected logic module
        try:
            self.optimizationlogic = self.optimization_logic()
            
            # Connect signals if the logic module provides them - use try/except for each signal
            try:
                if hasattr(self.optimizationlogic, 'load_optimization_dictionary_signal'):
                    self.optimizationlogic.load_optimization_dictionary_signal.connect(
                        self.update_optimization_dictionary
                    )
            except Exception as e:
                self.log_message(f"Could not connect dictionary signal: {str(e)}")
                
            try:
                if hasattr(self.optimizationlogic, 'controls_update_signal'):
                    self.optimizationlogic.controls_update_signal.connect(
                        self._handle_controls_update
                    )
            except Exception as e:
                self.log_message(f"Could not connect controls signal: {str(e)}")
                
            try:
                if hasattr(self.optimizationlogic, 'fom_plot_signal'):
                    self.optimizationlogic.fom_plot_signal.connect(
                        self._handle_fom_update
                    )
            except Exception as e:
                self.log_message(f"Could not connect FoM signal: {str(e)}")
                
            try:
                if hasattr(self.optimizationlogic, 'message_label_signal'):
                    self.optimizationlogic.message_label_signal.connect(
                        self.log_message
                    )
            except Exception as e:
                self.log_message(f"Could not connect message signal: {str(e)}")
                
            try:
                # Custom connection for running status
                if hasattr(self.optimizationlogic, 'is_running_signal'):
                    self.optimizationlogic.is_running_signal.connect(
                        self._on_optimization_status_changed  # Direct method connection
                    )
            except Exception as e:
                self.log_message(f"Could not connect running signal: {str(e)}")
                
            try:
                # Custom connection for status updates
                if hasattr(self.optimizationlogic, 'optimization_status_signal'):
                    self.optimizationlogic.optimization_status_signal.connect(
                        self.update_optimization_status  # Direct method connection
                    )
            except Exception as e:
                self.log_message(f"Could not connect status signal: {str(e)}")
            
            # No need to connect our signals to logic - we'll use direct method calls
            
            self.log_message("Successfully connected to optimization logic")
        except Exception as e:
            self.log_message(f"Warning: Could not connect to logic: {str(e)}")
        
        # Initialize UI elements
        self.handle_ui_elements()
        return 0
        
    def _handle_controls_update(self, data):
        """Safely handle controls update signal
        
        Parameters
        ----------
        data : dict
            Data containing pulse information
        """
        try:
            pulses = data.get('pulses', [])
            timegrids = data.get('timegrids', [])
            self.update_pulse_plot(pulses, timegrids)
        except Exception as e:
            self.log_message(f"Error updating pulse plot: {str(e)}")
            
    def _handle_fom_update(self, data):
        """Safely handle FoM update signal
        
        Parameters
        ----------
        data : dict
            Data containing FoM information
        """
        try:
            fom_history = data.get('fom_history', [])
            initial_fom = data.get('initial_fom', 0)
            
            if fom_history:
                iterations = list(range(len(fom_history) + 1))
                values = [initial_fom] + fom_history
                self.update_convergence_plot(iterations, values)
        except Exception as e:
            self.log_message(f"Error updating convergence plot: {str(e)}")
            
    
    def _on_save_clicked_with_logic(self):
        """Handle Save button click using the logic's save method."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Optimization Configuration", "", "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)"
            )
            if file_path:
                try:
                    self._update_optimization_params()  # Make sure params are up to date
                    self.optimizationlogic.save_configuration(file_path, self.optimization_params)
                except Exception as e:
                    self.log_message(f"Error saving configuration: {str(e)}")
        except Exception as e:
            self.log_message(f"Error in save dialog: {str(e)}")
                
    def _on_load_clicked_with_logic(self):
        """Handle Load button click using the logic's load method."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Optimization Configuration", "", "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)"
            )
            if file_path:
                try:
                    params = self.optimizationlogic.load_configuration(file_path)
                    self._update_ui_from_params(params)
                except Exception as e:
                    self.log_message(f"Error loading configuration: {str(e)}")
        except Exception as e:
            self.log_message(f"Error in load dialog: {str(e)}")

    def on_deactivate(self):
        """Module deactivation in Qudi."""
        # Properly close the window when module is deactivated
        self.close()
        return 0

    def show(self):
        """Show the GUI window and bring it to the front."""
        # Don't call super().show() as this would call GuiBase.show() which raises NotImplementedError
        # Instead, call directly QMainWindow.show()
        QMainWindow.show(self)
        self.activateWindow()
        self.raise_()