"""
GUI for the Optimal Control plugin.
"""
from PySide2.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QLabel, QLineEdit, QTextEdit, 
                              QGroupBox, QFileDialog)
from PySide2.QtCore import Signal, Slot, Qt
from qudi.core.module import GuiBase
from qudi.core.connector import Connector

class OptimizationGUI(GuiBase, QMainWindow):
    """
    GUI class for the Optimal Control plugin,
    integrating Qudi's GuiBase and a QMainWindow.
    """

    # Define connectors to the Logic modules
    optimization_logic = Connector(interface="OptimizationLogic")

    # Define signals
    optimization_started_signal = Signal()
    optimization_stopped_signal = Signal()
    
    def __init__(self, config=None, **kwargs):
        # Initialize the parent classes with a single super() call
        super().__init__(config=config, **kwargs)
        self.setWindowTitle("Optimal Control")
        
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
        title_label = QLabel("Optimal Control Interface")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        info_label = QLabel("Simplified interface for optimal control experiments.")
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)
        
        # Configuration section
        config_group = QGroupBox("Optimization Configuration")
        config_layout = QVBoxLayout()
        config_group.setLayout(config_layout)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Dictionary File:")
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        self.browse_button = QPushButton("Browse")
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(self.browse_button)
        config_layout.addLayout(file_layout)
        
        # Status display
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status:")
        self.status_value = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_value)
        config_layout.addLayout(status_layout)
        
        main_layout.addWidget(config_group)
        
        # Log area
        log_group = QGroupBox("Optimization Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Optimization")
        self.stop_button = QPushButton("Stop Optimization")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)
        
        # Set size
        self.resize(800, 600)
    
    def _connect_internal_signals(self):
        """Connect button signals to slots."""
        self.browse_button.clicked.connect(self._on_browse_clicked)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        
        # Connect internal state change signals
        self.optimization_started_signal.connect(self._on_optimization_started)
        self.optimization_stopped_signal.connect(self._on_optimization_stopped)
    
    def _on_browse_clicked(self):
        """Handle Browse button click to select optimization dictionary file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Optimization Dictionary File", "", "JSON Files (*.json);;YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if file_path:
            self.file_path.setText(file_path)
            self.log_message(f"Selected file: {file_path}")
    
    def _on_start_clicked(self):
        """Handle Start button click."""
        if not self.file_path.text():
            self.log_message("Please select a dictionary file first!")
            return
        
        try:
            self.log_message("Starting optimization process...")
            self.optimization_started_signal.emit()
            # In a real implementation, we would call the optimization logic here
            # self.optimizationlogic.start_optimization(...)
        except Exception as e:
            self.log_message(f"Error starting optimization: {str(e)}")
            self.optimization_stopped_signal.emit()
    
    def _on_stop_clicked(self):
        """Handle Stop button click."""
        self.log_message("Stopping optimization process...")
        # In a real implementation, we would signal the optimization logic to stop
        self.optimization_stopped_signal.emit()
    
    def _on_optimization_started(self):
        """Update UI state when optimization starts."""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.browse_button.setEnabled(False)
        self.status_value.setText("Running")
    
    def _on_optimization_stopped(self):
        """Update UI state when optimization stops."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.browse_button.setEnabled(True)
        self.status_value.setText("Ready")
    
    def log_message(self, message):
        """Add a message to the log text area."""
        self.log_text.append(message)
    
    def update_optimization_dictionary(self, optimization_dict):
        """Update the optimization dictionary.
        
        This is a stub for the method expected by the original OptimizationLogic.
        """
        self.log_message(f"Received optimization dictionary with {len(optimization_dict)} entries")
    
    def handle_ui_elements(self):
        """Handle UI elements.
        
        This is a stub for the method expected by the original OptimizationLogic.
        """
        pass
    
    def on_activate(self):
        """Module activation in Qudi."""
        # Get connected logic module
        self.optimizationlogic = self.optimization_logic()
        
        try:
            # Connect signals if the logic module provides them
            self.optimizationlogic.load_optimization_dictionary_signal.connect(
                self.update_optimization_dictionary
            )
            self.log_message("Successfully connected to optimization logic")
        except Exception as e:
            self.log_message(f"Warning: Could not connect signals: {str(e)}")
        
        # Initialize UI elements
        self.handle_ui_elements()
        return 0

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