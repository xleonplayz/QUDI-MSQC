"""
GUI for the NV Center Simulator module.
"""
from PySide2.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QLabel, QLineEdit, QTextEdit, 
                              QGroupBox, QFileDialog, QComboBox, QTabWidget,
                              QDoubleSpinBox, QSpinBox, QFormLayout, QCheckBox,
                              QSplitter, QScrollArea, QToolButton, QColorDialog,
                              QProgressBar, QSlider, QRadioButton, QButtonGroup)
from PySide2.QtCore import Signal, Slot, Qt, QSize
from PySide2.QtGui import QColor, QPalette
import numpy as np
from qudi.core.module import GuiBase
from qudi.core.connector import Connector
import pyqtgraph as pg
from qudi.util.widgets.plotting.plot_widget import RubberbandZoomPlotWidget
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class MatplotlibWidget(QWidget):
    """Widget to embed a Matplotlib figure in Qt."""
    
    def __init__(self, parent=None):
        """Initialize the Matplotlib widget."""
        super().__init__(parent)
        
        # Create a Matplotlib figure and canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # Add a navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Create a single axes for the figure
        self.axes = self.figure.add_subplot(111)
        
    def clear(self):
        """Clear the figure."""
        self.axes.clear()
        self.canvas.draw()
        
    def plot(self, *args, **kwargs):
        """Plot data using matplotlib's plot method."""
        self.axes.plot(*args, **kwargs)
        self.canvas.draw()
        
    def set_title(self, title):
        """Set the plot title."""
        self.axes.set_title(title)
        self.canvas.draw()
        
    def set_labels(self, xlabel=None, ylabel=None):
        """Set the x and y axis labels."""
        if xlabel is not None:
            self.axes.set_xlabel(xlabel)
        if ylabel is not None:
            self.axes.set_ylabel(ylabel)
        self.canvas.draw()
        
    def set_grid(self, grid=True):
        """Set the grid visibility."""
        self.axes.grid(grid)
        self.canvas.draw()

class NVBlochSphereWidget(QWidget):
    """Widget to display the NV center quantum state on a Bloch sphere."""
    
    def __init__(self, parent=None):
        """Initialize the Bloch sphere widget."""
        super().__init__(parent)
        
        # Create a Matplotlib figure and canvas for the Bloch sphere
        layout = QVBoxLayout()
        self.mpl_widget = MatplotlibWidget()
        layout.addWidget(self.mpl_widget)
        self.setLayout(layout)
        
        # Initialize the Bloch sphere figure
        self._init_bloch_sphere()
        
    def _init_bloch_sphere(self):
        """Initialize the Bloch sphere plot."""
        self.mpl_widget.clear()
        
        # Create the figure with a 3D projection
        self.mpl_widget.figure.clear()
        self.bloch_ax = self.mpl_widget.figure.add_subplot(111, projection='3d')
        
        # Plot the Bloch sphere surface
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = 0.98 * np.outer(np.cos(u), np.sin(v))
        y = 0.98 * np.outer(np.sin(u), np.sin(v))
        z = 0.98 * np.outer(np.ones(np.size(u)), np.cos(v))
        self.bloch_ax.plot_surface(x, y, z, color='w', alpha=0.1, linewidth=0)
        
        # Draw the coordinate system
        self.bloch_ax.quiver(-1.3, 0, 0, 2.6, 0, 0, color='k', arrow_length_ratio=0.05, linewidth=1.5)
        self.bloch_ax.quiver(0, -1.3, 0, 0, 2.6, 0, color='k', arrow_length_ratio=0.05, linewidth=1.5)
        self.bloch_ax.quiver(0, 0, -1.3, 0, 0, 2.6, color='k', arrow_length_ratio=0.05, linewidth=1.5)
        
        # Add coordinate labels
        self.bloch_ax.text(1.5, 0, 0, r'$x$', fontsize=12)
        self.bloch_ax.text(0, 1.5, 0, r'$y$', fontsize=12)
        self.bloch_ax.text(0, 0, 1.5, r'$z$', fontsize=12)
        
        # Set the plot limits and aspect ratio
        self.bloch_ax.set_xlim([-1.5, 1.5])
        self.bloch_ax.set_ylim([-1.5, 1.5])
        self.bloch_ax.set_zlim([-1.5, 1.5])
        self.bloch_ax.set_aspect('equal')
        
        # Remove ticks and labels to make the plot cleaner
        self.bloch_ax.set_xticks([])
        self.bloch_ax.set_yticks([])
        self.bloch_ax.set_zticks([])
        
        # Set the title
        self.bloch_ax.set_title('NV State Visualization')
        
        # Initialize the state vector (no states yet)
        self.state_vector = None
        self.state_line = None
        
        # Redraw the canvas
        self.mpl_widget.canvas.draw()
        
    def update_state(self, rho):
        """Update the Bloch sphere with a new quantum state.
        
        Parameters
        ----------
        rho : np.ndarray
            Density matrix of the NV center state (3x3 complex matrix)
        """
        # Calculate the Bloch vector for the state
        if rho is None or rho.shape != (3, 3):
            return
            
        # For a spin-1 system, we need to calculate the expectation values
        # of the spin operators to represent the state on the Bloch sphere
        # using the Majorana representation for higher spins
        
        # Extract the populations (probabilities for |0⟩, |+1⟩, |-1⟩)
        pops = np.real(np.diag(rho))
        
        # Calculate expectation values of spin operators
        sx = np.real(np.trace(np.dot(rho, self._get_sx())))
        sy = np.real(np.trace(np.dot(rho, self._get_sy())))
        sz = np.real(np.trace(np.dot(rho, self._get_sz())))
        
        # Normalize the Bloch vector (for visualization purposes)
        magnitude = np.sqrt(sx**2 + sy**2 + sz**2)
        if magnitude > 0:
            sx, sy, sz = sx/magnitude, sy/magnitude, sz/magnitude
        
        # Clear previous state vector if it exists
        if hasattr(self, 'state_vector') and self.state_vector is not None:
            for artist in self.state_vector:
                artist.remove()
        
        # Plot the new state vector
        self.state_vector = self.bloch_ax.quiver(0, 0, 0, sx, sy, sz, 
                                               color='r', arrow_length_ratio=0.1, 
                                               linewidth=2.0)
        
        # Create a text annotation with the state populations
        text = f"|0⟩: {pops[0]:.2f}, |+1⟩: {pops[1]:.2f}, |-1⟩: {pops[2]:.2f}"
        
        # Remove old text if it exists
        if hasattr(self, 'pop_text') and self.pop_text is not None:
            self.pop_text.remove()
        
        # Add new population text
        self.pop_text = self.bloch_ax.text(0, 0, -1.8, text, 
                                          horizontalalignment='center',
                                          fontsize=10)
        
        # Redraw the canvas
        self.mpl_widget.canvas.draw()
        
    def _get_sx(self):
        """Get the Sx operator for a spin-1 system."""
        return 1/np.sqrt(2) * np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=complex)
    
    def _get_sy(self):
        """Get the Sy operator for a spin-1 system."""
        return 1j/np.sqrt(2) * np.array([
            [0, -1, 0],
            [1, 0, -1],
            [0, 1, 0]
        ], dtype=complex)
    
    def _get_sz(self):
        """Get the Sz operator for a spin-1 system."""
        return np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, -1]
        ], dtype=complex)
        
class NVSimulatorGUI(GuiBase, QMainWindow):
    """
    GUI class for the NV Center Simulator, integrating Qudi's GuiBase and QMainWindow.
    """
    # Define connectors to the Logic module
    nv_simulator_logic = Connector(interface="NVSimulatorLogic")
    
    def __init__(self, config=None, **kwargs):
        # Initialize the parent classes with a single super() call
        super().__init__(config=config, **kwargs)
        self.setWindowTitle("NV Center Quantum Simulator")
        
        # Initialize state variables
        self.simulator_params = {
            'd_gs': 2.87e9,        # Zero-field splitting (Hz)
            'e_gs': 2e6,           # Strain parameter (Hz)
            'gamma_e': 28e6,       # Gyromagnetic ratio (Hz/mT)
            'b_field': [0, 0, 0.5], # External B field (mT)
            't1': 100e-6,          # T1 relaxation time (s)
            't2': 50e-6,           # T2 relaxation time (s)
            'noise_level': 0.05    # Measurement noise level
        }
        
        # Experiment parameters
        self.experiment_params = {
            'experiment_type': 'rabi',
            'max_time': 1e-6,      # Maximum experiment time (s)
            'steps': 100,          # Number of steps for the experiment
            'amplitude': 1.0       # Pulse amplitude (normalized)
        }
        
        # Create the main window layout
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
        title_label = QLabel("NV Center Quantum Simulator")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        main_layout.addWidget(title_label)
        
        info_label = QLabel("Interactive simulation of NV center dynamics and experiments")
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
        
        # Physical parameters tab
        physical_tab = self._create_physical_params_tab()
        tabs.addTab(physical_tab, "NV Parameters")
        
        # Experiment configuration tab
        experiment_tab = self._create_experiment_tab()
        tabs.addTab(experiment_tab, "Experiments")
        
        # Pulse sequence tab
        pulse_tab = self._create_pulse_tab()
        tabs.addTab(pulse_tab, "Pulse Sequences")
        
        # Add tabs to left layout
        left_layout.addWidget(tabs)
        
        # Status display
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status:")
        self.status_value = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_value)
        left_layout.addLayout(status_layout)
        left_layout.addWidget(self.progress_bar)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset NV State")
        self.start_button = QPushButton("Run Experiment")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        left_layout.addLayout(button_layout)
        
        # Right panel - Visualization and Results
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # Results tabs
        result_tabs = QTabWidget()
        
        # Bloch sphere tab
        self.bloch_widget = NVBlochSphereWidget()
        result_tabs.addTab(self.bloch_widget, "Bloch Sphere")
        
        # Experiment Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout()
        self.results_plot = pg.PlotWidget()
        self.results_plot.setLabel('bottom', 'Time', 's')
        self.results_plot.setLabel('left', 'Population', '')
        results_layout.addWidget(self.results_plot)
        results_widget.setLayout(results_layout)
        result_tabs.addTab(results_widget, "Experiment Results")
        
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
    
    def _create_physical_params_tab(self):
        """Create the physical parameters configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # NV Ground State Parameters
        gs_group = QGroupBox("NV Center Ground State Parameters")
        gs_form = QFormLayout()
        
        # Zero-field splitting (D)
        self.d_gs_spin = QDoubleSpinBox()
        self.d_gs_spin.setRange(2.8e9, 2.9e9)
        self.d_gs_spin.setDecimals(1)
        self.d_gs_spin.setSingleStep(1e6)
        self.d_gs_spin.setValue(self.simulator_params['d_gs'])
        self.d_gs_spin.setSuffix(" Hz")
        self.d_gs_spin.setToolTip("Zero-field splitting parameter D, typically around 2.87 GHz")
        gs_form.addRow("Zero-field Splitting (D):", self.d_gs_spin)
        
        # Strain parameter (E)
        self.e_gs_spin = QDoubleSpinBox()
        self.e_gs_spin.setRange(0, 10e6)
        self.e_gs_spin.setDecimals(2)
        self.e_gs_spin.setSingleStep(1e5)
        self.e_gs_spin.setValue(self.simulator_params['e_gs'])
        self.e_gs_spin.setSuffix(" Hz")
        self.e_gs_spin.setToolTip("Strain parameter E, typically in the MHz range")
        gs_form.addRow("Strain Parameter (E):", self.e_gs_spin)
        
        # Gyromagnetic ratio
        self.gamma_e_spin = QDoubleSpinBox()
        self.gamma_e_spin.setRange(27e6, 29e6)
        self.gamma_e_spin.setDecimals(1)
        self.gamma_e_spin.setSingleStep(0.1e6)
        self.gamma_e_spin.setValue(self.simulator_params['gamma_e'])
        self.gamma_e_spin.setSuffix(" Hz/mT")
        self.gamma_e_spin.setToolTip("Gyromagnetic ratio for the electron spin, approximately 28 MHz/mT")
        gs_form.addRow("Gyromagnetic Ratio (γ):", self.gamma_e_spin)
        
        gs_group.setLayout(gs_form)
        layout.addWidget(gs_group)
        
        # Magnetic Field Parameters
        b_field_group = QGroupBox("External Magnetic Field")
        b_field_form = QFormLayout()
        
        # B-field in x direction
        self.bx_spin = QDoubleSpinBox()
        self.bx_spin.setRange(-10, 10)
        self.bx_spin.setDecimals(3)
        self.bx_spin.setSingleStep(0.1)
        self.bx_spin.setValue(self.simulator_params['b_field'][0])
        self.bx_spin.setSuffix(" mT")
        b_field_form.addRow("B-field X:", self.bx_spin)
        
        # B-field in y direction
        self.by_spin = QDoubleSpinBox()
        self.by_spin.setRange(-10, 10)
        self.by_spin.setDecimals(3)
        self.by_spin.setSingleStep(0.1)
        self.by_spin.setValue(self.simulator_params['b_field'][1])
        self.by_spin.setSuffix(" mT")
        b_field_form.addRow("B-field Y:", self.by_spin)
        
        # B-field in z direction
        self.bz_spin = QDoubleSpinBox()
        self.bz_spin.setRange(-10, 10)
        self.bz_spin.setDecimals(3)
        self.bz_spin.setSingleStep(0.1)
        self.bz_spin.setValue(self.simulator_params['b_field'][2])
        self.bz_spin.setSuffix(" mT")
        b_field_form.addRow("B-field Z:", self.bz_spin)
        
        b_field_group.setLayout(b_field_form)
        layout.addWidget(b_field_group)
        
        # Decoherence Parameters
        decoherence_group = QGroupBox("Decoherence Parameters")
        decoherence_form = QFormLayout()
        
        # T1 relaxation time
        self.t1_spin = QDoubleSpinBox()
        self.t1_spin.setRange(1e-6, 10)
        self.t1_spin.setDecimals(6)
        self.t1_spin.setSingleStep(10e-6)
        self.t1_spin.setValue(self.simulator_params['t1'])
        self.t1_spin.setSuffix(" s")
        self.t1_spin.setToolTip("Longitudinal relaxation time (T1)")
        decoherence_form.addRow("T1 Time:", self.t1_spin)
        
        # T2 relaxation time
        self.t2_spin = QDoubleSpinBox()
        self.t2_spin.setRange(1e-6, 10)
        self.t2_spin.setDecimals(6)
        self.t2_spin.setSingleStep(10e-6)
        self.t2_spin.setValue(self.simulator_params['t2'])
        self.t2_spin.setSuffix(" s")
        self.t2_spin.setToolTip("Transverse relaxation time (T2)")
        decoherence_form.addRow("T2 Time:", self.t2_spin)
        
        # Measurement noise level
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0, 0.5)
        self.noise_spin.setDecimals(3)
        self.noise_spin.setSingleStep(0.01)
        self.noise_spin.setValue(self.simulator_params['noise_level'])
        self.noise_spin.setSuffix("")
        self.noise_spin.setToolTip("Noise level for simulated measurements")
        decoherence_form.addRow("Measurement Noise:", self.noise_spin)
        
        decoherence_group.setLayout(decoherence_form)
        layout.addWidget(decoherence_group)
        
        # Initial State Controls
        initial_state_group = QGroupBox("Initial NV State")
        initial_state_layout = QVBoxLayout()
        
        # Radio buttons for pre-defined states
        state_radio_layout = QHBoxLayout()
        self.state_group = QButtonGroup()
        
        self.state_0_radio = QRadioButton("|0⟩")
        self.state_0_radio.setChecked(True)
        self.state_group.addButton(self.state_0_radio, 0)
        
        self.state_plus_radio = QRadioButton("|+1⟩")
        self.state_group.addButton(self.state_plus_radio, 1)
        
        self.state_minus_radio = QRadioButton("|-1⟩")
        self.state_group.addButton(self.state_minus_radio, 2)
        
        self.state_superpos_radio = QRadioButton("Superposition")
        self.state_group.addButton(self.state_superpos_radio, 3)
        
        state_radio_layout.addWidget(self.state_0_radio)
        state_radio_layout.addWidget(self.state_plus_radio)
        state_radio_layout.addWidget(self.state_minus_radio)
        state_radio_layout.addWidget(self.state_superpos_radio)
        
        initial_state_layout.addLayout(state_radio_layout)
        
        # Custom state controls (enabled when "Superposition" is selected)
        custom_state_form = QFormLayout()
        
        # Probabilities for each basis state
        self.prob_0_spin = QDoubleSpinBox()
        self.prob_0_spin.setRange(0, 1)
        self.prob_0_spin.setDecimals(2)
        self.prob_0_spin.setSingleStep(0.1)
        self.prob_0_spin.setValue(1.0)
        self.prob_0_spin.setEnabled(False)
        custom_state_form.addRow("Prob. |0⟩:", self.prob_0_spin)
        
        self.prob_plus_spin = QDoubleSpinBox()
        self.prob_plus_spin.setRange(0, 1)
        self.prob_plus_spin.setDecimals(2)
        self.prob_plus_spin.setSingleStep(0.1)
        self.prob_plus_spin.setValue(0.0)
        self.prob_plus_spin.setEnabled(False)
        custom_state_form.addRow("Prob. |+1⟩:", self.prob_plus_spin)
        
        self.prob_minus_spin = QDoubleSpinBox()
        self.prob_minus_spin.setRange(0, 1)
        self.prob_minus_spin.setDecimals(2)
        self.prob_minus_spin.setSingleStep(0.1)
        self.prob_minus_spin.setValue(0.0)
        self.prob_minus_spin.setEnabled(False)
        custom_state_form.addRow("Prob. |-1⟩:", self.prob_minus_spin)
        
        initial_state_layout.addLayout(custom_state_form)
        initial_state_group.setLayout(initial_state_layout)
        layout.addWidget(initial_state_group)
        
        # Set the tab layout
        layout.addStretch(1)
        tab.setLayout(layout)
        return tab
    
    def _create_experiment_tab(self):
        """Create the experiment configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Experiment type selection
        exp_group = QGroupBox("Experiment Type")
        exp_layout = QVBoxLayout()
        
        self.exp_combo = QComboBox()
        self.exp_combo.addItems(["Rabi Oscillation", "Ramsey Interferometry", "Spin Echo"])
        exp_layout.addWidget(self.exp_combo)
        
        # Experiment description
        self.exp_description = QLabel("Rabi oscillation: Measures coherent state rotations with a resonant MW pulse of varying duration.")
        self.exp_description.setWordWrap(True)
        exp_layout.addWidget(self.exp_description)
        
        exp_group.setLayout(exp_layout)
        layout.addWidget(exp_group)
        
        # Experiment parameters
        params_group = QGroupBox("Experiment Parameters")
        params_form = QFormLayout()
        
        # Maximum experiment time
        self.max_time_spin = QDoubleSpinBox()
        self.max_time_spin.setRange(1e-9, 1)
        self.max_time_spin.setDecimals(9)
        self.max_time_spin.setSingleStep(1e-7)
        self.max_time_spin.setValue(self.experiment_params['max_time'])
        self.max_time_spin.setSuffix(" s")
        params_form.addRow("Maximum Time:", self.max_time_spin)
        
        # Number of steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 1000)
        self.steps_spin.setValue(self.experiment_params['steps'])
        params_form.addRow("Number of Steps:", self.steps_spin)
        
        # Pulse amplitude
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(0, 10)
        self.amplitude_spin.setDecimals(2)
        self.amplitude_spin.setSingleStep(0.1)
        self.amplitude_spin.setValue(self.experiment_params['amplitude'])
        self.amplitude_spin.setSuffix("")
        params_form.addRow("Pulse Amplitude:", self.amplitude_spin)
        
        params_group.setLayout(params_form)
        layout.addWidget(params_group)
        
        # Experiment visualization
        visualization_group = QGroupBox("Experiment Visualization")
        visualization_layout = QVBoxLayout()
        
        # Experiment schematic
        self.exp_schematic = QLabel("Experiment sequence will be shown here")
        self.exp_schematic.setAlignment(Qt.AlignCenter)
        self.exp_schematic.setFixedHeight(100)
        visualization_layout.addWidget(self.exp_schematic)
        
        visualization_group.setLayout(visualization_layout)
        layout.addWidget(visualization_group)
        
        # Set the tab layout
        layout.addStretch(1)
        tab.setLayout(layout)
        return tab
    
    def _create_pulse_tab(self):
        """Create the pulse sequence tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Pulse sequence builder
        pulse_group = QGroupBox("Pulse Sequence Builder")
        pulse_layout = QVBoxLayout()
        
        # Pulse parameters
        params_form = QFormLayout()
        
        # Pulse duration
        self.pulse_duration_spin = QDoubleSpinBox()
        self.pulse_duration_spin.setRange(1e-9, 1e-3)
        self.pulse_duration_spin.setDecimals(9)
        self.pulse_duration_spin.setSingleStep(1e-8)
        self.pulse_duration_spin.setValue(100e-9)
        self.pulse_duration_spin.setSuffix(" s")
        params_form.addRow("Pulse Duration:", self.pulse_duration_spin)
        
        # Pulse amplitude
        self.pulse_amplitude_spin = QDoubleSpinBox()
        self.pulse_amplitude_spin.setRange(-1, 1)
        self.pulse_amplitude_spin.setDecimals(2)
        self.pulse_amplitude_spin.setSingleStep(0.1)
        self.pulse_amplitude_spin.setValue(1.0)
        params_form.addRow("Pulse Amplitude:", self.pulse_amplitude_spin)
        
        # Pulse phase
        self.pulse_phase_spin = QDoubleSpinBox()
        self.pulse_phase_spin.setRange(-180, 180)
        self.pulse_phase_spin.setDecimals(1)
        self.pulse_phase_spin.setSingleStep(15)
        self.pulse_phase_spin.setValue(0)
        self.pulse_phase_spin.setSuffix("°")
        params_form.addRow("Pulse Phase:", self.pulse_phase_spin)
        
        pulse_layout.addLayout(params_form)
        
        # Pulse control buttons
        pulse_buttons_layout = QHBoxLayout()
        self.add_pulse_button = QPushButton("Add Pulse")
        self.clear_sequence_button = QPushButton("Clear Sequence")
        self.apply_sequence_button = QPushButton("Apply Sequence")
        
        pulse_buttons_layout.addWidget(self.add_pulse_button)
        pulse_buttons_layout.addWidget(self.clear_sequence_button)
        pulse_buttons_layout.addWidget(self.apply_sequence_button)
        pulse_layout.addLayout(pulse_buttons_layout)
        
        # Pulse sequence display
        self.pulse_sequence_text = QTextEdit()
        self.pulse_sequence_text.setReadOnly(True)
        self.pulse_sequence_text.setMaximumHeight(100)
        pulse_layout.addWidget(QLabel("Current Pulse Sequence:"))
        pulse_layout.addWidget(self.pulse_sequence_text)
        
        # Pulse sequence visualization
        self.pulse_sequence_plot = pg.PlotWidget()
        self.pulse_sequence_plot.setLabel('bottom', 'Time', 's')
        self.pulse_sequence_plot.setLabel('left', 'Amplitude', '')
        pulse_layout.addWidget(self.pulse_sequence_plot)
        
        pulse_group.setLayout(pulse_layout)
        layout.addWidget(pulse_group)
        
        # Common pulse sequences
        templates_group = QGroupBox("Predefined Sequences")
        templates_layout = QHBoxLayout()
        
        self.pi_pulse_button = QPushButton("π Pulse")
        self.pi_half_pulse_button = QPushButton("π/2 Pulse")
        self.ramsey_button = QPushButton("Ramsey Sequence")
        self.spin_echo_button = QPushButton("Spin Echo")
        
        templates_layout.addWidget(self.pi_pulse_button)
        templates_layout.addWidget(self.pi_half_pulse_button)
        templates_layout.addWidget(self.ramsey_button)
        templates_layout.addWidget(self.spin_echo_button)
        
        templates_group.setLayout(templates_layout)
        layout.addWidget(templates_group)
        
        # Set the tab layout
        tab.setLayout(layout)
        return tab
    
    def _connect_internal_signals(self):
        """Connect internal signals to slots."""
        # NV parameters tab connections
        self.d_gs_spin.valueChanged.connect(self._update_nv_parameters)
        self.e_gs_spin.valueChanged.connect(self._update_nv_parameters)
        self.gamma_e_spin.valueChanged.connect(self._update_nv_parameters)
        self.bx_spin.valueChanged.connect(self._update_nv_parameters)
        self.by_spin.valueChanged.connect(self._update_nv_parameters)
        self.bz_spin.valueChanged.connect(self._update_nv_parameters)
        self.t1_spin.valueChanged.connect(self._update_nv_parameters)
        self.t2_spin.valueChanged.connect(self._update_nv_parameters)
        self.noise_spin.valueChanged.connect(self._update_nv_parameters)
        
        # Initial state radio buttons
        self.state_group.buttonClicked.connect(self._handle_state_selection)
        
        # Experiment tab connections
        self.exp_combo.currentIndexChanged.connect(self._update_experiment_description)
        self.max_time_spin.valueChanged.connect(self._update_experiment_parameters)
        self.steps_spin.valueChanged.connect(self._update_experiment_parameters)
        self.amplitude_spin.valueChanged.connect(self._update_experiment_parameters)
        
        # Pulse tab connections
        self.add_pulse_button.clicked.connect(self._add_pulse_to_sequence)
        self.clear_sequence_button.clicked.connect(self._clear_pulse_sequence)
        self.apply_sequence_button.clicked.connect(self._apply_pulse_sequence)
        self.pi_pulse_button.clicked.connect(lambda: self._load_predefined_sequence("pi"))
        self.pi_half_pulse_button.clicked.connect(lambda: self._load_predefined_sequence("pi_half"))
        self.ramsey_button.clicked.connect(lambda: self._load_predefined_sequence("ramsey"))
        self.spin_echo_button.clicked.connect(lambda: self._load_predefined_sequence("spin_echo"))
        
        # Control buttons
        self.reset_button.clicked.connect(self._reset_nv_state)
        self.start_button.clicked.connect(self._start_experiment)
        self.stop_button.clicked.connect(self._stop_experiment)
    
    def _update_nv_parameters(self):
        """Update the NV center simulator parameters from the UI."""
        try:
            self.simulator_params.update({
                'd_gs': self.d_gs_spin.value(),
                'e_gs': self.e_gs_spin.value(),
                'gamma_e': self.gamma_e_spin.value(),
                'b_field': [
                    self.bx_spin.value(),
                    self.by_spin.value(),
                    self.bz_spin.value()
                ],
                't1': self.t1_spin.value(),
                't2': self.t2_spin.value(),
                'noise_level': self.noise_spin.value()
            })
            
            # If connected to the logic module, update its parameters
            if self.nvsimulatorlogic is not None:
                try:
                    # Update the Hamiltonian with new parameters
                    self.nvsimulatorlogic.update_hamiltonian(
                        b_field=self.simulator_params['b_field'],
                        e_gs=self.simulator_params['e_gs'],
                        d_gs=self.simulator_params['d_gs']
                    )
                    
                    # Update other parameters
                    self.nvsimulatorlogic.t1 = self.simulator_params['t1']
                    self.nvsimulatorlogic.t2 = self.simulator_params['t2']
                    self.nvsimulatorlogic.noise_level = self.simulator_params['noise_level']
                    
                    self.log_message("Updated NV center parameters")
                except Exception as e:
                    self.log_message(f"Error updating parameters: {str(e)}")
        except Exception as e:
            self.log_message(f"Error in parameter update: {str(e)}")
    
    def _handle_state_selection(self, button):
        """Handle the selection of an initial state."""
        # Enable/disable custom state controls
        is_superposition = button is self.state_superpos_radio
        self.prob_0_spin.setEnabled(is_superposition)
        self.prob_plus_spin.setEnabled(is_superposition)
        self.prob_minus_spin.setEnabled(is_superposition)
        
        # Set default state values based on the selected state
        if button is self.state_0_radio:
            self.prob_0_spin.setValue(1.0)
            self.prob_plus_spin.setValue(0.0)
            self.prob_minus_spin.setValue(0.0)
        elif button is self.state_plus_radio:
            self.prob_0_spin.setValue(0.0)
            self.prob_plus_spin.setValue(1.0)
            self.prob_minus_spin.setValue(0.0)
        elif button is self.state_minus_radio:
            self.prob_0_spin.setValue(0.0)
            self.prob_plus_spin.setValue(0.0)
            self.prob_minus_spin.setValue(1.0)
        
        # Reset the NV state if the logic module is connected
        self._reset_nv_state()
    
    def _update_experiment_description(self):
        """Update the experiment description based on the selected experiment."""
        experiment = self.exp_combo.currentText()
        
        if experiment == "Rabi Oscillation":
            description = "Rabi oscillation: Measures coherent state rotations with a resonant MW pulse of varying duration."
            self.experiment_params['experiment_type'] = 'rabi'
        elif experiment == "Ramsey Interferometry":
            description = "Ramsey interferometry: Measures phase evolution during free precession between two π/2 pulses."
            self.experiment_params['experiment_type'] = 'ramsey'
        elif experiment == "Spin Echo":
            description = "Spin Echo: Measures coherence preserved by a π pulse between two π/2 pulses, canceling static field inhomogeneities."
            self.experiment_params['experiment_type'] = 'spin_echo'
        else:
            description = ""
            
        self.exp_description.setText(description)
        self._update_experiment_schematic()
    
    def _update_experiment_schematic(self):
        """Update the experiment schematic visualization."""
        # This would ideally render a schematic of the pulse sequence
        # For now, we'll just update the text description
        experiment = self.exp_combo.currentText()
        
        if experiment == "Rabi Oscillation":
            self.exp_schematic.setText("|----- variable duration MW pulse -----|→ measure")
        elif experiment == "Ramsey Interferometry":
            self.exp_schematic.setText("|π/2|-- variable free evolution --|π/2|→ measure")
        elif experiment == "Spin Echo":
            self.exp_schematic.setText("|π/2|--- τ ---|π|--- τ ---|π/2|→ measure")
    
    def _update_experiment_parameters(self):
        """Update the experiment parameters from the UI."""
        self.experiment_params.update({
            'max_time': self.max_time_spin.value(),
            'steps': self.steps_spin.value(),
            'amplitude': self.amplitude_spin.value()
        })
    
    def _reset_nv_state(self):
        """Reset the NV center state."""
        try:
            if self.nvsimulatorlogic is not None:
                # Determine the initial state from the UI
                if self.state_superpos_radio.isChecked():
                    # Custom superposition state
                    state = [
                        self.prob_0_spin.value(),
                        self.prob_plus_spin.value(),
                        self.prob_minus_spin.value()
                    ]
                elif self.state_plus_radio.isChecked():
                    state = [0, 1, 0]  # |+1⟩ state
                elif self.state_minus_radio.isChecked():
                    state = [0, 0, 1]  # |-1⟩ state
                else:
                    state = [1, 0, 0]  # |0⟩ state (default)
                
                # Reset the NV state
                self.nvsimulatorlogic.reset_state(state)
                self.log_message(f"Reset NV state to {state}")
            else:
                self.log_message("Warning: NV simulator logic not connected")
        except Exception as e:
            self.log_message(f"Error resetting state: {str(e)}")
    
    def _start_experiment(self):
        """Start the selected experiment."""
        try:
            if self.nvsimulatorlogic is None:
                self.log_message("Error: NV simulator logic not connected")
                return
            
            # Update UI state
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_value.setText("Running experiment...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Determine which experiment to run
            experiment_type = self.experiment_params['experiment_type']
            
            # Log experiment start
            self.log_message(f"Starting {experiment_type} experiment with {self.steps_spin.value()} steps...")
            
            # Clear previous results
            self.results_plot.clear()
            
            # Run the selected experiment
            if experiment_type == 'rabi':
                # Start the Rabi experiment
                self.nvsimulatorlogic.simulate_rabi_oscillation(
                    max_time=self.experiment_params['max_time'],
                    steps=self.experiment_params['steps'],
                    amplitude=self.experiment_params['amplitude']
                )
            elif experiment_type == 'ramsey':
                # Start the Ramsey experiment
                self.nvsimulatorlogic.simulate_ramsey(
                    free_evolution_time=self.experiment_params['max_time'],
                    steps=self.experiment_params['steps']
                )
            elif experiment_type == 'spin_echo':
                # Start the Spin Echo experiment
                self.nvsimulatorlogic.simulate_spin_echo(
                    max_time=self.experiment_params['max_time'],
                    steps=self.experiment_params['steps']
                )
        except Exception as e:
            self.log_message(f"Error starting experiment: {str(e)}")
            self._experiment_finished(success=False)
    
    def _stop_experiment(self):
        """Stop the current experiment."""
        try:
            if self.nvsimulatorlogic is not None:
                with self.nvsimulatorlogic._mutex:
                    self.nvsimulatorlogic._stop_requested = True
                self.log_message("Stopping experiment...")
            self._experiment_finished(success=False)
        except Exception as e:
            self.log_message(f"Error stopping experiment: {str(e)}")
    
    def _experiment_finished(self, success=True):
        """Update the UI when an experiment is finished."""
        # Update UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_value.setText("Experiment completed")
        else:
            self.status_value.setText("Experiment stopped")
    
    def _add_pulse_to_sequence(self):
        """Add a pulse to the sequence."""
        # Get pulse parameters from the UI
        amplitude = self.pulse_amplitude_spin.value()
        phase_deg = self.pulse_phase_spin.value()
        phase_rad = phase_deg * (np.pi / 180)  # Convert to radians
        duration = self.pulse_duration_spin.value()
        
        # Add to the internal pulse sequence
        if not hasattr(self, 'pulse_sequence'):
            self.pulse_sequence = []
        
        self.pulse_sequence.append((amplitude, phase_rad, duration))
        
        # Update the pulse sequence display
        self._update_pulse_sequence_display()
        
        self.log_message(f"Added pulse: amp={amplitude}, phase={phase_deg}°, duration={duration:.3e} s")
    
    def _clear_pulse_sequence(self):
        """Clear the pulse sequence."""
        if hasattr(self, 'pulse_sequence'):
            self.pulse_sequence = []
        
        # Clear the pulse sequence display
        self._update_pulse_sequence_display()
        
        self.log_message("Cleared pulse sequence")
    
    def _apply_pulse_sequence(self):
        """Apply the current pulse sequence to the NV state."""
        try:
            if self.nvsimulatorlogic is None:
                self.log_message("Error: NV simulator logic not connected")
                return
            
            if not hasattr(self, 'pulse_sequence') or not self.pulse_sequence:
                self.log_message("Error: No pulse sequence defined")
                return
            
            # Update UI state
            self.status_value.setText("Applying pulse sequence...")
            
            # Apply the pulse sequence
            self.nvsimulatorlogic.apply_pulse_sequence(self.pulse_sequence)
            
            self.log_message(f"Applied pulse sequence with {len(self.pulse_sequence)} pulses")
            self.status_value.setText("Ready")
        except Exception as e:
            self.log_message(f"Error applying pulse sequence: {str(e)}")
            self.status_value.setText("Error")
    
    def _update_pulse_sequence_display(self):
        """Update the display of the current pulse sequence."""
        # Update text display
        if not hasattr(self, 'pulse_sequence') or not self.pulse_sequence:
            self.pulse_sequence_text.setText("No pulses defined")
            self.pulse_sequence_plot.clear()
            return
        
        # Format the pulse sequence as text
        text = "Current pulse sequence:\n"
        for i, (amp, phase, duration) in enumerate(self.pulse_sequence):
            phase_deg = phase * (180 / np.pi)  # Convert to degrees
            text += f"Pulse {i+1}: amp={amp:.2f}, phase={phase_deg:.1f}°, duration={duration:.3e} s\n"
        
        self.pulse_sequence_text.setText(text)
        
        # Update the plot
        self.pulse_sequence_plot.clear()
        
        # Create a time grid for the entire sequence
        total_duration = sum(duration for _, _, duration in self.pulse_sequence)
        if total_duration <= 0:
            return
            
        time_points = []
        amplitudes = []
        
        current_time = 0
        for amp, phase, duration in self.pulse_sequence:
            # Create time points for this pulse
            # For better visualization, add a small gap between pulses
            t_pulse = np.linspace(current_time, current_time + duration, 100)
            
            # Calculate the x and y components of the pulse
            x_comp = amp * np.cos(phase)
            y_comp = amp * np.sin(phase)
            
            # For visualization, we'll just show the amplitude
            # In a more sophisticated visualization, we'd show both amplitude and phase
            a_pulse = amp * np.ones_like(t_pulse)
            
            time_points.extend(t_pulse)
            amplitudes.extend(a_pulse)
            
            current_time += duration
            
            # Add a small gap for visualization
            time_points.append(current_time)
            amplitudes.append(0)
            
            # Add a gap between pulses
            current_time += 1e-11
        
        # Plot the sequence
        self.pulse_sequence_plot.plot(time_points, amplitudes, pen='r')
        
        # Set axis labels
        self.pulse_sequence_plot.setLabel('bottom', 'Time', 's')
        self.pulse_sequence_plot.setLabel('left', 'Amplitude', '')
    
    def _load_predefined_sequence(self, sequence_type):
        """Load a predefined pulse sequence."""
        if not hasattr(self, 'pulse_sequence'):
            self.pulse_sequence = []
        else:
            self.pulse_sequence = []
        
        if self.nvsimulatorlogic is None:
            self.log_message("Warning: NV simulator logic not connected, using default π pulse time")
            pi_time = 50e-9  # Default π pulse time
        else:
            # Estimate π pulse time from the logic
            pi_time = self.nvsimulatorlogic._estimate_pi_pulse_time()
        
        if sequence_type == "pi":
            # Add a π pulse
            self.pulse_sequence.append((1.0, 0.0, pi_time))
            self.log_message(f"Loaded π pulse sequence (duration: {pi_time:.3e} s)")
        
        elif sequence_type == "pi_half":
            # Add a π/2 pulse
            self.pulse_sequence.append((1.0, 0.0, pi_time / 2))
            self.log_message(f"Loaded π/2 pulse sequence (duration: {pi_time/2:.3e} s)")
        
        elif sequence_type == "ramsey":
            # Ramsey sequence: π/2 - τ - π/2
            self.pulse_sequence.append((1.0, 0.0, pi_time / 2))  # First π/2 pulse
            self.pulse_sequence.append((0.0, 0.0, 1e-6))  # Free evolution (τ)
            self.pulse_sequence.append((1.0, 0.0, pi_time / 2))  # Second π/2 pulse
            self.log_message("Loaded Ramsey sequence")
        
        elif sequence_type == "spin_echo":
            # Spin Echo sequence: π/2 - τ - π - τ - π/2
            self.pulse_sequence.append((1.0, 0.0, pi_time / 2))  # First π/2 pulse
            self.pulse_sequence.append((0.0, 0.0, 0.5e-6))  # Free evolution (τ)
            self.pulse_sequence.append((1.0, 0.0, pi_time))  # π pulse
            self.pulse_sequence.append((0.0, 0.0, 0.5e-6))  # Free evolution (τ)
            self.pulse_sequence.append((1.0, 0.0, pi_time / 2))  # Second π/2 pulse
            self.log_message("Loaded Spin Echo sequence")
        
        # Update the pulse sequence display
        self._update_pulse_sequence_display()
    
    def log_message(self, message):
        """Add a message to the log text area."""
        self.log_text.append(message)
    
    def _handle_measurement_complete(self, result_data):
        """Handle the completion of a measurement or experiment.
        
        Parameters
        ----------
        result_data : dict
            Dictionary containing experiment results
        """
        experiment_type = result_data.get('type')
        data = result_data.get('data')
        
        if not data:
            self.log_message("Error: No data received from experiment")
            self._experiment_finished(success=False)
            return
        
        # Update the plot with the results
        try:
            self.results_plot.clear()
            
            # Extract the data
            time_points = data.get('time_points')
            pop_0 = data.get('population_0')
            pop_plus = data.get('population_plus')
            pop_minus = data.get('population_minus')
            
            # Plot the results
            if time_points is not None and pop_0 is not None:
                self.results_plot.plot(time_points, pop_0, pen='b', name="|0⟩")
            if time_points is not None and pop_plus is not None:
                self.results_plot.plot(time_points, pop_plus, pen='r', name="|+1⟩")
            if time_points is not None and pop_minus is not None:
                self.results_plot.plot(time_points, pop_minus, pen='g', name="|-1⟩")
            
            # Add a legend
            legend = self.results_plot.addLegend()
            
            # Set axis labels based on experiment type
            if experiment_type == 'rabi':
                self.results_plot.setTitle("Rabi Oscillation Results")
                self.results_plot.setLabel('bottom', 'Pulse Duration', 's')
            elif experiment_type == 'ramsey':
                self.results_plot.setTitle("Ramsey Interferometry Results")
                self.results_plot.setLabel('bottom', 'Free Evolution Time', 's')
            elif experiment_type == 'spin_echo':
                self.results_plot.setTitle("Spin Echo Results")
                self.results_plot.setLabel('bottom', 'Total Evolution Time', 's')
            
            self.results_plot.setLabel('left', 'Population', '')
            
            self.log_message(f"Experiment {experiment_type} completed successfully")
            self._experiment_finished(success=True)
            
        except Exception as e:
            self.log_message(f"Error plotting results: {str(e)}")
            self._experiment_finished(success=False)
    
    def _handle_simulation_progress(self, progress):
        """Handle simulation progress updates.
        
        Parameters
        ----------
        progress : float
            Progress value between 0 and 1
        """
        # Update the progress bar
        self.progress_bar.setValue(int(progress * 100))
    
    def _handle_state_updated(self, state_data):
        """Handle state updates from the simulator.
        
        Parameters
        ----------
        state_data : dict
            Dictionary containing state information
        """
        # Update the Bloch sphere visualization
        if 'density_matrix' in state_data:
            self.bloch_widget.update_state(state_data['density_matrix'])
    
    def on_activate(self):
        """Module activation in Qudi."""
        # Get connected logic module
        try:
            self.nvsimulatorlogic = self.nv_simulator_logic()
            
            # Connect signals from the logic module
            self.nvsimulatorlogic.sigStateUpdated.connect(self._handle_state_updated)
            self.nvsimulatorlogic.sigMeasurementComplete.connect(self._handle_measurement_complete)
            self.nvsimulatorlogic.sigSimulationProgress.connect(self._handle_simulation_progress)
            
            # Initialize the NV parameters
            self._update_nv_parameters()
            
            # Reset the NV state to initialize visualization
            self._reset_nv_state()
            
            self.log_message("Successfully connected to NV simulator logic")
            
        except Exception as e:
            self.log_message(f"Warning: Could not connect to logic: {str(e)}")
        
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