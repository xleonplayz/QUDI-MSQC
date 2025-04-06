"""
This is the logic class for the optimization with QUOCS integration.
"""
import time
import numpy as np
import json
import os
from PySide2.QtCore import Signal
from typing import Dict, List, Any, Optional

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.util.mutex import Mutex

try:
    import quocs
    from quocs.pulseoptimization import PulseOptimization
    from quocs.pulses import PulseSequence
    from quocs.controls import Control
    from quocs.gradients import Gradient
    from quocs.optimalalgorithms import GRAPE, CRAB, GOAT, dCRAB, GROUP
    QUOCS_AVAILABLE = True
except ImportError:
    QUOCS_AVAILABLE = False

class OptimizationLogic(LogicBase):
    """Logic module for optimization control with QUOCS integration"""
    # Define a proper _threaded attribute for the Logic module
    _threaded = True

    # Configurables
    quocs_default_config = ConfigOption('quocs_default_config', None)

    # Connectors
    fom_logic = Connector(interface="WorkerFom")
    controls_logic = Connector(interface="WorkerControls")
    
    # Optional connectors for dummy hardware from qudi-iqo-modules
    pulser_dummy = Connector(interface='PulserInterface', optional=True)
    optimizer_dummy = Connector(interface='OptimizerInterface', optional=True)
    
    # Define all signals
    load_optimization_dictionary_signal = Signal(dict)
    send_controls_signal = Signal(list, list, list)
    wait_fom_signal = Signal(str)
    is_running_signal = Signal(bool)
    message_label_signal = Signal(str)
    fom_plot_signal = Signal(object)
    controls_update_signal = Signal(object)
    optimization_status_signal = Signal(dict)
    
    # Additional signals for compatibility with qudi-iqo-modules dummy hardware
    pulser_on_signal = Signal(bool)  # For controlling dummy pulser hardware
    get_samples_signal = Signal(list, list)  # For getting samples from dummy hardware
    sigPulseOptimizationComplete = Signal(object)  # For pulse optimization completion notification

    def __init__(self, config=None, **kwargs):
        """Initialize the base class"""
        super().__init__(config=config, **kwargs)

        self.opti_comm_dict = {}
        self.is_fom_computed = False
        self.fom_max = 10 ** 10
        self.fom = 10**10
        self.std = 0.0
        self.status_code = 0
        self._threadlock = Mutex()
        self._running = False
        self.optimization_obj = None
        
        # QUOCS-specific attributes
        self.quocs_available = QUOCS_AVAILABLE
        self.pulse_optimization = None
        self.fom_history = []
        self.current_iteration = 0
        self.initial_fom = None
        self.best_fom = None
        self.best_pulses = None
        self._use_dummy_hardware = False  # Will be set in on_activate based on available connectors
        self._simulate_optimization = True  # Default to simulation mode
        self.best_iteration = 0
        
        # Dictionary to convert between QUOCS algorithm classes and names
        self.algorithm_map = {
            'GRAPE': GRAPE if QUOCS_AVAILABLE else None,
            'CRAB': CRAB if QUOCS_AVAILABLE else None,
            'dCRAB': dCRAB if QUOCS_AVAILABLE else None,
            'GOAT': GOAT if QUOCS_AVAILABLE else None,
            'GROUP': GROUP if QUOCS_AVAILABLE else None
        }

    def on_activate(self):
        """ Activation """
        self.log.info("Starting the Optimization Logic with QUOCS integration")
        
        if not self.quocs_available:
            self.log.warning("QUOCS is not available! Install it with 'pip install quocs' to enable full functionality")
            self.message_label_signal.emit("WARNING: QUOCS is not available. Install with 'pip install quocs'")
        
        # Connect signals between components
        self.send_controls_signal.connect(self.controls_logic().update_controls)
        self.fom_logic().send_fom_signal.connect(self.update_FoM)
        self.wait_fom_signal.connect(self.fom_logic().wait_for_fom)
        
        # Connect to dummy hardware modules if available
        self._use_dummy_hardware = False
        try:
            if self.pulser_dummy.is_connected and self.optimizer_dummy.is_connected:
                self.pulser = self.pulser_dummy()
                self.optimizer = self.optimizer_dummy()
                
                # Connect signals to dummy hardware
                if hasattr(self.pulser, 'sigPulserRunningChanged'):
                    self.pulser_on_signal.connect(self.pulser.sigPulserRunningChanged)
                    
                if hasattr(self.optimizer, 'sigPulseOptimizationComplete'):
                    self.sigPulseOptimizationComplete.connect(self.optimizer.sigPulseOptimizationComplete)
                
                self.log.info("Connected to dummy pulser and optimizer from qudi-iqo-modules")
                self._use_dummy_hardware = True
            elif self.pulser_dummy.is_connected or self.optimizer_dummy.is_connected:
                self.log.warning("Only one of pulser_dummy or optimizer_dummy is connected. Need both for dummy mode.")
        except Exception as e:
            self.log.warning(f"Error connecting to dummy hardware: {str(e)}")
        
        # Load default configuration if provided
        if self.quocs_default_config:
            try:
                if os.path.isfile(self.quocs_default_config):
                    with open(self.quocs_default_config, 'r') as f:
                        if self.quocs_default_config.endswith('.json'):
                            default_config = json.load(f)
                        else:
                            import yaml
                            default_config = yaml.safe_load(f)
                    self.load_opti_comm_dict(default_config)
                    self.log.info(f"Loaded default QUOCS configuration from {self.quocs_default_config}")
            except Exception as e:
                self.log.error(f"Error loading default QUOCS configuration: {str(e)}")
        
        # Notify that we're ready
        if self._use_dummy_hardware:
            self.message_label_signal.emit("Optimization logic activated with dummy hardware")
        else:
            self.message_label_signal.emit("Optimization logic activated and ready")

    def update_FoM(self, fom_dict):
        """ Update the figure of merit from the fom logic """
        self.status_code = fom_dict.setdefault("status_code", 0)
        self.std = fom_dict.setdefault("std", 0.0)
        self.fom = fom_dict["FoM"]
        self.is_fom_computed = True
        
        # Add to history if optimization is running
        if self._running and self.status_code == 0:
            if self.initial_fom is None:
                self.initial_fom = self.fom
                
            self.fom_history.append(self.fom)
            
            # Track best FoM
            if self.best_fom is None or self.fom < self.best_fom:
                self.best_fom = self.fom
                # We would also store the best pulses here in a real implementation
                self.best_iteration = self.current_iteration
        
        # Emit signal to update plot
        self.fom_plot_signal.emit({
            "fom": self.fom,
            "std": self.std,
            "status": self.status_code,
            "fom_history": self.fom_history,
            "initial_fom": self.initial_fom,
            "best_fom": self.best_fom,
            "best_iteration": self.best_iteration
        })

    def is_computed(self):
        """ Check if the figure of is updated """
        return self.is_fom_computed

    def get_FoM(self, pulses, parameters, timegrids):
        """ Send the controls to the worker controls and wait the figure of merit from the worker fom """
        # Only process if we're running
        if not self._running:
            return {"FoM": self.fom_max, "status_code": -1}
            
        # Update UI with current pulse shapes
        self.controls_update_signal.emit({
            "pulses": pulses,
            "timegrids": timegrids,
            "parameters": parameters,
            "iteration": self.current_iteration
        })
        
        # Special case for simulation mode
        if not self.quocs_available and not self._use_dummy_hardware:
            # In simulation mode, create a synthetic FoM that improves over time
            try:
                # Generate an improving FoM over iterations
                if self.current_iteration > 0:
                    base_fom = 0.8 - 0.7 * (self.current_iteration / 100.0)
                    # Add some randomness to make it more realistic
                    noise = np.random.normal(0, 0.02)
                    simulated_fom = max(0.01, min(0.9, base_fom + noise))
                    
                    # Update our FoM value
                    self.fom = simulated_fom
                    self.std = 0.01
                    self.status_code = 0
                    
                    # Update history
                    if self.initial_fom is None:
                        self.initial_fom = self.fom
                    else:
                        self.fom_history.append(self.fom)
                        
                    # Check if this is the best FoM
                    if self.best_fom is None or self.fom < self.best_fom:
                        self.best_fom = self.fom
                        self.best_pulses = pulses
                        self.best_iteration = self.current_iteration
                    
                    # Update FoM plot in GUI
                    self.fom_plot_signal.emit({
                        'fom_history': self.fom_history,
                        'initial_fom': self.initial_fom
                    })
                    
                    # Simulate some processing time
                    time.sleep(0.2)
                    
                    return {"FoM": self.fom, "std": self.std, "status_code": self.status_code}
                else:
                    # Initial FoM
                    self.fom = 0.8 + np.random.normal(0, 0.02)
                    self.initial_fom = self.fom
                    self.std = 0.05
                    self.status_code = 0
                    return {"FoM": self.fom, "std": self.std, "status_code": self.status_code}
            except Exception as e:
                self.log.error(f"Error in simulation FoM calculation: {str(e)}")
                return {"FoM": 0.5, "std": 0.1, "status_code": -3}
        
        # Normal processing for non-simulation mode - try to use worker modules
        try:
            # Send the controls to the worker
            self.send_controls(pulses, parameters, timegrids)
            
            # Wait for FoM calculation
            self.wait_fom_signal.emit("Start to wait for figure of merit")
            
            # Wait for computation to complete
            timeout = 30.0  # seconds
            start_time = time.time()
            while not self.is_computed():
                time.sleep(0.1)
                if time.time() - start_time > timeout:
                    self.log.warning(f"Timeout waiting for FoM computation after {timeout}s")
                    return {"FoM": self.fom_max, "status_code": -2}
                
                if not self._running:
                    return {"FoM": self.fom_max, "status_code": -1}
                    
            # Reset computation flag and return results
            self.is_fom_computed = False
            
            # Update FoM history
            if self.initial_fom is None:
                self.initial_fom = self.fom
            else:
                self.fom_history.append(self.fom)
                
            # Check if this is the best FoM
            if self.best_fom is None or self.fom < self.best_fom:
                self.best_fom = self.fom
                self.best_pulses = pulses
                self.best_iteration = self.current_iteration
                
            # Update FoM plot in GUI
            self.fom_plot_signal.emit({
                'fom_history': self.fom_history,
                'initial_fom': self.initial_fom
            })
            
            return {"FoM": self.fom, "std": self.std, "status_code": self.status_code}
            
        except Exception as e:
            self.log.error(f"Error in FoM calculation: {str(e)}")
            return {"FoM": self.fom_max, "status_code": -3}

    def load_opti_comm_dict(self, opti_comm_dict):
        """ Load the optimization configuration dictionary and send it to the GUI """
        self.opti_comm_dict = opti_comm_dict
        self.load_optimization_dictionary_signal.emit(opti_comm_dict)
        self.log.info(f"Loaded optimization dictionary with {len(opti_comm_dict)} entries")

    def save_configuration(self, file_path, params=None):
        """Save optimization configuration to a file
        
        Parameters
        ----------
        file_path : str
            Path to save the configuration
        params : dict, optional
            Parameters to save. If None, save current opti_comm_dict.
        """
        if params is None:
            params = self.opti_comm_dict
            
        try:
            with open(file_path, 'w') as f:
                if file_path.endswith('.json'):
                    json.dump(params, f, indent=2)
                else:
                    import yaml
                    yaml.dump(params, f, default_flow_style=False)
                    
            self.message_label_signal.emit(f"Configuration saved to {file_path}")
            self.log.info(f"Saved optimization configuration to {file_path}")
            return True
        except Exception as e:
            self.log.error(f"Error saving configuration to {file_path}: {str(e)}")
            self.message_label_signal.emit(f"Error saving configuration: {str(e)}")
            return False

    def load_configuration(self, file_path):
        """Load optimization configuration from a file
        
        Parameters
        ----------
        file_path : str
            Path to load the configuration from
            
        Returns
        -------
        dict
            Loaded configuration parameters
        """
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    params = json.load(f)
                else:
                    import yaml
                    params = yaml.safe_load(f)
                    
            self.load_opti_comm_dict(params)
            self.message_label_signal.emit(f"Configuration loaded from {file_path}")
            self.log.info(f"Loaded optimization configuration from {file_path}")
            return params
        except Exception as e:
            self.log.error(f"Error loading configuration from {file_path}: {str(e)}")
            self.message_label_signal.emit(f"Error loading configuration: {str(e)}")
            return {}

    def start_optimization(self, opti_comm_dict=None):
        """Start the QUOCS optimization process or use dummy hardware
        
        Parameters
        ----------
        opti_comm_dict : dict, optional
            Optimization parameters dictionary. If None, use current opti_comm_dict.
        """
        if self._running:
            self.log.warning("An optimization is already running")
            return
            
        # Update parameters if provided
        if opti_comm_dict is not None:
            self.load_opti_comm_dict(opti_comm_dict)
            
        # Set running status
        self._running = True
        self.is_running_signal.emit(True)
        
        # Reset optimization state
        self.fom_history = []
        self.current_iteration = 0
        self.initial_fom = None
        self.best_fom = None
        self.best_pulses = None
        self.best_iteration = 0
        
        # Run using dummy hardware if available
        if self._use_dummy_hardware:
            self.message_label_signal.emit("Starting optimization with dummy hardware")
            try:
                # Use dummy hardware for optimization
                self._run_dummy_optimization()
            except Exception as e:
                self.log.error(f"Error during dummy optimization: {str(e)}")
                self.message_label_signal.emit(f"Error: {str(e)}")
                self._running = False
                self.is_running_signal.emit(False)
            return
        
        # Check if QUOCS is available for direct implementation
        if not self.quocs_available:
            # If dummy hardware is not available, use simulation mode instead
            self.message_label_signal.emit("QUOCS is not available. Using simulation mode instead.")
            self.log.warning("QUOCS is not available. Using simulation mode for optimization.")
            
            # Initialize simulation
            try:
                # Initialize simulation
                self._initialize_quocs_optimization()
                
                # Run simulation
                self._run_quocs_optimization()
                return
            except Exception as e:
                self.log.error(f"Error during simulation: {str(e)}")
                self.message_label_signal.emit(f"Error in simulation: {str(e)}")
                self._running = False
                self.is_running_signal.emit(False)
                return
            
        # QUOCS implementation
        self.message_label_signal.emit("Initializing QUOCS optimization process")
        
        try:
            # Initialize QUOCS optimization
            self._initialize_quocs_optimization()
            
            # Start the optimization process
            self._run_quocs_optimization()
            
        except Exception as e:
            self.log.error(f"Error during optimization: {str(e)}")
            self.message_label_signal.emit(f"Error: {str(e)}")
            self._running = False
            self.is_running_signal.emit(False)
            
    def _run_dummy_optimization(self):
        """Run optimization using dummy hardware from qudi-iqo-modules"""
        self.log.info("Starting optimization with dummy hardware")
        
        try:
            # Extract parameters from opti_comm_dict
            algorithm_name = self.opti_comm_dict.get('algorithm', 'GRAPE')
            max_iterations = self.opti_comm_dict.get('iterations', 100)
            pulse_count = self.opti_comm_dict.get('pulse_count', 1)
            sample_count = self.opti_comm_dict.get('sample_count', 100)
            pulse_duration = self.opti_comm_dict.get('pulse_duration', 1.0)
            
            # Check if dummy hardware supports these parameters
            if hasattr(self.optimizer, 'set_optimization_parameters'):
                # Prepare parameters in a format suitable for the dummy
                dummy_params = {
                    'algorithm': algorithm_name,
                    'max_iterations': max_iterations,
                    'control_count': pulse_count,
                    'sample_points': sample_count,
                    'duration': pulse_duration,
                    'constraints': {
                        'min_amplitude': self.opti_comm_dict.get('min_amplitude', -1.0),
                        'max_amplitude': self.opti_comm_dict.get('max_amplitude', 1.0),
                        'zero_boundaries': self.opti_comm_dict.get('zero_boundaries', True),
                    }
                }
                
                # Pass parameters to optimizer
                self.optimizer.set_optimization_parameters(dummy_params)
            
            # Turn on the dummy pulser
            if hasattr(self.pulser, 'pulser_on'):
                self.pulser_on_signal.emit(True)
                self.log.debug("Dummy pulser turned on")
            
            # Start optimization in dummy optimizer
            if hasattr(self.optimizer, 'start_optimization'):
                self.optimizer.start_optimization()
                self.log.info("Dummy optimization started")
                self.message_label_signal.emit(f"Started {algorithm_name} optimization with dummy hardware")
                
                # Set up a timer to simulate optimization progress
                # In a real implementation, the dummy hardware would emit signals
                # that we would connect to in order to track progress
                iteration = 0
                max_iter = max_iterations
                
                # Simulate optimization process with dummy hardware
                while self._running and iteration < max_iter:
                    iteration += 1
                    self.current_iteration = iteration
                    
                    # Simulate a random FoM that improves over time
                    current_fom = 0.5 * ((max_iter - iteration) / max_iter) + 0.01 * np.random.random()
                    self.fom_history.append(current_fom)
                    
                    if self.best_fom is None or current_fom < self.best_fom:
                        self.best_fom = current_fom
                        self.best_iteration = iteration
                    
                    # Update status
                    self.optimization_status_signal.emit({
                        'status': f"Running dummy iteration {iteration}/{max_iter}",
                        'message': f"Iteration {iteration}/{max_iter}: FoM = {current_fom:.6f}",
                        'iterations': list(range(len(self.fom_history) + 1)),
                        'fom_values': [0.5] + self.fom_history
                    })
                    
                    # Generate random pulse shapes for visualization
                    pulses = []
                    timegrids = []
                    for i in range(pulse_count):
                        t = np.linspace(0, pulse_duration, sample_count)
                        # Create a pulse that evolves over iterations
                        freq = 2 + iteration % 5  # Changing frequency
                        progress = iteration / max_iter
                        pulse = np.sin(2 * np.pi * freq * t / pulse_duration) * progress
                        pulses.append(pulse.tolist())
                        timegrids.append(t.tolist())
                    
                    # Update visualization
                    self.controls_update_signal.emit({
                        'pulses': pulses,
                        'timegrids': timegrids,
                        'iteration': iteration
                    })
                    
                    # Update FoM plot
                    self.fom_plot_signal.emit({
                        'fom': current_fom,
                        'fom_history': self.fom_history,
                        'initial_fom': 0.5,
                        'best_fom': self.best_fom,
                        'best_iteration': self.best_iteration
                    })
                    
                    # Delay for visualization
                    time.sleep(0.2)
                
                # Optimization complete or stopped
                if self._running:
                    # Normal completion
                    self.message_label_signal.emit(f"Dummy optimization completed after {iteration} iterations")
                    
                    # Emit optimization completion signal for dummy hardware
                    result = {
                        'pulses': pulses,
                        'fom': self.best_fom,
                        'iterations': iteration
                    }
                    self.sigPulseOptimizationComplete.emit(result)
                    
                else:
                    # User stopped
                    self.message_label_signal.emit(f"Dummy optimization stopped by user after {iteration} iterations")
                
                # Turn off pulser
                if hasattr(self.pulser, 'pulser_off'):
                    self.pulser_on_signal.emit(False)
                    self.log.debug("Dummy pulser turned off")
                
        except Exception as e:
            self.log.error(f"Error during dummy optimization: {str(e)}")
            self.message_label_signal.emit(f"Error in dummy optimization: {str(e)}")
            
        finally:
            self._running = False
            self.is_running_signal.emit(False)

    def _initialize_quocs_optimization(self):
        """Initialize the QUOCS optimization with current parameters"""
        # Extract parameters from opti_comm_dict
        algorithm_name = self.opti_comm_dict.get('algorithm', 'GRAPE')
        max_iterations = self.opti_comm_dict.get('iterations', 100)
        pulse_count = self.opti_comm_dict.get('pulse_count', 1)
        sample_count = self.opti_comm_dict.get('sample_count', 100)
        pulse_duration = self.opti_comm_dict.get('pulse_duration', 1.0)
        max_amplitude = self.opti_comm_dict.get('max_amplitude', 1.0)
        min_amplitude = self.opti_comm_dict.get('min_amplitude', -1.0)
        
        # Log optimization parameters
        self.log.info(f"Initializing {algorithm_name} optimization with {max_iterations} iterations")
        self.log.info(f"Pulses: {pulse_count}, Samples: {sample_count}, Duration: {pulse_duration}")
        
        # Create time grids for pulses
        timegrids = []
        for i in range(pulse_count):
            timegrids.append(np.linspace(0, pulse_duration, sample_count))
            
        # Initialize pulses
        pulses = []
        parameters = []
        for i in range(pulse_count):
            # Initialize with flat pulse at minimum amplitude
            pulse = np.zeros(sample_count)
            pulses.append(pulse.tolist())
            
            # Add parameters for each pulse
            parameters.append({"amplitude": max_amplitude})
        
        # Send initial pulses to FoM calculation
        self.message_label_signal.emit("Calculating initial figure of merit...")
        initial_result = self.get_FoM(pulses, parameters, [t.tolist() for t in timegrids])
        
        # In a real implementation, we would now:
        # 1. Create QUOCS objects for pulses, controls, gradients
        # 2. Create the algorithm object based on selected algorithm
        # 3. Configure the PulseOptimization object with these components
        
        # For now, we'll just prepare for the simulated optimization
        self.pulse_optimization = {
            'algorithm': algorithm_name,
            'max_iterations': max_iterations,
            'pulses': pulses,
            'parameters': parameters,
            'timegrids': [t.tolist() for t in timegrids],
            'initial_fom': initial_result['FoM'],
            'current_iteration': 0
        }
        
        self.message_label_signal.emit(f"Optimization initialized with initial FoM = {initial_result['FoM']:.6f}")

    def _run_quocs_optimization(self):
        """Run the QUOCS optimization process or a simulation if QUOCS is not available"""
        if not self._running or self.pulse_optimization is None:
            return
            
        try:
            # Extract parameters from pulse_optimization
            max_iterations = self.pulse_optimization['max_iterations']
            algorithm = self.pulse_optimization['algorithm']
            
            # Log the start of the optimization
            self.message_label_signal.emit(f"Starting {algorithm} optimization process with {max_iterations} iterations")
            self.log.info(f"Starting {algorithm} optimization with {max_iterations} iterations")
            
            # Decide whether to use real QUOCS or simulation
            if self.quocs_available and not self._simulate_optimization:
                # Use real QUOCS implementation
                self.message_label_signal.emit("Using QUOCS library for optimization")
                # This would be implemented with actual QUOCS code
                # self.pulse_optimization_obj.run()
                # For now, fall back to simulation
                self.message_label_signal.emit("QUOCS integration not fully implemented, using simulation")
                self._simulate_quocs_optimization(max_iterations, algorithm)
            else:
                # Use simulation
                self.message_label_signal.emit("Running simulation of optimization process")
                self._simulate_quocs_optimization(max_iterations, algorithm)
        
        except Exception as e:
            self.log.error(f"Error in QUOCS optimization: {str(e)}")
            self.message_label_signal.emit(f"Error: {str(e)}")
            self._running = False
            self.is_running_signal.emit(False)
    
    def _simulate_quocs_optimization(self, max_iterations, algorithm):
        """Simulate a QUOCS optimization process"""
        # Simulate optimization process with improving pulses
        for i in range(max_iterations):
                if not self._running:
                    self.message_label_signal.emit("Optimization stopped by user")
                    break
                
                self.current_iteration = i + 1
                
                # Simulate pulses evolving over iterations
                pulses = []
                for p in self.pulse_optimization['pulses']:
                    # Create a pulse shape that evolves over iterations
                    t = np.linspace(0, 1, len(p))
                    # Add some randomness but generally improve over time
                    improvement_factor = (i + 1) / max_iterations
                    evolved_pulse = (
                        np.sin(2 * np.pi * 3 * t) * improvement_factor * 0.8 +
                        np.sin(2 * np.pi * 7 * t) * (1-improvement_factor) * 0.3 +
                        np.random.random(len(p)) * (1-improvement_factor) * 0.2
                    )
                    pulses.append(evolved_pulse.tolist())
                
                # Update parameters
                parameters = self.pulse_optimization['parameters']
                timegrids = self.pulse_optimization['timegrids']
                
                # Calculate FoM for current pulses
                result = self.get_FoM(pulses, parameters, timegrids)
                
                # Update optimization status
                self.optimization_status_signal.emit({
                    'status': f"Running iteration {i+1}/{max_iterations}",
                    'message': f"Iteration {i+1}/{max_iterations}: FoM = {result['FoM']:.6f}",
                    'iterations': list(range(len(self.fom_history) + 1)),
                    'fom_values': [self.initial_fom] + self.fom_history,
                    'pulses': pulses,
                    'timegrids': timegrids
                })
                
                # Update pulse optimization state
                self.pulse_optimization['pulses'] = pulses
                self.pulse_optimization['current_iteration'] = i + 1
                
                # Check if we've reached a threshold
                target_fidelity = self.opti_comm_dict.get('target_fidelity', 0.99)
                convergence_threshold = self.opti_comm_dict.get('convergence_threshold', 1e-5)
                
                # Simple convergence check - just for demonstration
                if len(self.fom_history) > 5:
                    # Check last 5 iterations for convergence
                    recent_foms = self.fom_history[-5:]
                    fom_std = np.std(recent_foms)
                    if fom_std < convergence_threshold:
                        self.message_label_signal.emit(f"Optimization converged after {i+1} iterations with FoM = {result['FoM']:.6f}")
                        break
                
                # Check if reached target fidelity (assuming FoM is a loss function that should be minimized)
                if len(self.fom_history) > 0 and self.fom_history[-1] < (1 - target_fidelity):
                    self.message_label_signal.emit(f"Reached target fidelity after {i+1} iterations with FoM = {result['FoM']:.6f}")
                    break
                
                # Small delay to avoid UI freezing
                time.sleep(0.1)
            
            # Optimization finished
            if self._running:
                self.message_label_signal.emit("Optimization complete")
                
                # Send final results
                self.optimization_status_signal.emit({
                    'status': "Complete",
                    'message': f"Optimization complete. Best FoM = {self.best_fom:.6f} at iteration {self.best_iteration}",
                    'iterations': list(range(len(self.fom_history) + 1)),
                    'fom_values': [self.initial_fom] + self.fom_history,
                    'pulses': self.pulse_optimization['pulses'],
                    'timegrids': self.pulse_optimization['timegrids']
                })
                
        except Exception as e:
            self.log.error(f"Error during optimization: {str(e)}")
            self.message_label_signal.emit(f"Error: {str(e)}")
        finally:
            self._running = False
            self.is_running_signal.emit(False)

    def stop_optimization(self):
        """Stop the current optimization process, whether QUOCS or dummy hardware"""
        if self._running:
            self._running = False
            self.is_running_signal.emit(False)
            self.message_label_signal.emit("Optimization stopped by user")
            self.log.info("Optimization process stopped by user")
            
            # If using dummy hardware, stop the dummy pulser
            if self._use_dummy_hardware:
                try:
                    # Turn off pulser if it has the method
                    if hasattr(self.pulser, 'pulser_off'):
                        self.pulser_on_signal.emit(False)
                        self.log.debug("Dummy pulser turned off")
                        
                    # Stop the optimizer if it has the method
                    if hasattr(self.optimizer, 'stop_optimization'):
                        self.optimizer.stop_optimization()
                        self.log.debug("Dummy optimizer stopped")
                except Exception as e:
                    self.log.error(f"Error stopping dummy hardware: {str(e)}")

    def send_controls(self, pulses_list, parameters_list, timegrids_list):
        """ Send the controls to the worker controls """
        self.log.debug("Sending controls to the worker controls")
        self.send_controls_signal.emit(pulses_list, parameters_list, timegrids_list)

    def on_deactivate(self):
        """ Function called during the deactivation """
        # Stop any running optimization
        if self._running:
            self.stop_optimization()
            
        # Clean up hardware connections if needed
        if self._use_dummy_hardware:
            try:
                # Make sure pulser is off
                if hasattr(self.pulser, 'pulser_off'):
                    self.pulser_on_signal.emit(False)
                    
                # Clean up any other hardware resources
                self.log.info("Disconnecting from dummy hardware")
            except Exception as e:
                self.log.warning(f"Error during dummy hardware cleanup: {str(e)}")
        
        self.log.info("Closing the Optimization logic")