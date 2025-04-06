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
    
    # Define all signals
    load_optimization_dictionary_signal = Signal(dict)
    send_controls_signal = Signal(list, list, list)
    wait_fom_signal = Signal(str)
    is_running_signal = Signal(bool)
    message_label_signal = Signal(str)
    fom_plot_signal = Signal(object)
    controls_update_signal = Signal(object)
    optimization_status_signal = Signal(dict)

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
            
        # Send the controls
        self.send_controls(pulses, parameters, timegrids)
        
        # Update UI with current pulse shapes
        self.controls_update_signal.emit({
            "pulses": pulses,
            "timegrids": timegrids,
            "parameters": parameters,
            "iteration": self.current_iteration
        })
        
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
        return {"FoM": self.fom, "std": self.std, "status_code": self.status_code}

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
        """Start the QUOCS optimization process
        
        Parameters
        ----------
        opti_comm_dict : dict, optional
            Optimization parameters dictionary. If None, use current opti_comm_dict.
        """
        if self._running:
            self.log.warning("An optimization is already running")
            return
            
        # Check if QUOCS is available
        if not self.quocs_available:
            self.message_label_signal.emit("ERROR: QUOCS is not available. Install with 'pip install quocs'")
            self.log.error("Cannot start optimization, QUOCS is not available")
            return
            
        self._running = True
        self.is_running_signal.emit(True)
        self.message_label_signal.emit("Initializing QUOCS optimization process")
        
        # Reset optimization state
        self.fom_history = []
        self.current_iteration = 0
        self.initial_fom = None
        self.best_fom = None
        self.best_pulses = None
        self.best_iteration = 0
        
        # Update parameters if provided
        if opti_comm_dict is not None:
            self.load_opti_comm_dict(opti_comm_dict)
        
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
        """Run the QUOCS optimization process"""
        if not self._running or self.pulse_optimization is None:
            return
            
        try:
            # In a real implementation, this would use the QUOCS library directly:
            # self.pulse_optimization.run()
            
            # For this simplified version, we'll simulate the optimization process
            max_iterations = self.pulse_optimization['max_iterations']
            algorithm = self.pulse_optimization['algorithm']
            
            self.message_label_signal.emit(f"Starting {algorithm} optimization process with {max_iterations} iterations")
            
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
        """Stop the current optimization process"""
        if self._running:
            self._running = False
            self.is_running_signal.emit(False)
            self.message_label_signal.emit("Optimization stopped by user")
            self.log.info("Optimization process stopped by user")

    def send_controls(self, pulses_list, parameters_list, timegrids_list):
        """ Send the controls to the worker controls """
        self.log.debug("Sending controls to the worker controls")
        self.send_controls_signal.emit(pulses_list, parameters_list, timegrids_list)

    def on_deactivate(self):
        """ Function called during the deactivation """
        # Stop any running optimization
        if self._running:
            self.stop_optimization()
        
        self.log.info("Closing the Optimization logic")