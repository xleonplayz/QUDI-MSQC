"""
Worker controls class for OptimalControl with QUOCS integration.

This module handles the control inputs for the optimization process.
"""
from qudi.core.module import LogicBase
from qudi.core.configoption import ConfigOption
from PySide2.QtCore import Signal, Slot

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    import quocs
    from quocs.pulses import PulseSequence
    from quocs.controls import Control
    QUOCS_AVAILABLE = True
except ImportError:
    QUOCS_AVAILABLE = False

class WorkerControls(LogicBase):
    """Worker logic for controls manipulation with QUOCS integration"""
    # Define a proper _threaded attribute for the Logic module
    _threaded = True
    
    # Configurables
    hw_output_enabled = ConfigOption('hw_output_enabled', False)
    pulse_clipping = ConfigOption('pulse_clipping', True)
    
    # Signals
    pulse_updated_signal = Signal(object)
    
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.log.info("Worker Controls initialization")
        self.fom_max = 10**10
        self.fom = 10**10
        self.status_code = 0
        self.max_time = 30.0  # Default timeout in seconds
        self.previous_time = time.time()
        self.is_optimization_running = False
        self.are_pulses_calculated = False
        self.is_active = False
        
        # QUOCS-specific attributes
        self.quocs_available = QUOCS_AVAILABLE
        
        # The controls
        self.pulses, self.parameters, self.timegrids = None, None, None
        
        # Pulse constraints
        self.max_amplitude = 1.0
        self.min_amplitude = -1.0
        self.constraints = {
            'apply_smoothing': False,
            'smoothing_factor': 0.1,
            'apply_rise_time': False,
            'rise_time': 0.1,
            'zero_boundaries': True,
            'apply_spectral_constraints': False,
            'min_frequency': 0.0,
            'max_frequency': 100.0
        }
        
        # History of controls for tracking optimization progress
        self.pulse_history = []
        self.last_pulse_update_time = time.time()
        self.pulse_update_count = 0

    def on_activate(self):
        """Module called during activation"""
        self.log.info("Starting the Worker Controls with QUOCS integration")
        self.is_active = True
        
        if not self.quocs_available:
            self.log.warning("QUOCS is not available! Some control features will be limited.")
            
        return 0

    def on_deactivate(self):
        """Module called during deactivation"""
        self.log.info("Closing the Worker Controls")
        self.is_active = False
        
        if self.pulse_update_count > 0:
            elapsed = time.time() - self.last_pulse_update_time
            self.log.info(f"Control statistics: {self.pulse_update_count} updates, "
                         f"avg interval: {elapsed/self.pulse_update_count:.4f}s")
        
        # Clear any stored controls
        self.pulses = None
        self.parameters = None
        self.timegrids = None
        self.pulse_history = []
        
        return 0

    def set_max_time(self, max_time: float):
        """Set the maximum time for waiting for the controls
        
        Parameters
        ----------
        max_time : float
            Maximum time in seconds
        """
        self.max_time = max_time
        self.log.debug(f"Setting max calculation time to {max_time} seconds")
        
    def set_amplitude_constraints(self, min_amplitude: float, max_amplitude: float):
        """Set amplitude constraints for the pulses
        
        Parameters
        ----------
        min_amplitude : float
            Minimum allowed amplitude
        max_amplitude : float
            Maximum allowed amplitude
        """
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.log.debug(f"Setting amplitude constraints: [{min_amplitude}, {max_amplitude}]")
        
    def set_constraints(self, constraints: Dict[str, Any]):
        """Set constraints for pulse generation
        
        Parameters
        ----------
        constraints : Dict[str, Any]
            Dictionary containing constraint settings
        """
        # Update only the provided constraints
        for key, value in constraints.items():
            if key in self.constraints:
                self.constraints[key] = value
                
        self.log.debug(f"Updated pulse constraints: {self.constraints}")

    @Slot(list, list, list)
    def update_controls(self, pulses, parameters, timegrids):
        """Update the controls with the ones provided by the optimization algorithm
        
        Parameters
        ----------
        pulses : list
            List of pulse sequences
        parameters : list
            List of parameters
        timegrids : list
            List of time grids
        """
        if not self.is_optimization_running:
            self.log.warning("The optimization is stopped. Controls will not be updated.")
            self.pulses, self.parameters, self.timegrids = None, None, None
            time.sleep(0.5)  # Brief delay
            return
            
        # Apply constraints to the pulses if enabled
        constrained_pulses = self._apply_constraints(pulses) if self.pulse_clipping else pulses
            
        # Store the controls
        self.pulses, self.parameters, self.timegrids = constrained_pulses, parameters, timegrids
        self.are_pulses_calculated = True
        
        # Optionally add to history
        if self.is_optimization_running:
            # Store a copy to prevent modification
            self.pulse_history.append({
                'pulses': [list(p) for p in constrained_pulses],
                'parameters': parameters.copy() if parameters else None,
                'timegrids': [list(t) for t in timegrids],
                'timestamp': time.time()
            })
            # Limit history size
            if len(self.pulse_history) > 100:
                self.pulse_history.pop(0)
        
        # Log receipt of controls
        self.log.debug(f"Received new controls: {len(constrained_pulses)} pulse sequences")
        
        # Update statistics
        now = time.time()
        if self.pulse_update_count > 0:
            interval = now - self.last_pulse_update_time
            self.log.debug(f"Time since last pulse update: {interval:.4f}s")
        self.last_pulse_update_time = now
        self.pulse_update_count += 1
        
        # Emit signal with pulse data
        self.pulse_updated_signal.emit({
            'pulses': constrained_pulses,
            'parameters': parameters,
            'timegrids': timegrids,
            'update_count': self.pulse_update_count
        })
        
        # Send to hardware if enabled
        if self.hw_output_enabled:
            self._send_to_hardware(constrained_pulses, parameters, timegrids)
        
        # Log the controls
        self._log_controls()
        
    def _apply_constraints(self, pulses: List) -> List:
        """Apply constraints to the pulses
        
        Parameters
        ----------
        pulses : List
            List of pulse sequences
            
        Returns
        -------
        List
            List of constrained pulse sequences
        """
        constrained_pulses = []
        
        for pulse in pulses:
            # Convert to numpy array for easier manipulation
            pulse_array = np.array(pulse)
            
            # Amplitude clipping
            if self.pulse_clipping:
                pulse_array = np.clip(pulse_array, self.min_amplitude, self.max_amplitude)
                
            # Zero boundaries if enabled
            if self.constraints['zero_boundaries'] and len(pulse_array) > 1:
                pulse_array[0] = 0.0
                pulse_array[-1] = 0.0
                
            # Apply smoothing if enabled
            if self.constraints['apply_smoothing'] and len(pulse_array) > 2:
                smoothing_factor = self.constraints['smoothing_factor']
                # Simple moving average smoothing
                window_size = max(3, int(len(pulse_array) * smoothing_factor * 0.1))
                if window_size % 2 == 0:
                    window_size += 1  # Ensure odd window size
                    
                if window_size < len(pulse_array):
                    kernel = np.ones(window_size) / window_size
                    # Use convolution with mode 'same' to preserve pulse length
                    smoothed = np.convolve(pulse_array, kernel, mode='same')
                    
                    # Preserve the original boundaries if required
                    if self.constraints['zero_boundaries']:
                        smoothed[0] = pulse_array[0]
                        smoothed[-1] = pulse_array[-1]
                        
                    pulse_array = smoothed
            
            # Apply rise time constraints if enabled
            if self.constraints['apply_rise_time'] and len(pulse_array) > 2:
                rise_time = self.constraints['rise_time']
                # Calculate number of samples in rise time
                rise_samples = max(2, int(rise_time * len(pulse_array) / 2))
                
                if rise_samples * 2 < len(pulse_array):
                    # Create rise and fall windows
                    rise_window = np.sin(np.linspace(0, np.pi/2, rise_samples))**2
                    fall_window = np.sin(np.linspace(np.pi/2, 0, rise_samples))**2
                    
                    # Apply windowing to beginning and end of pulse
                    pulse_array[:rise_samples] *= rise_window
                    pulse_array[-rise_samples:] *= fall_window
            
            # Convert back to list and add to result
            constrained_pulses.append(pulse_array.tolist())
            
        return constrained_pulses
        
    def _send_to_hardware(self, pulses, parameters, timegrids):
        """Send pulses to hardware (simulation)
        
        Parameters
        ----------
        pulses : list
            List of pulse sequences
        parameters : list
            List of parameters
        timegrids : list
            List of time grids
        """
        # This is just a simulated placeholder
        # In a real implementation, this would connect to your hardware
        self.log.debug("Simulating hardware output for pulses")
        
        # For detailed debugging, show pulse stats
        for i, pulse in enumerate(pulses):
            pulse_arr = np.array(pulse)
            stats = {
                'min': np.min(pulse_arr),
                'max': np.max(pulse_arr),
                'mean': np.mean(pulse_arr),
                'std': np.std(pulse_arr)
            }
            self.log.debug(f"Pulse {i+1} stats: {stats}")

    def _log_controls(self):
        """Log the current controls for debugging"""
        if self.pulses is None or not self.are_pulses_calculated:
            return
            
        try:
            # Get summary information about pulses
            pulse_count = len(self.pulses)
            pulse_lens = [len(p) for p in self.pulses]
            param_count = len(self.parameters) if self.parameters else 0
            
            self.log.debug(f"Controls: {pulse_count} pulses, lengths: {pulse_lens}, {param_count} parameters")
            
            # Add statistics for each pulse
            for i, pulse in enumerate(self.pulses):
                if pulse:
                    stats = {
                        'min': min(pulse),
                        'max': max(pulse),
                        'mean': sum(pulse) / len(pulse),
                    }
                    self.log.debug(f"Pulse {i+1} summary: {stats}")
        except Exception as e:
            self.log.error(f"Error logging controls: {str(e)}")

    @Slot(bool)
    def update_optimization_status(self, is_running):
        """Update the optimization running status
        
        Parameters
        ----------
        is_running : bool
            Whether the optimization is currently running
        """
        self.log.debug(f"Change the status of the optimization in the controls logic to: {is_running}")
        self.is_optimization_running = is_running
        
        if not is_running:
            # Reset pulse history when stopping
            self.pulse_history = []
            
    def get_latest_pulses(self) -> Tuple[List, List, List]:
        """Get the latest pulse data
        
        Returns
        -------
        Tuple[List, List, List]
            Tuple of (pulses, parameters, timegrids)
        """
        return self.pulses, self.parameters, self.timegrids
        
    def get_pulse_history(self) -> List[Dict[str, Any]]:
        """Get the pulse history
        
        Returns
        -------
        List[Dict[str, Any]]
            List of pulse history entries
        """
        return self.pulse_history