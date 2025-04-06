"""
Worker class for figure of merit (FOM) calculations in OptimalControl with QUOCS integration.

This module calculates and provides the figure of merit for the optimization process.
"""
import time
import numpy as np
from PySide2.QtCore import Signal
from typing import Dict, Any, Optional, List, Callable

from qudi.core.module import LogicBase
from qudi.core.configoption import ConfigOption

try:
    import quocs
    QUOCS_AVAILABLE = True
except ImportError:
    QUOCS_AVAILABLE = False

class WorkerFom(LogicBase):
    """Worker logic for figure of merit calculations with QUOCS integration"""
    # Define a proper _threaded attribute for the Logic module
    _threaded = True
    
    # Configure options
    fom_default_target = ConfigOption('fom_default_target', 0.99)
    simulation_mode = ConfigOption('simulation_mode', True)
    
    # Signal definitions
    send_fom_signal = Signal(dict)
    test_message_signal = Signal(str)

    def __init__(self, config=None, **kwargs):
        """Initialize the base class"""
        super().__init__(config=config, **kwargs)
        self.fom_max = 10**10
        self.fom = self.fom_max
        self.std = 0.0
        self.status_code = 0
        self.is_optimization_running = False
        self.is_fom_computed = False
        self.max_time = 30.0  # Default timeout in seconds
        
        # QUOCS-specific attributes
        self.quocs_available = QUOCS_AVAILABLE
        self.target_fidelity = self.fom_default_target
        self.fom_function = None
        self.current_pulses = None
        self.current_parameters = None 
        self.current_timegrids = None
        
        # Statistics for FoM calculation
        self.fom_calc_count = 0
        self.total_calc_time = 0.0

    def on_activate(self):
        """Module called during the activation"""
        self.log.info("Worker FoM logic activated")
        
        if not self.quocs_available and not self.simulation_mode:
            self.log.warning("QUOCS is not available and simulation mode is disabled. FoM calculations will fail.")
        elif not self.quocs_available:
            self.log.info("QUOCS is not available. Running in simulation mode.")
        
        return 0

    def on_deactivate(self):
        """Module called during the deactivation"""
        self.log.info("Worker FoM logic deactivated")
        
        # Log FoM calculation statistics
        if self.fom_calc_count > 0:
            avg_time = self.total_calc_time / self.fom_calc_count
            self.log.info(f"FoM calculation statistics: {self.fom_calc_count} calls, "
                         f"avg time: {avg_time:.4f}s, total time: {self.total_calc_time:.2f}s")
        
        return 0

    def set_max_time(self, max_time: float):
        """Set the maximum time for the calculation of the figure of merit"""
        self.max_time = max_time
        self.log.debug(f"Setting max calculation time to {max_time} seconds")
        
    def set_target_fidelity(self, target_fidelity: float):
        """Set the target fidelity for optimization
        
        Parameters
        ----------
        target_fidelity : float
            Target fidelity value (0-1)
        """
        self.target_fidelity = max(0.0, min(1.0, target_fidelity))
        self.log.debug(f"Setting target fidelity to {self.target_fidelity}")

    def set_fom_function(self, fom_function: Callable):
        """Set a custom function for figure of merit calculation
        
        Parameters
        ----------
        fom_function : Callable
            Function that takes (pulses, parameters, timegrids) and returns a FoM value
        """
        self.fom_function = fom_function
        self.log.info("Custom FoM function set")

    def update_fom(self, fom: float, std: float = 0.0, status_code: int = 0):
        """Update the figure of merit values
        
        Parameters
        ----------
        fom : float
            The figure of merit value
        std : float, optional
            Standard deviation of the FoM, by default 0.0
        status_code : int, optional
            Status code for the calculation (0=ok, <0=error), by default 0
        """
        self.log.debug(f"Updating FoM: {fom}, std: {std}, status: {status_code}")
        self.fom = fom
        self.std = std
        self.status_code = status_code
        self.is_fom_computed = True

    def calculate_fom(self, pulses: List, parameters: List, timegrids: List) -> Dict[str, Any]:
        """Calculate the figure of merit for given pulses
        
        Parameters
        ----------
        pulses : List
            List of pulse sequences
        parameters : List
            List of parameters for each pulse
        timegrids : List
            List of time grids for each pulse
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with FoM, std, and status_code
        """
        # Store current pulse data
        self.current_pulses = pulses
        self.current_parameters = parameters
        self.current_timegrids = timegrids
        
        start_time = time.time()
        self.fom_calc_count += 1
        
        try:
            # Use custom FoM function if available
            if self.fom_function is not None:
                fom_result = self.fom_function(pulses, parameters, timegrids)
                if isinstance(fom_result, dict):
                    fom = fom_result.get('FoM', fom_result.get('fom', None))
                    std = fom_result.get('std', 0.0)
                    status_code = fom_result.get('status_code', 0)
                else:
                    # Assume the function returned just the FoM value
                    fom = float(fom_result)
                    std = 0.0
                    status_code = 0
                    
                self.update_fom(fom, std, status_code)
                
            # Use QUOCS for FoM calculation if available and not in simulation mode
            elif self.quocs_available and not self.simulation_mode:
                # In a real implementation, this would use QUOCS to calculate the FoM
                # For example:
                # from quocs.costfunctions import StateFidelity
                # cost_function = StateFidelity(target_state)
                # fom = cost_function.compute(pulses, parameters, timegrids)
                
                # For now, we'll simulate it
                self._simulate_quocs_fom(pulses, parameters, timegrids)
                
            # Use simulation mode
            else:
                self._simulate_fom(pulses, parameters, timegrids)
                
        except Exception as e:
            self.log.error(f"Error calculating FoM: {str(e)}")
            self.update_fom(self.fom_max, 0.0, -1)
            
        # Calculate and store calculation time
        calc_time = time.time() - start_time
        self.total_calc_time += calc_time
        self.log.debug(f"FoM calculation took {calc_time:.4f}s")
        
        return {
            "FoM": self.fom,
            "std": self.std,
            "status_code": self.status_code
        }
        
    def _simulate_quocs_fom(self, pulses, parameters, timegrids):
        """Simulate a QUOCS FoM calculation with realistic behavior
        
        Parameters
        ----------
        pulses : List
            List of pulse sequences
        parameters : List
            List of parameters for each pulse
        timegrids : List
            List of time grids for each pulse
        """
        # Convert to numpy arrays for calculations
        pulse_arrays = [np.array(p) for p in pulses]
        
        # Calculate a "fidelity" based on pulse smoothness and area
        # This simulates a common goal in quantum control: smooth pulses with specific area
        fidelities = []
        for pulse in pulse_arrays:
            # Calculate smoothness (penalize rapid changes)
            if len(pulse) > 1:
                smoothness = 1.0 - min(1.0, np.mean(np.abs(np.diff(pulse))) * 5.0)
            else:
                smoothness = 1.0
                
            # Calculate normalized pulse area (target is around 0.5)
            if len(pulse) > 0:
                area = np.mean(np.abs(pulse))
                area_score = 1.0 - min(1.0, abs(area - 0.5) * 2.0)
            else:
                area_score = 0.0
                
            # Combine scores (70% smoothness, 30% area)
            fidelity = 0.7 * smoothness + 0.3 * area_score
            fidelities.append(fidelity)
            
        # Combine fidelities from multiple pulses (if any)
        if fidelities:
            fom = 1.0 - np.mean(fidelities)  # Convert to error (lower is better)
            std = np.std(fidelities) if len(fidelities) > 1 else 0.0
        else:
            fom = 0.5  # Default mediocre value
            std = 0.0
            
        # Add a bit of noise
        fom += np.random.normal(0, 0.01)
        fom = max(0.0, min(1.0, fom))  # Clamp to [0,1]
        
        self.update_fom(fom, std, 0)

    def _simulate_fom(self, pulses, parameters, timegrids):
        """Simulate a simple FoM calculation for testing
        
        Parameters
        ----------
        pulses : List
            List of pulse sequences
        parameters : List
            List of parameters for each pulse
        timegrids : List
            List of time grids for each pulse
        """
        # Simple random FoM that improves over time
        if not hasattr(self, '_sim_iter'):
            self._sim_iter = 0
        else:
            self._sim_iter += 1
            
        # Decay function that approaches the target (more iterations = better FoM)
        # Starts at around 0.5 (mediocre) and approaches 0.01 (excellent)
        decay_rate = 0.97
        base_fom = 0.5 * (decay_rate ** self._sim_iter)
        
        # Add noise that decreases with iterations
        noise_scale = 0.1 * (decay_rate ** self._sim_iter)
        noise = np.random.normal(0, noise_scale)
        
        # Final FoM (lower is better in optimization)
        fom = max(0.01, base_fom + noise)  # Ensure minimum of 0.01
        std = noise_scale
        
        self.update_fom(fom, std, 0)

    def send_fom(self):
        """Send the dictionary containing the figure of merit and the status code to the optimization logic"""
        # Check if the worker is still active
        if not self.is_optimization_running:
            self.log.info("The worker FoM is not running")
            time.sleep(0.5)  # Brief delay
            return

        # Calculate FoM if we have pulse data but FoM hasn't been computed yet
        if not self.is_fom_computed and self.current_pulses is not None:
            self.calculate_fom(self.current_pulses, self.current_parameters, self.current_timegrids)

        # For simulation, generate a random FoM if one still hasn't been computed
        if not self.is_fom_computed and self.simulation_mode:
            # Simple simulation: random value between 0 and 1
            random_fom = np.random.random()
            self.fom = random_fom
            self.std = random_fom * 0.1  # 10% of the value
            self.is_fom_computed = True
            self.log.debug(f"Generated fallback simulated FoM: {self.fom}")

        # Send the FoM data
        self.send_fom_signal.emit({
            "FoM": self.fom, 
            "std": self.std, 
            "status_code": self.status_code
        })
        
        # Reset the computation flag
        self.is_fom_computed = False

    def update_optimization_status(self, is_running):
        """Update the optimization running status
        
        Parameters
        ----------
        is_running : bool
            Whether the optimization is currently running
        """
        self.log.debug(f"Change the status of the optimization in the worker FoM to: {is_running}")
        self.is_optimization_running = is_running
        
        if not is_running:
            # Reset simulation variables when stopping
            if hasattr(self, '_sim_iter'):
                self._sim_iter = 0

    def wait_for_fom(self, message):
        """Activate the waiting function for the figure of merit
        
        Parameters
        ----------
        message : str
            Message from the optimization logic
        """
        self.log.debug(f"FoM waiting triggered: {message}")
        self.send_fom()