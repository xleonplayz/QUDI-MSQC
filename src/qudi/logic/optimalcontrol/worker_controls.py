"""
Worker controls class for OptimalControl.

This module handles the control inputs for the optimization process.
Simplified version that doesn't rely on external dependencies.
"""
from qudi.core.module import LogicBase
from PySide2.QtCore import Signal, Slot

import time
import numpy as np

class WorkerControls(LogicBase):
    """Worker logic for controls manipulation"""
    # Define a proper _threaded attribute for the Logic module
    _threaded = True
    
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
        
        # The controls
        self.pulses, self.parameters, self.timegrids = None, None, None

    def on_activate(self):
        """Module called during activation"""
        self.log.info("Starting the Worker Controls")
        self.is_active = True
        return 0

    def on_deactivate(self):
        """Module called during deactivation"""
        self.log.info("Closing the Worker Controls")
        self.is_active = False
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
            
        # Store the controls
        self.pulses, self.parameters, self.timegrids = pulses, parameters, timegrids
        self.are_pulses_calculated = True
        
        # Log receipt of controls
        self.log.debug(f"Received new controls: {len(pulses)} pulse sequences")
        
        # In a real implementation, these controls would be sent to hardware
        # For simulation, we'll just log them
        self._log_controls()

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