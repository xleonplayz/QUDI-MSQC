"""
Worker class for figure of merit (FOM) calculations in OptimalControl.

This module calculates and provides the figure of merit for the optimization process.
Simplified version that doesn't rely on external dependencies.
"""
import time
import numpy as np
from PySide2.QtCore import Signal

from qudi.core.module import LogicBase

class WorkerFom(LogicBase):
    """Worker logic for figure of merit calculations"""
    # Define a proper _threaded attribute for the Logic module
    _threaded = True
    
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

    def on_activate(self):
        """Module called during the activation"""
        self.log.info("Worker FoM logic activated")
        return 0

    def on_deactivate(self):
        """Module called during the deactivation"""
        self.log.info("Worker FoM logic deactivated")
        return 0

    def set_max_time(self, max_time: float):
        """Set the maximum time for the calculation of the figure of merit"""
        self.max_time = max_time
        self.log.debug(f"Setting max calculation time to {max_time} seconds")

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

    def send_fom(self):
        """Send the dictionary containing the figure of merit and the status code to the optimization logic"""
        # Check if the worker is still active
        if not self.is_optimization_running:
            self.log.info("The worker FoM is not running")
            time.sleep(0.5)  # Brief delay
            return

        # For this simplified version, generate a random FoM if one hasn't been computed
        if not self.is_fom_computed:
            # Simple simulation: random value between 0 and 1
            random_fom = np.random.random()
            self.fom = random_fom
            self.std = random_fom * 0.1  # 10% of the value
            self.is_fom_computed = True
            self.log.debug(f"Generated simulated FoM: {self.fom}")

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

    def wait_for_fom(self, message):
        """Activate the waiting function for the figure of merit
        
        Parameters
        ----------
        message : str
            Message from the optimization logic
        """
        self.log.debug(f"FoM waiting triggered: {message}")
        self.send_fom()