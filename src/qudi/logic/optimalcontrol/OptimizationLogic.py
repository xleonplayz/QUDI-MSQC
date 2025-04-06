"""
This is the logic class for the optimization - simplified version.
"""
import time
import numpy as np
from PySide2.QtCore import Signal

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.util.mutex import Mutex

class OptimizationLogic(LogicBase):
    """Logic module for optimization control - simplified version"""
    # Define a proper _threaded attribute for the Logic module
    _threaded = True

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

    def on_activate(self):
        """ Activation """
        self.log.info("Starting the Optimization Logic")
        
        # Connect signals between components
        self.send_controls_signal.connect(self.controls_logic().update_controls)
        self.fom_logic().send_fom_signal.connect(self.update_FoM)
        self.wait_fom_signal.connect(self.fom_logic().wait_for_fom)
        
        # Notify that we're ready
        self.message_label_signal.emit("Optimization logic activated and ready")

    def update_FoM(self, fom_dict):
        """ Update the figure of merit from the fom logic """
        self.status_code = fom_dict.setdefault("status_code", 0)
        self.std = fom_dict.setdefault("std", 0.0)
        self.fom = fom_dict["FoM"]
        self.is_fom_computed = True
        
        # Emit signal to update plot
        self.fom_plot_signal.emit({
            "fom": self.fom,
            "std": self.std,
            "status": self.status_code
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
        """ Load the opti communication dictionary and send it to the GUI """
        self.opti_comm_dict = opti_comm_dict
        self.load_optimization_dictionary_signal.emit(opti_comm_dict)
        self.log.info(f"Loaded optimization dictionary with {len(opti_comm_dict)} entries")

    def start_optimization(self, opti_comm_dict=None):
        """Start a simplified optimization process"""
        if self._running:
            self.log.warning("An optimization is already running")
            return
            
        self._running = True
        self.is_running_signal.emit(True)
        self.message_label_signal.emit("Starting optimization process")
        
        if opti_comm_dict is not None:
            self.load_opti_comm_dict(opti_comm_dict)
        
        # In a real implementation, this would start the actual optimization
        # For this simplified version, we'll just simulate some basic behavior
        try:
            # Simulate optimization with random pulses
            for i in range(5):
                if not self._running:
                    break
                    
                # Generate random pulses as example
                pulses = [np.random.random(10).tolist()]
                params = [{"amplitude": np.random.random()}]
                timegrids = [list(range(10))]
                
                # Send controls and get FoM
                result = self.get_FoM(pulses, params, timegrids)
                
                # Log progress
                self.message_label_signal.emit(f"Iteration {i+1}/5: FoM = {result['FoM']:.4f}")
                time.sleep(1)  # Simulate processing time
                
            self.message_label_signal.emit("Optimization complete")
            
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