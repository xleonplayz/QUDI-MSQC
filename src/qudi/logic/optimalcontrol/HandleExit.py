"""
HandleExit class to manage optimization status and termination with QUOCS integration.

This class manages the coordination between the optimization process,
UI components, and QUOCS optimization library.
"""

from PySide2 import QtCore
from qudi.core.module import LogicBase
from qudi.core.logger import get_logger
from qudi.core.connector import Connector

try:
    import quocs
    QUOCS_AVAILABLE = True
except ImportError:
    QUOCS_AVAILABLE = False

class HandleExitLogic(LogicBase):
    """
    This class checks and updates the current optimization status and notifies
    all components about it (GUI, optimization logic, worker modules, and QUOCS).
    """
    # Define a proper _threaded attribute for the Logic module
    _threaded = True
    
    # Define connectors to the worker modules
    fom_worker = Connector(interface="WorkerFom", optional=True)
    controls_worker = Connector(interface="WorkerControls", optional=True)
    
    # Signals for components
    is_optimization_running_fom_signal = QtCore.Signal(bool)
    is_optimization_running_controls_signal = QtCore.Signal(bool)
    optimization_status_changed_signal = QtCore.Signal(dict)

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.is_user_running = False
        self.quocs_available = QUOCS_AVAILABLE
        
        # Optimization status information
        self.optimization_status = {
            'running': False,
            'algorithm': None,
            'iteration': 0,
            'max_iterations': 0,
            'fom': None,
            'start_time': None,
            'elapsed_time': 0,
            'status_message': 'Ready'
        }

    def on_activate(self):
        """Module activation in Qudi"""
        self.log.info("Starting the HandleExit logic")
        
        if not self.quocs_available:
            self.log.warning("QUOCS is not available! Using simulated optimization.")
        
        return 0

    def on_deactivate(self):
        """Module deactivation in Qudi"""
        self.log.info("Stopping the HandleExit logic")
        
        # Ensure optimization is stopped before deactivation
        if self.is_user_running:
            self.set_is_user_running(False)
            
        return 0

    @QtCore.Slot(bool)
    def set_is_user_running(self, is_running: bool):
        """
        Module connected with the Client Interface GUI. Controls optimization status
        when the user interacts with the GUI.
        
        Parameters
        ----------
        is_running : bool
            Whether the optimization should be running
        """
        self.is_user_running = is_running
        self.log.info(f"The optimization is running: {is_running}")
        
        # Update internal status
        self.optimization_status['running'] = is_running
        
        # If stopping, update status message
        if not is_running:
            self.optimization_status['status_message'] = 'Stopped by user' if self.optimization_status['iteration'] > 0 else 'Ready'
        else:
            self.optimization_status['status_message'] = 'Starting optimization...'
            
        # Emit status change signal
        self.optimization_status_changed_signal.emit(self.optimization_status)
        
        # Notify the worker components
        self.is_optimization_running_fom_signal.emit(is_running)
        self.is_optimization_running_controls_signal.emit(is_running)
        
        # Connect to worker modules if available
        try:
            if is_running and self.fom_worker.is_connected:
                self.fom_worker().update_optimization_status(is_running)
        except Exception as e:
            self.log.error(f"Error updating FoM worker status: {str(e)}")
            
        try:
            if is_running and self.controls_worker.is_connected:
                self.controls_worker().update_optimization_status(is_running)
        except Exception as e:
            self.log.error(f"Error updating Controls worker status: {str(e)}")
            
    def update_optimization_status(self, status_update: dict):
        """
        Update the optimization status with new information
        
        Parameters
        ----------
        status_update : dict
            Dictionary containing status information to update
        """
        # Update only the provided fields
        for key, value in status_update.items():
            if key in self.optimization_status:
                self.optimization_status[key] = value
                
        # Emit status change signal
        self.optimization_status_changed_signal.emit(self.optimization_status)
        self.log.debug(f"Updated optimization status: {status_update}")
        
    def get_optimization_status(self):
        """
        Get the current optimization status
        
        Returns
        -------
        dict
            Dictionary containing current optimization status
        """
        return self.optimization_status.copy()
