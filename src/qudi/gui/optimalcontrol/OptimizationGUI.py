"""
GUI for the Optimal Control plugin.
"""
from PySide2.QtWidgets import QMainWindow
from qudi.core.module import GuiBase
from qudi.core.connector import Connector
from quocspyside2interface.gui.OptimizationBasicGui import OptimizationBasicGui

class OptimizationGUI(GuiBase, QMainWindow, OptimizationBasicGui):
    """
    GUI class for the Optimal Control plugin,
    integrating Qudi's GuiBase and a QMainWindow.
    """

    # Define connectors to the Logic modules
    optimization_logic = Connector(interface="OptimizationLogic")

    def __init__(self, config=None, **kwargs):
        # Initialize the parent classes with a single super() call
        super().__init__(config=config, **kwargs)
        self.setWindowTitle("Optimal Control")
    
    def on_activate(self):
        """Module activation in Qudi."""
        # Get connected logic module
        self.optimizationlogic = self.optimization_logic()
        
        # Connect signals
        self.optimizationlogic.load_optimization_dictionary_signal.connect(
            self.update_optimization_dictionary
        )
        
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
        super().show()
        self.activateWindow()
        self.raise_()
