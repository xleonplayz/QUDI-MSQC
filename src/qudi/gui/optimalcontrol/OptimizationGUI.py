from PySide2.QtWidgets import QMainWindow
from qudi.core.module import GuiBase
from qudi.core.connector import Connector
from quocspyside2interface.gui.OptimizationBasicGui import OptimizationBasicGui

class OptimizationGUI(GuiBase, QMainWindow, OptimizationBasicGui):
    """
    GUI-Klasse für das Optimal-Control-Plugin,
    die Qudi's GuiBase und ein QMainWindow einbindet.
    """

    optimization_logic = Connector(interface="OptimizationLogic")

    def __init__(self, config=None, **kwargs):
        # 1. Ein einziger super() Aufruf:
        super().__init__(config=config, **kwargs)

        # Danach kannst du beliebige lokale Initialisierung machen:
        # (z. B. self.setWindowTitle("Optimal Control"))
        # ...
    
    def on_activate(self):
        """Modul-Aktivierung in Qudi."""
        self.optimizationlogic = self.optimization_logic()
        self.optimizationlogic.load_optimization_dictionary_signal.connect(
            self.update_optimization_dictionary
        )
        self.handle_ui_elements()
        return 0

    def on_deactivate(self):
        """Modul-Deaktivierung."""
        self.close()
        return 0

    def show(self):
        super().show()
        self.activateWindow()
        self.raise_()
