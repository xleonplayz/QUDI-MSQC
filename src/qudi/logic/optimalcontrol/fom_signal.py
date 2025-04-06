"""
Figure of Merit (FOM) signal class for OptimalControl with QUOCS integration.

This file provides a wrapper class for figure of merit calculations, connecting
the optimization algorithm with the appropriate figure of merit computation sources.
"""
from typing import Callable, Dict, List, Any, Union, Optional
import numpy as np
import time

class FomSignal:
    """Wrapper class to handle figure of merit calculation callbacks"""

    def __init__(self, get_fom: Callable):
        """Constructor for an external figure of merit

        Parameters
        ----------
        get_fom : Callable
            Function that will be called to calculate the figure of merit
        """
        # Use the attribute of optimization logic class
        self.get_FoM = get_fom
        # Track performance stats
        self.call_count = 0
        self.total_time = 0.0
        self.best_fom = None
        self.best_pulses = None
        
    def __call__(self, pulses: List, parameters: List, timegrids: List) -> Dict[str, Any]:
        """Make the class callable, redirecting to the get_FoM method
        
        Parameters
        ----------
        pulses : List
            List of pulse sequences
        parameters : List
            List of parameters
        timegrids : List
            List of time grids
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with FoM calculation results
        """
        start_time = time.time()
        self.call_count += 1
        
        result = self.get_FoM(pulses, parameters, timegrids)
        
        # Calculate and store call time
        call_time = time.time() - start_time
        self.total_time += call_time
        
        # Track best FoM
        current_fom = result.get('FoM', None)
        if current_fom is not None:
            if self.best_fom is None or current_fom < self.best_fom:
                self.best_fom = current_fom
                # Store a copy of the pulses that gave the best result
                self.best_pulses = [list(p) for p in pulses]
                
        # Add performance info to the result
        result['call_time'] = call_time
        result['call_count'] = self.call_count
        result['avg_time'] = self.total_time / self.call_count if self.call_count > 0 else 0
        
        return result
        
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with performance statistics
        """
        return {
            'call_count': self.call_count,
            'total_time': self.total_time,
            'avg_time': self.total_time / self.call_count if self.call_count > 0 else 0,
            'best_fom': self.best_fom
        }
        
    def reset_stats(self):
        """Reset performance statistics"""
        self.call_count = 0
        self.total_time = 0.0
        self.best_fom = None
        self.best_pulses = None
        
    def get_best_pulses(self) -> Optional[List]:
        """Get the pulse sequence that gave the best FoM
        
        Returns
        -------
        Optional[List]
            List of best pulse sequences, or None if no calculation has been performed
        """
        return self.best_pulses