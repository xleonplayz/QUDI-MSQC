"""
NV-Zentrum-Simulator Logic Module.

Diese Logik-Klasse implementiert einen physikalischen Simulator für NV-Zentren in Diamant,
mit Fokus auf die Eigenschaften die für Pulsoptimierung relevant sind.
"""
import time
import numpy as np
from scipy.linalg import expm
from typing import Dict, List, Tuple, Optional, Union, Any

from qudi.core.module import LogicBase
from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.util.mutex import Mutex
from PySide2.QtCore import Signal

class NVSimulatorLogic(LogicBase):
    """
    Logic-Modul für die Simulation von NV-Zentren im Diamant mit Fokus auf Pulsoptimierung.
    
    Diese Klasse implementiert einen quantenmechanischen Simulator für NV-Zentren im Grundzustand,
    der für Pulsoptimierungsexperimente verwendet werden kann.
    """
    # Thread-Einstellung für das Logic-Modul
    _threaded = True
    
    # Konfigurationsoptionen
    initial_state = ConfigOption('initial_state', [1, 0, 0])  # |0⟩ Zustand als Standard
    d_gs = ConfigOption('d_gs', 2.87e9)  # Nullfeldaufspaltung in Hz
    e_gs = ConfigOption('e_gs', 2e6)     # Strain-Parameter in Hz
    gamma_e = ConfigOption('gamma_e', 28e6)  # gyromagnetisches Verhältnis in Hz/mT
    b_field = ConfigOption('b_field', [0, 0, 0.5])  # externes B-Feld in mT (Standard: 0.5 mT in z-Richtung)
    t1 = ConfigOption('t1', 100e-6)  # T1 Zeit in Sekunden
    t2 = ConfigOption('t2', 50e-6)   # T2 Zeit in Sekunden
    noise_level = ConfigOption('noise_level', 0.05)  # Rauschpegel für Messungen (5%)
    
    # Signale für die GUI
    sigStateUpdated = Signal(object)  # Sendet aktuellen Zustand
    sigMeasurementComplete = Signal(object)  # Sendet Messergebnisse
    sigSimulationProgress = Signal(float)  # Fortschrittsanzeige
    
    def __init__(self, config=None, **kwargs):
        """Initialisierung der NV-Simulator Logic."""
        super().__init__(config=config, **kwargs)
        
        # Interne Variablen
        self._mutex = Mutex()
        self._stop_requested = False
        self._simulation_running = False
        
        # Physikalische Parameter
        self.nv_state = None  # aktueller Zustand des NV-Zentrums
        self.hamiltonian = None  # statischer Hamiltonian
        self.pauli_matrices = None  # Pauli-Matrizen für S=1 System
        self.init_physical_parameters()
        
        # Protokoll-Parameter
        self.pulse_sequence = []  # Liste von (amplitude, phase, duration) Tupeln
        self.measurement_results = {}  # Speichert Ergebnisse von Simulationen
        
    def on_activate(self):
        """Wird beim Aktivieren des Moduls aufgerufen"""
        self.log.info("NV-Zentrum-Simulator aktiviert")
        
        # Initialisiere den Zustand des NV-Zentrums
        self.reset_state()
        
        # Berechne den statischen Hamiltonian basierend auf den Konfigurationsparametern
        self.update_hamiltonian()
        
        return 0
        
    def on_deactivate(self):
        """Wird beim Deaktivieren des Moduls aufgerufen"""
        self.log.info("NV-Zentrum-Simulator deaktiviert")
        
        # Sicherstellen, dass laufende Simulationen gestoppt werden
        with self._mutex:
            self._stop_requested = True
            
        # Warte, bis laufende Simulationen beendet sind
        while self._simulation_running:
            time.sleep(0.1)
            
        return 0
        
    def init_physical_parameters(self):
        """Initialisiere die physikalischen Parameter und Operatoren"""
        # Erstelle Pauli-Matrizen für S=1
        sx = 1/np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        sy = 1j/np.sqrt(2) * np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]], dtype=complex)
        sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
        
        self.pauli_matrices = {
            'sx': sx,
            'sy': sy,
            'sz': sz,
            'sx2': np.matmul(sx, sx),
            'sy2': np.matmul(sy, sy),
            'sz2': np.matmul(sz, sz),
            'id': np.eye(3, dtype=complex)
        }
            
    def update_hamiltonian(self, b_field=None, e_gs=None, d_gs=None):
        """Aktualisiere den statischen Hamiltonian des NV-Zentrums
        
        Parameters
        ----------
        b_field : np.ndarray, optional
            Externes Magnetfeld in mT, [Bx, By, Bz]
        e_gs : float, optional
            Strain-Parameter in Hz
        d_gs : float, optional
            Nullfeldaufspaltung in Hz
        """
        # Verwende die übergebenen Parameter oder die Standardwerte
        b_field = b_field if b_field is not None else self.b_field
        e_gs = e_gs if e_gs is not None else self.e_gs
        d_gs = d_gs if d_gs is not None else self.d_gs
        
        # Umwandeln des B-Feld-Arrays in ein NumPy-Array, falls nötig
        if not isinstance(b_field, np.ndarray):
            b_field = np.array(b_field)
            
        # Berechne den statischen Hamiltonian: H = D*Sz^2 + E*(Sx^2 - Sy^2) + gamma_e*(Bx*Sx + By*Sy + Bz*Sz)
        h_static = (
            d_gs * self.pauli_matrices['sz2'] + 
            e_gs * (self.pauli_matrices['sx2'] - self.pauli_matrices['sy2']) +
            self.gamma_e * (
                b_field[0] * self.pauli_matrices['sx'] +
                b_field[1] * self.pauli_matrices['sy'] +
                b_field[2] * self.pauli_matrices['sz']
            )
        )
        
        self.hamiltonian = h_static
        self.log.debug(f"Hamiltonian aktualisiert: D={d_gs/1e6} MHz, E={e_gs/1e6} MHz, B={b_field} mT")
        return h_static
    
    def reset_state(self, initial_state=None):
        """Setze den Zustand des NV-Zentrums zurück
        
        Parameters
        ----------
        initial_state : list, optional
            Anfangszustand als Liste [c0, c+, c-] für |0⟩, |+1⟩, |-1⟩
        """
        if initial_state is None:
            initial_state = self.initial_state
            
        # Umwandeln in NumPy-Array und normalisieren
        state = np.array(initial_state, dtype=complex)
        state = state / np.linalg.norm(state)
        
        # Dichteoperator erstellen: rho = |psi⟩⟨psi|
        self.nv_state = np.outer(state, state.conj())
        
        # Signalisiere die Zustandsänderung
        self.sigStateUpdated.emit({
            'state_vector': state,
            'density_matrix': self.nv_state,
            'populations': np.real(np.diag(self.nv_state))
        })
        
        self.log.debug(f"NV-Zustand zurückgesetzt: {state}")
        return state
    
    def apply_pulse(self, amplitude: float, phase: float, duration: float, dt: float = 1e-9):
        """Wende einen Mikrowellenpuls auf den NV-Zustand an
        
        Parameters
        ----------
        amplitude : float
            Pulsamplitude (normalisiert, typ. -1 bis 1)
        phase : float
            Pulsphase in Radiant
        duration : float
            Pulsdauer in Sekunden
        dt : float, optional
            Zeitschritt für die Simulation in Sekunden, Default: 1 ns
            
        Returns
        -------
        np.ndarray
            Aktualisierter Zustand des NV-Zentrums (Dichteoperator)
        """
        with self._mutex:
            if self.nv_state is None:
                self.reset_state()
            
            # Puls-Hamiltonian erstellen: H_pulse = amplitude * (cos(phase)*Sx + sin(phase)*Sy)
            h_pulse = amplitude * (
                np.cos(phase) * self.pauli_matrices['sx'] + 
                np.sin(phase) * self.pauli_matrices['sy']
            )
            
            # Gesamthamiltonian: H = H_static + H_pulse
            h_total = self.hamiltonian + h_pulse
            
            # Zeitentwicklung für den Puls
            steps = int(duration / dt)
            u_step = expm(-1j * 2 * np.pi * h_total * dt)  # Zeitentwicklungsoperator für einen Schritt
            
            # Wende Zeitentwicklung auf den Zustand an und berücksichtige Dekohärenz
            for step in range(steps):
                # Unitäre Zeitentwicklung: rho' = U * rho * U†
                self.nv_state = np.matmul(u_step, np.matmul(self.nv_state, u_step.conj().T))
                
                # Einfaches Dekohärenzmodell für T1 und T2
                self._apply_decoherence(dt)
                
                # Prüfe auf Stoppanforderung
                if self._stop_requested:
                    break
            
            # Signalisiere die Zustandsänderung
            self.sigStateUpdated.emit({
                'state_vector': None,  # Nicht mehr wohldefiniert für gemischte Zustände
                'density_matrix': self.nv_state,
                'populations': np.real(np.diag(self.nv_state))
            })
                
            return self.nv_state
    
    def _apply_decoherence(self, dt: float):
        """Wende ein vereinfachtes Dekohärenzmodell auf den NV-Zustand an
        
        Parameters
        ----------
        dt : float
            Zeitschritt in Sekunden
        """
        # Extrahiere Besetzungswahrscheinlichkeiten
        populations = np.real(np.diag(self.nv_state)).copy()
        
        # T1-Relaxation: Annäherung an thermisches Gleichgewicht
        # Im Gleichgewicht bei Raumtemperatur sind fast alle Spins im |0⟩ Zustand
        equilibrium_pops = np.array([1.0, 0.0, 0.0])
        t1_decay = np.exp(-dt / self.t1)
        new_pops = populations * t1_decay + equilibrium_pops * (1 - t1_decay)
        
        # Aktualisiere die Diagonalelemente
        for i in range(3):
            self.nv_state[i, i] = new_pops[i]
        
        # T2-Dekohärenz: Reduktion der Nicht-Diagonalelemente
        t2_decay = np.exp(-dt / self.t2)
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.nv_state[i, j] *= t2_decay
    
    def apply_pulse_sequence(self, pulse_sequence: List[Tuple[float, float, float]], dt: float = 1e-9):
        """Wende eine Sequenz von Pulsen auf den NV-Zustand an
        
        Parameters
        ----------
        pulse_sequence : list
            Liste von (amplitude, phase, duration) Tupeln für jeden Puls
        dt : float, optional
            Zeitschritt für die Simulation in Sekunden
            
        Returns
        -------
        np.ndarray
            Aktualisierter Zustand des NV-Zentrums
        """
        self._simulation_running = True
        self._stop_requested = False
        
        self.log.info(f"Starte Simulation von {len(pulse_sequence)} Pulsen")
        start_time = time.time()
        
        try:
            # Für jeden Puls in der Sequenz
            for i, (amplitude, phase, duration) in enumerate(pulse_sequence):
                # Puls anwenden
                self.apply_pulse(amplitude, phase, duration, dt)
                
                # Fortschritt aktualisieren
                progress = (i + 1) / len(pulse_sequence)
                self.sigSimulationProgress.emit(progress)
                
                # Prüfe auf Stoppanforderung
                if self._stop_requested:
                    self.log.info("Simulation wurde vorzeitig gestoppt")
                    break
            
            end_time = time.time()
            self.log.info(f"Simulation abgeschlossen in {end_time - start_time:.3f} s")
            
            return self.nv_state
            
        finally:
            self._simulation_running = False
    
    def measure_population(self, add_noise: bool = True) -> np.ndarray:
        """Misst die Besetzungswahrscheinlichkeiten der NV-Zustände
        
        Parameters
        ----------
        add_noise : bool, optional
            Gibt an, ob realistisches Messrauschen hinzugefügt werden soll
            
        Returns
        -------
        np.ndarray
            Gemessene Besetzungswahrscheinlichkeiten [p0, p+, p-]
        """
        with self._mutex:
            if self.nv_state is None:
                self.reset_state()
                
            # Reale Besetzungswahrscheinlichkeiten
            populations = np.real(np.diag(self.nv_state))
            
            # Füge Messrauschen hinzu
            if add_noise:
                noise = np.random.normal(0, self.noise_level, size=3)
                populations = np.clip(populations + noise, 0, 1)
                # Renormalisiere, damit die Summe 1 bleibt
                populations = populations / np.sum(populations)
            
            return populations
    
    def simulate_rabi_oscillation(self, max_time: float, steps: int, amplitude: float = 1.0):
        """Simuliert ein Rabi-Oszillationsexperiment
        
        Parameters
        ----------
        max_time : float
            Maximale Pulsdauer in Sekunden
        steps : int
            Anzahl der Messpunkte
        amplitude : float, optional
            Pulsamplitude, standardmäßig 1.0
            
        Returns
        -------
        dict
            Ergebnisse des Experiments mit Zeitpunkten und Besetzungswahrscheinlichkeiten
        """
        self._simulation_running = True
        self._stop_requested = False
        
        # Zeitpunkte
        time_points = np.linspace(0, max_time, steps)
        
        # Ergebnisarrays initialisieren
        pop0 = np.zeros(steps)
        pop_plus = np.zeros(steps)
        pop_minus = np.zeros(steps)
        
        try:
            # Für jeden Zeitpunkt
            for i, t in enumerate(time_points):
                # Zustand zurücksetzen
                self.reset_state()
                
                # Puls anwenden
                self.apply_pulse(amplitude, 0.0, t)
                
                # Besetzungswahrscheinlichkeiten messen
                pops = self.measure_population()
                pop0[i] = pops[0]
                pop_plus[i] = pops[1]
                pop_minus[i] = pops[2]
                
                # Fortschritt aktualisieren
                progress = (i + 1) / steps
                self.sigSimulationProgress.emit(progress)
                
                # Prüfe auf Stoppanforderung
                if self._stop_requested:
                    self.log.info("Rabi-Simulation wurde vorzeitig gestoppt")
                    break
                
            # Ergebnisse zusammenstellen
            results = {
                'time_points': time_points[:i+1],
                'population_0': pop0[:i+1],
                'population_plus': pop_plus[:i+1],
                'population_minus': pop_minus[:i+1]
            }
            
            # Speichere und signalisiere Ergebnisse
            self.measurement_results['rabi'] = results
            self.sigMeasurementComplete.emit({'type': 'rabi', 'data': results})
            
            return results
            
        finally:
            self._simulation_running = False
    
    def simulate_ramsey(self, free_evolution_time: float, steps: int):
        """Simuliert ein Ramsey-Experiment (freie Präzession)
        
        Parameters
        ----------
        free_evolution_time : float
            Maximale freie Präzessionszeit in Sekunden
        steps : int
            Anzahl der Messpunkte
            
        Returns
        -------
        dict
            Ergebnisse des Experiments
        """
        self._simulation_running = True
        self._stop_requested = False
        
        # Zeitpunkte
        time_points = np.linspace(0, free_evolution_time, steps)
        
        # Ergebnisarrays initialisieren
        pop0 = np.zeros(steps)
        pop_plus = np.zeros(steps)
        pop_minus = np.zeros(steps)
        
        try:
            # Für jeden Zeitpunkt
            for i, t in enumerate(time_points):
                # Zustand zurücksetzen
                self.reset_state()
                
                # π/2-Puls anwenden (um Kohärenz zu erzeugen)
                # Bestimme π/2-Pulsdauer basierend auf dem Hamiltonian
                pi_over_2_time = self._estimate_pi_pulse_time() / 2
                self.apply_pulse(1.0, 0.0, pi_over_2_time)
                
                # Freie Evolution für Zeit t
                # Dies entspricht einem "Puls" mit Amplitude 0
                self.apply_pulse(0.0, 0.0, t)
                
                # Zweiten π/2-Puls anwenden
                self.apply_pulse(1.0, 0.0, pi_over_2_time)
                
                # Besetzungswahrscheinlichkeiten messen
                pops = self.measure_population()
                pop0[i] = pops[0]
                pop_plus[i] = pops[1]
                pop_minus[i] = pops[2]
                
                # Fortschritt aktualisieren
                progress = (i + 1) / steps
                self.sigSimulationProgress.emit(progress)
                
                # Prüfe auf Stoppanforderung
                if self._stop_requested:
                    self.log.info("Ramsey-Simulation wurde vorzeitig gestoppt")
                    break
                
            # Ergebnisse zusammenstellen
            results = {
                'time_points': time_points[:i+1],
                'population_0': pop0[:i+1],
                'population_plus': pop_plus[:i+1],
                'population_minus': pop_minus[:i+1]
            }
            
            # Speichere und signalisiere Ergebnisse
            self.measurement_results['ramsey'] = results
            self.sigMeasurementComplete.emit({'type': 'ramsey', 'data': results})
            
            return results
            
        finally:
            self._simulation_running = False
    
    def simulate_spin_echo(self, max_time: float, steps: int):
        """Simuliert ein Spin-Echo-Experiment (Hahn-Echo)
        
        Parameters
        ----------
        max_time : float
            Maximale Gesamtzeit in Sekunden
        steps : int
            Anzahl der Messpunkte
            
        Returns
        -------
        dict
            Ergebnisse des Experiments
        """
        self._simulation_running = True
        self._stop_requested = False
        
        # Zeitpunkte
        time_points = np.linspace(0, max_time, steps)
        
        # Ergebnisarrays initialisieren
        pop0 = np.zeros(steps)
        pop_plus = np.zeros(steps)
        pop_minus = np.zeros(steps)
        
        try:
            # Für jeden Zeitpunkt
            for i, t in enumerate(time_points):
                # Zustand zurücksetzen
                self.reset_state()
                
                # π/2-Puls anwenden
                pi_over_2_time = self._estimate_pi_pulse_time() / 2
                pi_time = 2 * pi_over_2_time
                
                # Spin-Echo-Sequenz: π/2 - τ - π - τ - π/2
                self.apply_pulse(1.0, 0.0, pi_over_2_time)  # π/2-Puls
                self.apply_pulse(0.0, 0.0, t/2)  # Freie Evolution für τ = t/2
                self.apply_pulse(1.0, 0.0, pi_time)  # π-Puls
                self.apply_pulse(0.0, 0.0, t/2)  # Freie Evolution für τ = t/2
                self.apply_pulse(1.0, 0.0, pi_over_2_time)  # π/2-Puls
                
                # Besetzungswahrscheinlichkeiten messen
                pops = self.measure_population()
                pop0[i] = pops[0]
                pop_plus[i] = pops[1]
                pop_minus[i] = pops[2]
                
                # Fortschritt aktualisieren
                progress = (i + 1) / steps
                self.sigSimulationProgress.emit(progress)
                
                # Prüfe auf Stoppanforderung
                if self._stop_requested:
                    self.log.info("Spin-Echo-Simulation wurde vorzeitig gestoppt")
                    break
                
            # Ergebnisse zusammenstellen
            results = {
                'time_points': time_points[:i+1],
                'population_0': pop0[:i+1],
                'population_plus': pop_plus[:i+1],
                'population_minus': pop_minus[:i+1]
            }
            
            # Speichere und signalisiere Ergebnisse
            self.measurement_results['spin_echo'] = results
            self.sigMeasurementComplete.emit({'type': 'spin_echo', 'data': results})
            
            return results
            
        finally:
            self._simulation_running = False
    
    def _estimate_pi_pulse_time(self):
        """Schätzt die Dauer eines π-Pulses basierend auf dem Hamiltonian
        
        Returns
        -------
        float
            Geschätzte Zeit für einen π-Puls in Sekunden
        """
        # Für einen Spin-1 System mit Zeeman- und Nullfeldaufspaltung ist das nicht trivial.
        # Eine einfache Näherung: wir verwenden eine Rabi-Frequenz von 10 MHz für eine Amplitude von 1.0
        # In einer realen Implementierung würde dies genauer berechnet werden.
        rabi_frequency = 10e6  # 10 MHz
        pi_time = 1 / (2 * rabi_frequency)  # Zeit für einen π-Puls
        
        return pi_time
    
    def optimize_pulse(self, initial_state, target_state, max_duration, max_amplitude=1.0):
        """Schnittstelle zur Integration mit dem Pulsoptimierungsmodul
        
        Parameters
        ----------
        initial_state : np.ndarray
            Anfangszustand des NV-Zentrums
        target_state : np.ndarray
            Zielzustand des NV-Zentrums
        max_duration : float
            Maximale Pulsdauer in Sekunden
        max_amplitude : float, optional
            Maximale Pulsamplitude, standardmäßig 1.0
            
        Returns
        -------
        dict
            Optimierungsergebnisse
        """
        # Diese Methode würde als Schnittstelle zur Pulsoptimierung dienen
        # Sie würde die Kostenfunktion (Figure of Merit) berechnen und 
        # für die Optimierung zurückgeben
        
        # In einer vollständigen Implementierung würde hier die Integration
        # mit dem Pulsoptimierungsmodul erfolgen
        
        # Beispiel für eine Kostenfunktion: Fidelität zwischen Ziel- und erreichtem Zustand
        self.log.info("Pulsoptimierung angefordert (noch nicht implementiert)")
        
        # Platzhalter für Optimierungsergebnisse
        results = {
            'success': False,
            'message': 'Pulsoptimierung noch nicht implementiert',
            'initial_state': initial_state,
            'target_state': target_state,
            'max_duration': max_duration,
            'max_amplitude': max_amplitude
        }
        
        return results
    
    def calculate_fidelity(self, state1, state2):
        """Berechnet die Fidelität zwischen zwei Zuständen
        
        Parameters
        ----------
        state1 : np.ndarray
            Erster Zustand (Dichteoperator)
        state2 : np.ndarray
            Zweiter Zustand (Dichteoperator)
            
        Returns
        -------
        float
            Fidelität zwischen den Zuständen
        """
        # Für reine Zustände: F = |⟨ψ|φ⟩|²
        # Für gemischte Zustände: F = Tr[√(√ρ σ √ρ)]²
        
        # Vereinfachte Implementierung für den Fall, dass mindestens einer der Zustände rein ist
        if state1.shape[0] == state2.shape[0] == 3:  # 3x3 Dichteoperatoren
            # Nehmen wir an, beide sind Dichteoperatoren
            sqrt_state1 = np.sqrt(state1)
            inner = np.matmul(np.matmul(sqrt_state1, state2), sqrt_state1)
            
            # Eigenwerte der inneren Matrix berechnen
            eigenvalues = np.linalg.eigvals(inner)
            
            # Fidelität berechnen
            fidelity = np.real(np.sum(np.sqrt(eigenvalues)))**2
            
            return fidelity
        else:
            self.log.error("Fidelitätsberechnung fehlgeschlagen: Ungültige Zustandsformate")
            return 0.0