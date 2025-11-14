"""
ParameterManager - A focused implementation for managing simulation parameters.
"""

import numpy as np
from typing import Dict, Any, Union, Optional
from collections import UserDict
from abc import ABC, abstractmethod


class ParameterResponse(ABC):
    """
    Base class for parameter response specifications.

    Handles the logic of managing parameter change requests with time-based responses.
    Subclasses must implement the compute method to define specific response behavior.
    """

    def __init__(self):
        self.target_value = None
        self.last_update_time = None

    def request(self, target_value: float):
        """Register a new target value (overrides previous)."""
        self.target_value = target_value

    def _has_pending_request(self):
        """Check if there's a pending request."""
        return self.target_value is not None

    def _clear_if_reached(self, value: float):
        """Clear target if we've reached it."""
        if value == self.target_value:
            self.target_value = None

    def __call__(self, current_value: float, time: float) -> float:
        """
        Update the parameter value based on time.

        Tracks time automatically and calls compute() for the actual logic.
        """
        self._clear_if_reached(current_value)  # Check if already at target

        if not self._has_pending_request():
            return current_value

        new_value = self.compute(current_value, self.target_value, time, self.last_update_time)
        self.last_update_time = time
        self._clear_if_reached(new_value)  # Check if we've now reached target

        return new_value

    @abstractmethod
    def compute(self, current_value: float, target_value: float, time: float, last_time: Optional[float]) -> float:
        """
        Compute the parameter value at the given time.

        Args:
            current_value: Current parameter value
            target_value: Target parameter value
            time: Current simulation time
            last_time: Last update time (None on first call)

        Returns:
            New parameter value
        """
        pass


class ParameterManager(UserDict):
    """
    Parameter manager that acts as a dictionary while maintaining parameter calculation logic.

    Inherits from UserDict to provide full dictionary functionality while
    encapsulating parameter calculation and update logic.
    """

    def __init__(self, parameter_responses: dict = None, **kwargs):
        """
        Initialize with parameter values and optional response specifications.

        Args:
            parameter_responses: Dict mapping parameter names to ParameterResponse instances
            **kwargs: Parameter values
        """
        super().__init__()

        # Store response specifications
        self.parameter_responses = parameter_responses or {}

        # Store initial parameters as protected
        self.__init_params_kwargs = kwargs.copy()

        # Initialize data using reset logic
        self.reset()

    @classmethod
    def from_defaults(
        cls,
        parameter_responses: dict = None,
        # Material properties
        density: float = 7850,  # kg/m³
        specific_heat: float = 460,  # J/(kg·K)
        thermal_diffusivity: float = 1.172e-5,  # m²/s
        surface_tension: float = 1.88,  # N/m
        viscosity: float = 0.0058,  # kg/(m·s)
        epsilon: float = 0.013,  # universal constant

        # Laser parameters
        laser_power: float = 650,  # W
        laser_absorptivity: float = 0.6,
        laser_radius: float = 0.43e-3,  # m

        # Process parameters
        powder_feed_rate: float = 2 * (1/1000) * (1/60),  # kg/s
        scan_speed: float = 0.003,  # m/s
        particle_radius: float = 42e-6,  # m
        gas_feed_rate: float = 2.5/60000,  # m³/s

        # Geometric parameters
        nozzle_angle: float = np.radians(50),  # radians
        nozzle_height: float = 8.1e-3,  # m
        nozzle_radius: float = 0.7e-3,  # m
        powder_divergence_angle: float = np.radians(5.8),  # radians

        # Temperature parameters
        initial_temp: float = 298,  # K
        melting_temp: float = 1811,  # K
        ambient_temp: float = 298,  # K

        # Beam characteristics
        beam_waist_radius: float = 0.43e-3,  # m
        beam_divergence_angle: float = 0.02,  # rad
        beam_waist_position: float = 13.81e-3,  # m

        # Phase change parameters
        latent_heat_of_fusion: float = 204500,  # J/kg
        solidus_temp: float = 1788,  # K
        liquidus_temp: float = 1811,  # K

        **kwargs  # Additional parameters
    ):
        """
        Create ParameterManager with default values.

        Args:
            parameter_responses: Dict mapping parameter names to ParameterResponse instances
            All other args: Parameter values with defaults
        """
        # Collect all parameters
        params = {
            'density': density,
            'specific_heat': specific_heat,
            'thermal_diffusivity': thermal_diffusivity,
            'surface_tension': surface_tension,
            'viscosity': viscosity,
            'epsilon': epsilon,
            'laser_power': laser_power,
            'laser_absorptivity': laser_absorptivity,
            'laser_radius': laser_radius,
            'powder_feed_rate': powder_feed_rate,
            'scan_speed': scan_speed,
            'particle_radius': particle_radius,
            'gas_feed_rate': gas_feed_rate,
            'nozzle_angle': nozzle_angle,
            'nozzle_height': nozzle_height,
            'nozzle_radius': nozzle_radius,
            'powder_divergence_angle': powder_divergence_angle,
            'initial_temp': initial_temp,
            'melting_temp': melting_temp,
            'ambient_temp': ambient_temp,
            'beam_waist_radius': beam_waist_radius,
            'beam_divergence_angle': beam_divergence_angle,
            'beam_waist_position': beam_waist_position,
            'latent_heat_of_fusion': latent_heat_of_fusion,
            'solidus_temp': solidus_temp,
            'liquidus_temp': liquidus_temp,
        }
        params.update(kwargs)

        return cls(parameter_responses=parameter_responses, **params)

    @property
    def _init_params_kwargs(self) -> Dict[str, Any]:
        """Protected property to access initial parameter kwargs."""
        return self.__init_params_kwargs.copy()

    def reset(self):
        """Reset parameters to their initial values."""
        self.data.clear()
        self.data.update(self.__init_params_kwargs)
        self.update_derived()

    @staticmethod
    def calc_powder_stream_radius_at_z_equal_0(params: Union[Dict, 'ParameterManager']) -> float:
        """
        Calculate the average powder stream radius at the substrate surface (z=0).
        See Appendix in Huang et al. (2019a)
        """
        phi = params['nozzle_angle']
        r0 = params['nozzle_radius']
        H = params['nozzle_height']
        theta = params['powder_divergence_angle']
        r_avg = r0 + H * np.tan(theta) / np.sin(phi)
        return r_avg

    @staticmethod
    def calc_laser_radius_at_z_equal_0(params: Union[Dict, 'ParameterManager']) -> float:
        """
        Calculate the laser beam radius at the substrate surface (z=0).
        Equation 6 (and in Appendix) in Huang et al. (2019a)
        """
        R0L = params['beam_waist_radius']
        theta_L = params['beam_divergence_angle']
        z0 = params['beam_waist_position']
        RL = np.sqrt((R0L ** 2) + 4 * (theta_L ** 2) * (z0 ** 2))
        return RL

    @staticmethod
    def calc_particle_velocity(params: Union[Dict, 'ParameterManager']) -> float:
        """Calculate particle velocity based on gas feed rate and nozzle radius."""
        gas_feed_rate = params['gas_feed_rate']
        nozzle_radius = params['nozzle_radius']
        nozzle_area = np.pi * nozzle_radius ** 2
        return gas_feed_rate / nozzle_area

    def update_derived(self):
        """
        Update all derived parameters based on current values.

        Calculates:
        - particle_mass: Mass of single spherical particle (4/3 * π * r³ * ρ)
        - thermal_conductivity: k = α * ρ * cp (Fourier's law)
        - average_powder_stream_radius: Powder stream radius at substrate
        - laser_beam_radius: Laser beam radius at substrate
        - particle_velocity: Particle velocity from gas flow
        """
        # Only calculate if required parameters exist
        if 'particle_radius' in self and 'density' in self:
            self.data['particle_mass'] = (4/3) * np.pi * self['particle_radius']**3 * self['density']

        if all(k in self for k in ['thermal_diffusivity', 'density', 'specific_heat']):
            self.data['thermal_conductivity'] = (
                self['thermal_diffusivity'] * self['density'] * self['specific_heat']
            )

        # Calculate powder stream radius if all required parameters exist
        if all(k in self for k in ['nozzle_angle', 'nozzle_radius', 'nozzle_height', 'powder_divergence_angle']):
            self.data['average_powder_stream_radius'] = self.calc_powder_stream_radius_at_z_equal_0(self)

        # Calculate laser beam radius if all required parameters exist
        if all(k in self for k in ['beam_waist_radius', 'beam_divergence_angle', 'beam_waist_position']):
            self.data['laser_beam_radius'] = self.calc_laser_radius_at_z_equal_0(self)

        # Calculate particle velocity if all required parameters exist
        if all(k in self for k in ['gas_feed_rate', 'nozzle_radius']):
            self.data['particle_velocity'] = self.calc_particle_velocity(self)

    def request_change(self, **changes):
        """
        Request parameter changes (phase 1).

        Args:
            **changes: Parameter changes requested

        For parameters with response specifications in parameter_responses,
        registers the request. Otherwise, changes are immediate.
        """
        for param, new_value in changes.items():
            if param in self.parameter_responses:
                self.parameter_responses[param].request(new_value)
            else:
                # Immediate change for parameters without response specs
                self.data[param] = new_value

    def process_requested_changes(self, time: float):
        """
        Process all requested parameter changes based on time (phase 2).

        Args:
            time: Current simulation time

        Updates all parameters with response handlers based on elapsed time,
        then recalculates derived parameters.
        """
        for param, response in self.parameter_responses.items():
            self.data[param] = response(self.data[param], time)

        # Update derived parameters after all changes
        self.update_derived()


def set_params(**kwargs) -> ParameterManager:
    """
    Backward compatible function that creates parameters dictionary.

    Returns ParameterManager which acts as a dictionary.
    """
    return ParameterManager.from_defaults(**kwargs)


if __name__ == "__main__":

    # ========================================================================
    # TEST 1: BASIC USAGE WITH DEFAULT VALUES
    # ========================================================================
    print("TEST 1: Basic usage with default values")
    print("=" * 60)

    # Create parameter manager with all defaults
    params = ParameterManager.from_defaults()

    # Display key parameters and calculated values
    print(f"Laser power: {params['laser_power']} W")
    print(f"Scan speed: {params['scan_speed'] * 1000} mm/s")
    print(f"Powder stream radius: {params['average_powder_stream_radius'] * 1000:.3f} mm")
    print(f"Laser beam radius: {params['laser_beam_radius'] * 1000:.3f} mm")
    print(f"Particle velocity: {params['particle_velocity']:.3f} m/s")
    print(f"Particle mass: {params['particle_mass']:.3e} kg")
    print(f"Thermal conductivity: {params['thermal_conductivity']:.3f} W/(m·K)")

    # ========================================================================
    # TEST 2: DIRECT PARAMETER UPDATES
    # ========================================================================
    print("\n\nTEST 2: Direct parameter updates")
    print("=" * 60)

    # Direct dictionary-style updates
    params['nozzle_height'] = 10e-3  # Change to 10mm
    params['gas_feed_rate'] = 3.0/60000  # Change gas flow

    # Must manually trigger recalculation for direct updates
    params.update_derived()

    print(f"New nozzle height: {params['nozzle_height'] * 1000} mm")
    print(f"New powder stream radius: {params['average_powder_stream_radius'] * 1000:.3f} mm")
    print(f"New particle velocity: {params['particle_velocity']:.3f} m/s")

    # ========================================================================
    # TEST 3: BACKWARD COMPATIBILITY
    # ========================================================================
    print("\n\nTEST 3: Backward compatibility with set_params")
    print("=" * 60)

    # Old-style function call that returns ParameterManager instance
    old_params = set_params(
        laser_power=750,
        scan_speed=0.004
    )

    print(f"Type returned: {type(old_params)}")
    print(f"Laser power: {old_params['laser_power']} W")
    print(f"Particle velocity: {old_params['particle_velocity']:.3f} m/s")

    # ========================================================================
    # TEST 4: INITIAL PARAMETERS AND RESET
    # ========================================================================
    print("\n\nTEST 4: Initial parameters and reset")
    print("=" * 60)

    # Create with minimal parameters using from_defaults for proper initialization
    params = ParameterManager.from_defaults(laser_power=600, scan_speed=0.003)

    # Modify current values
    params['laser_power'] = 900
    params['scan_speed'] = 0.005
    params.update_derived()

    print(f"Initial laser power: {params._init_params_kwargs['laser_power']} W")
    print(f"Current laser power: {params['laser_power']} W")
    print(f"Current scan speed: {params['scan_speed'] * 1000} mm/s")

    # Reset to initial values
    params.reset()
    print(f"After reset - laser power: {params['laser_power']} W")
    print(f"After reset - scan speed: {params['scan_speed'] * 1000} mm/s")

    # ========================================================================
    # TEST 5: PARAMETER RESPONSE SYSTEM
    # ========================================================================
    print("\n\nTEST 5: Parameter response system with time delays")
    print("=" * 60)

    # Define a simple linear ramp response
    class LinearRampResponse(ParameterResponse):
        """Linear ramp response over fixed duration."""

        def __init__(self, ramp_time: float):
            super().__init__()
            self.ramp_time = ramp_time
            self.start_value = None
            self.start_time = None

        def compute(self, current_value: float, target_value: float,
                   time: float, last_time: Optional[float]) -> float:
            # Initialize transition on first call
            if last_time is None or self.start_time is None:
                self.start_value = current_value
                self.start_time = time

            # Calculate progress
            elapsed = time - self.start_time
            if elapsed >= self.ramp_time:
                return target_value  # Transition complete

            # Linear interpolation
            progress = elapsed / self.ramp_time
            return self.start_value + (target_value - self.start_value) * progress

    # Create parameter manager with response handlers
    params = ParameterManager.from_defaults(
        parameter_responses={
            'laser_power': LinearRampResponse(ramp_time=0.5),  # 500ms ramp
            'powder_feed_rate': LinearRampResponse(ramp_time=1.0)  # 1s ramp
        }
    )

    print(f"Initial laser power: {params['laser_power']} W")
    print(f"Initial powder feed: {params['powder_feed_rate'] * 60 * 1000:.3f} g/min")

    # Request changes (phase 1)
    params.request_change(
        laser_power=800,
        powder_feed_rate=3e-5,
        scan_speed=0.005  # This changes immediately (no response handler)
    )

    print(f"\nAfter request (before processing):")
    print(f"  Scan speed: {params['scan_speed'] * 1000:.1f} mm/s (immediate)")
    print(f"  Laser power: {params['laser_power']:.1f} W (pending)")

    # Simulate time progression (phase 2)
    print(f"\nTime progression:")
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t in times:
        params.process_requested_changes(t)
        print(f"  t={t:.2f}s: Laser={params['laser_power']:.1f}W, "
              f"Feed={params['powder_feed_rate']*60*1000:.3f}g/min")

    # ========================================================================
    # TEST 6: CUSTOM PARAMETERS AND DICTIONARY OPERATIONS
    # ========================================================================
    print("\n\nTEST 6: Custom parameters and dictionary operations")
    print("=" * 60)

    # Add custom parameters
    params = ParameterManager.from_defaults(
        custom_param1=42,
        custom_param2="test_value",
        custom_param3=np.array([1, 2, 3])
    )

    print(f"Total parameters: {len(params)}")
    print(f"Custom param 1: {params['custom_param1']}")
    print(f"Custom param 2: {params['custom_param2']}")
    print(f"Custom param 3: {params['custom_param3']}")

    # Dictionary operations
    print("\nDictionary operations:")
    print(f"  'laser_power' in params: {'laser_power' in params}")
    print(f"  'nonexistent' in params: {'nonexistent' in params}")

    # List some parameters by category
    derived_params = ['particle_mass', 'thermal_conductivity', 'particle_velocity',
                     'average_powder_stream_radius', 'laser_beam_radius']
    print(f"\nDerived parameters present: {sum(1 for p in derived_params if p in params)}/{len(derived_params)}")