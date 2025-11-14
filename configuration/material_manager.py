"""
MaterialManager - Extends ParameterManager with material loading, mixing, and feeder management.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Union, Any, Optional

from configuration.process_parameters import ParameterManager


class MaterialManager(ParameterManager):
    """
    Material manager extending ParameterManager with material-specific functionality.

    Features:
    - Loads material properties from JSON files in the 'materials' directory
    - Handles automatic material mixing based on feeder settings
    - Manages feeder calibrations and flow rate conversions
    - Caches loaded materials for performance
    """

    # Materials directory relative to this file
    MATERIALS_DIR = Path(__file__).parent / "materials"

    # Material file storage mode: 'single-file' or 'multi-file'
    MATERIAL_FILE_MODE = 'multi-file'  # 'single-file' = all_materials.json, 'multi-file' = individual files

    # Feeder calibration constant (g/min per percent)
    FEED_SLOPE_G_PER_MIN_PER_PERCENT = 0.6419423077

    # Class-level cache for loaded material JSONs
    _material_cache = {}
    _all_materials_loaded = False

    @classmethod
    def load_material(cls, name: str) -> Dict[str, Any]:
        """
        Load material properties from JSON file with caching.

        Mode controlled by MATERIAL_FILE_MODE class attribute:
        - 'single-file': Load from all_materials.json
        - 'multi-file': Load from individual {name}.json files

        Args:
            name: Material name

        Returns:
            Dictionary with material properties

        Raises:
            FileNotFoundError: If material doesn't exist
        """
        # Check cache first
        if name in cls._material_cache:
            return cls._material_cache[name]

        if cls.MATERIAL_FILE_MODE == 'single-file':
            # Single-file mode: all_materials.json
            if not cls._all_materials_loaded:
                path = cls.MATERIALS_DIR / "all_materials.json"
                with open(path, 'r', encoding='utf-8') as f:
                    cls._material_cache = json.load(f)
                cls._all_materials_loaded = True

            if name not in cls._material_cache:
                raise FileNotFoundError(f"Material '{name}' not found in all_materials.json")

            return cls._material_cache[name]

        else:
            # Multi-file mode: individual files
            path = cls.MATERIALS_DIR / f"{name}.json"
            with open(path, 'r', encoding='utf-8') as f:
                material_data = json.load(f)

            cls._material_cache[name] = material_data
            return material_data

    @staticmethod
    def percent_to_kg_per_s(percent: float) -> float:
        """Convert feeder percentage to kg/s mass flow rate."""
        g_per_min = MaterialManager.FEED_SLOPE_G_PER_MIN_PER_PERCENT * percent
        return g_per_min / 60000.0  # Convert g/min to kg/s

    @staticmethod
    def L_per_min_to_m3_per_s(L_per_min: float) -> float:
        """Convert L/min gas flow to m³/s."""
        return L_per_min * 1e-3 / 60.0

    def request_change(self, **changes):
        """
        Request parameter changes with special handling for materials and gas flows.

        Special parameters:
        - 'material': Single material name (e.g., "316L")
        - 'feeders': Dict of material feeders and percentages (e.g., {"316L": 30.0, "17-4PH": 20.0})
        - 'carrier_gas_L_min': Carrier gas flow in L/min (converted to m³/s for gas_feed_rate)
        - 'shield_gas_L_min': Shield gas flow in L/min (stored as metadata)

        When 'material' is provided:
        - Loads the material's properties from JSON
        - Updates all material properties
        - Requires 'feeder_percent' to calculate powder_feed_rate

        When 'feeders' is provided:
        - Loads all specified materials
        - Calculates mass fractions based on flow rates
        - Mixes properties using appropriate mixing rules
        - Sets total powder_feed_rate

        Gas flows are converted and stored:
        - carrier_gas_L_min → gas_feed_rate (m³/s)
        - shield_gas_L_min → stored in metadata

        All other parameters are handled as normal (immediate or with response handlers).
        """
        # Extract special parameters
        material = changes.pop('material', None)
        feeders = changes.pop('feeders', None)
        carrier_gas_L_min = changes.pop('carrier_gas_L_min', None)
        shield_gas_L_min = changes.pop('shield_gas_L_min', None)
        feeder_percent = changes.pop('feeder_percent', None)

        # Handle single material
        if material is not None:
            if feeder_percent is None:
                raise ValueError("feeder_percent required when specifying single material")

            mat_data = self.load_material(material)

            # Update with material properties
            changes.update(mat_data['properties'])

            # Calculate powder feed rate
            changes['powder_feed_rate'] = self.percent_to_kg_per_s(feeder_percent)

            # Store metadata
            if '__material_meta__' not in self.data:
                self.data['__material_meta__'] = {}
            self.data['__material_meta__'].update({
                'materials': {material: 1.0},
                'feeders': {material: feeder_percent}
            })

        # Handle material mixture
        elif feeders is not None:
            materials = {}
            for mat_name in feeders:
                materials[mat_name] = self.load_material(mat_name)['properties']

            # Calculate mass flows and fractions
            mass_flows = {mat: self.percent_to_kg_per_s(pct)
                          for mat, pct in feeders.items()}
            total_flow = sum(mass_flows.values())

            if total_flow > 0:
                mass_fractions = {mat: flow / total_flow
                                  for mat, flow in mass_flows.items()}
            else:
                # Equal fractions if no flow
                mass_fractions = {mat: 1.0 / len(materials) for mat in materials}

            # Mix properties
            mixed = self._mix_properties(materials, mass_fractions)
            changes.update(mixed)

            # Set total powder feed rate
            changes['powder_feed_rate'] = total_flow

            # Store metadata
            if '__material_meta__' not in self.data:
                self.data['__material_meta__'] = {}
            self.data['__material_meta__'].update({
                'materials': mass_fractions,
                'feeders': feeders
            })

        # Handle gas flows
        if carrier_gas_L_min is not None:
            changes['gas_feed_rate'] = self.L_per_min_to_m3_per_s(carrier_gas_L_min)
            if '__material_meta__' not in self.data:
                self.data['__material_meta__'] = {}
            self.data['__material_meta__']['carrier_gas_L_min'] = carrier_gas_L_min

        if shield_gas_L_min is not None:
            if '__material_meta__' not in self.data:
                self.data['__material_meta__'] = {}
            self.data['__material_meta__']['shield_gas_L_min'] = shield_gas_L_min

        # Call parent method with processed changes
        super().request_change(**changes)

    def update_derived(self):
        """
        Update derived parameters with material-specific calculations.

        Extends parent update_derived to handle:
        - Recalculation of thermal diffusivity after mixing
        - Powder and gas flow rate conversions
        - Material mixture property updates
        """
        # Call parent update_derived first
        super().update_derived()

        # Additional material-specific derived parameters

        # Ensure thermal diffusivity is consistent after mixing
        if all(k in self for k in ['thermal_conductivity', 'density', 'specific_heat']):
            # Recalculate to ensure consistency
            self.data['thermal_diffusivity'] = (
                    self['thermal_conductivity'] / (self['density'] * self['specific_heat'])
            )

        # Update flow-related parameters if metadata exists
        if '__material_meta__' in self:
            meta = self['__material_meta__']

            # Ensure gas feed rate is updated if carrier gas was set
            if 'carrier_gas_L_min' in meta and 'gas_feed_rate' not in self:
                self.data['gas_feed_rate'] = self.L_per_min_to_m3_per_s(meta['carrier_gas_L_min'])

            # Calculate total mass flow from feeders if not set
            if 'feeders' in meta and 'powder_feed_rate' not in self:
                total_flow = sum(self.percent_to_kg_per_s(pct)
                                 for pct in meta['feeders'].values())
                self.data['powder_feed_rate'] = total_flow

    @staticmethod
    def _mix_properties(materials: Dict[str, Dict], fractions: Dict[str, float]) -> Dict:
        """
        Mix material properties using appropriate mixing rules.

        Mixing rules:
        - density: Harmonic mean (1/ρ_mix = Σ(w_i/ρ_i))
        - thermal_diffusivity: Recalculated from mixed k, ρ, cp
        - All others: Mass-weighted arithmetic mean

        Args:
            materials: Dict mapping material names to property dicts
            fractions: Dict mapping material names to mass fractions

        Returns:
            Dictionary of mixed properties
        """
        mixed = {}

        # Get all unique property names
        all_props = set()
        for mat_props in materials.values():
            all_props.update(mat_props.keys())

        for prop in all_props:
            # Get values for this property from all materials
            values = {}
            for mat, props in materials.items():
                if prop in props:
                    values[mat] = props[prop]

            # Skip if not all materials have this property
            if len(values) != len(materials):
                continue

            if prop == 'density':
                # Harmonic mean for density (based on specific volumes)
                mixed[prop] = 1.0 / sum(fractions[mat] / val for mat, val in values.items())
            else:
                # Arithmetic mean for most properties
                mixed[prop] = sum(fractions[mat] * val for mat, val in values.items())

        return mixed


if __name__ == "__main__":
    # ========================================================================
    # EXAMPLE USAGE AND TESTS
    # ========================================================================

    print("MaterialManager Tests")
    print("=" * 60)

    # Note: These tests assume you have created the materials directory
    # and the JSON files (316L.json, 17-4PH.json)

    try:
        # Test 1: Single material
        print("\nTest 1: Single material (316L)")
        print("-" * 40)

        manager = MaterialManager.from_defaults()
        manager.request_change(
            material="316L",
            feeder_percent=30.0,
            carrier_gas_L_min=6.0,
            shield_gas_L_min=8.0,
            laser_power=800
        )

        print(f"Material: 316L")
        print(f"Density: {manager['density']} kg/m³")
        print(f"Powder feed rate: {manager['powder_feed_rate'] * 60 * 1000:.3f} g/min")
        print(f"Gas feed rate: {manager['gas_feed_rate'] * 60 * 1000:.3f} L/min")
        print(f"Laser power: {manager['laser_power']} W")

        # Test 2: Material mixture
        print("\n\nTest 2: Material mixture (70% 316L, 30% 17-4PH)")
        print("-" * 40)

        manager.request_change(
            feeders={"316L": 40.0, "17-4PH": 17.14},  # Results in ~70/30 mix
            carrier_gas_L_min=7.0
        )

        meta = manager['__material_meta__']
        print(f"Materials: {', '.join(f'{m}: {f:.1%}' for m, f in meta['materials'].items())}")
        print(f"Mixed density: {manager['density']:.1f} kg/m³")
        print(f"Mixed melting temp: {manager['melting_temp']:.1f} K")
        print(f"Total powder feed: {manager['powder_feed_rate'] * 60 * 1000:.3f} g/min")

        # Test 3: Switching materials
        print("\n\nTest 3: Switching from mixture to single material")
        print("-" * 40)

        manager.request_change(
            material="17-4PH",
            feeder_percent=25.0
        )

        print(f"New material: 17-4PH")
        print(f"New density: {manager['density']} kg/m³")
        print(f"New feed rate: {manager['powder_feed_rate'] * 60 * 1000:.3f} g/min")

    except FileNotFoundError as e:
        print(f"\nNote: Material JSON files not found. Create the 'materials' directory")
        print(f"relative to this script and add 316L.json and 17-4PH.json files")
        print(f"with the structure shown in the comments above.")
        print(f"\nError: {e}")