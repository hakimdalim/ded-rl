"""
Generate random material property files for single-track experiments.

For each process parameter combination, generates 3 material variants with
randomly sampled independent properties. Dependent properties (like thermal
diffusivity) are calculated from the independent ones.

Independent properties (sampled):
- density
- specific_heat
- thermal_conductivity (used to calculate thermal_diffusivity)
- surface_tension
- viscosity
- epsilon (emissivity)
- laser_absorptivity
- melting_temp (average of solidus and liquidus)
- solidus_temp
- liquidus_temp
- latent_heat_of_fusion
- particle_radius

Dependent properties (calculated):
- thermal_diffusivity = thermal_conductivity / (density * specific_heat)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import argparse


class MaterialPropertyRanges:
    """Define sensible ranges for material properties based on 316L and 17-4PH"""
    
    # Independent properties with (min, max) ranges
    # Based on typical stainless steel properties with some variation
    RANGES = {
        # Density (kg/m³) - typical steel range
        'density': (7400.0, 8200.0),
        
        # Specific heat (J/(kg·K)) - room temperature value
        'specific_heat': (450.0, 580.0),
        
        # Thermal conductivity (W/(m·K)) - room temperature for stainless steels
        'thermal_conductivity': (11.0, 18.0),
        
        # Surface tension (N/m) - molten steel at melting point
        'surface_tension': (1.2, 1.8),
        
        # Viscosity (kg/(m·s)) - molten steel
        'viscosity': (0.004, 0.007),
        
        # Emissivity (dimensionless) - surface dependent
        'epsilon': (0.25, 0.45),
        
        # Laser absorptivity (dimensionless) - process dependent
        'laser_absorptivity': (0.35, 0.65),
        
        # Melting temperatures (K)
        'solidus_temp': (1600.0, 1760.0),
        'liquidus_temp': (1680.0, 1800.0),
        
        # Latent heat of fusion (J/kg)
        'latent_heat_of_fusion': (240000.0, 320000.0),
        
        # Particle radius (m) - typical powder size distribution
        'particle_radius': (1.2e-05, 2.0e-05),  # 12-20 µm radius = 24-40 µm diameter
    }
    
    @staticmethod
    def sample_property(prop_name: str, rng: np.random.Generator) -> float:
        """Sample a property value from its defined range"""
        min_val, max_val = MaterialPropertyRanges.RANGES[prop_name]
        return rng.uniform(min_val, max_val)
    
    @staticmethod
    def calculate_thermal_diffusivity(k: float, rho: float, cp: float) -> float:
        """Calculate thermal diffusivity from conductivity, density, and specific heat
        
        α = k / (ρ * cp)
        
        Args:
            k: thermal conductivity (W/(m·K))
            rho: density (kg/m³)
            cp: specific heat (J/(kg·K))
            
        Returns:
            thermal diffusivity (m²/s)
        """
        return k / (rho * cp)
    
    @staticmethod
    def validate_melting_temps(solidus: float, liquidus: float) -> Tuple[float, float]:
        """Ensure solidus < liquidus, swap if needed"""
        if solidus >= liquidus:
            # Add minimum gap of 20K
            liquidus = solidus + 20.0
        return solidus, liquidus
    
    @staticmethod
    def calculate_melting_temp(solidus: float, liquidus: float) -> float:
        """Calculate average melting temperature"""
        return (solidus + liquidus) / 2.0


def generate_random_material(
    base_name: str,
    variant_id: int,
    seed: int,
    rng: np.random.Generator
) -> Dict[str, Any]:
    """Generate a random material with physically reasonable properties
    
    Args:
        base_name: Base material identifier (e.g., "random_mat")
        variant_id: Variant number (0-2)
        seed: Random seed used for this material
        rng: Random number generator
        
    Returns:
        Dictionary containing material properties and metadata
    """
    
    # Sample independent properties
    density = MaterialPropertyRanges.sample_property('density', rng)
    specific_heat = MaterialPropertyRanges.sample_property('specific_heat', rng)
    thermal_conductivity = MaterialPropertyRanges.sample_property('thermal_conductivity', rng)
    surface_tension = MaterialPropertyRanges.sample_property('surface_tension', rng)
    viscosity = MaterialPropertyRanges.sample_property('viscosity', rng)
    epsilon = MaterialPropertyRanges.sample_property('epsilon', rng)
    laser_absorptivity = MaterialPropertyRanges.sample_property('laser_absorptivity', rng)
    solidus_temp = MaterialPropertyRanges.sample_property('solidus_temp', rng)
    liquidus_temp = MaterialPropertyRanges.sample_property('liquidus_temp', rng)
    latent_heat_of_fusion = MaterialPropertyRanges.sample_property('latent_heat_of_fusion', rng)
    particle_radius = MaterialPropertyRanges.sample_property('particle_radius', rng)
    
    # Validate and correct melting temperatures
    solidus_temp, liquidus_temp = MaterialPropertyRanges.validate_melting_temps(
        solidus_temp, liquidus_temp
    )
    melting_temp = MaterialPropertyRanges.calculate_melting_temp(solidus_temp, liquidus_temp)
    
    # Calculate dependent property
    thermal_diffusivity = MaterialPropertyRanges.calculate_thermal_diffusivity(
        thermal_conductivity, density, specific_heat
    )
    
    # Construct material name
    material_name = f"{base_name}_v{variant_id}"
    
    # Build material dictionary
    material = {
        "name": material_name,
        "properties": {
            "density": density,
            "specific_heat": specific_heat,
            "thermal_diffusivity": thermal_diffusivity,
            "surface_tension": surface_tension,
            "viscosity": viscosity,
            "epsilon": epsilon,
            "laser_absorptivity": laser_absorptivity,
            "melting_temp": melting_temp,
            "solidus_temp": solidus_temp,
            "liquidus_temp": liquidus_temp,
            "latent_heat_of_fusion": latent_heat_of_fusion,
            "particle_radius": particle_radius
        },
        "metadata": {
            "generation_info": {
                "type": "randomly_generated",
                "variant_id": variant_id,
                "seed": seed,
                "base_name": base_name,
                "generation_date": "2024-11-10"
            },
            "property_calculation": {
                "thermal_diffusivity": {
                    "formula": "α = k / (ρ * cp)",
                    "calculated_from": {
                        "thermal_conductivity": f"{thermal_conductivity:.6e} W/(m·K)",
                        "density": f"{density:.6e} kg/m³",
                        "specific_heat": f"{specific_heat:.6e} J/(kg·K)"
                    }
                },
                "melting_temp": {
                    "formula": "(T_solidus + T_liquidus) / 2",
                    "calculated_from": {
                        "solidus_temp": f"{solidus_temp:.6e} K",
                        "liquidus_temp": f"{liquidus_temp:.6e} K"
                    }
                }
            },
            "property_ranges_used": {
                key: {"min": min_val, "max": max_val}
                for key, (min_val, max_val) in MaterialPropertyRanges.RANGES.items()
            },
            "units": {
                "density": "kg/m³",
                "specific_heat": "J/(kg·K)",
                "thermal_diffusivity": "m²/s",
                "thermal_conductivity": "W/(m·K)",
                "surface_tension": "N/m",
                "viscosity": "kg/(m·s)",
                "epsilon": "dimensionless",
                "laser_absorptivity": "dimensionless",
                "melting_temp": "K",
                "solidus_temp": "K",
                "liquidus_temp": "K",
                "latent_heat_of_fusion": "J/kg",
                "particle_radius": "m"
            },
            "notes": (
                "This is a synthetically generated material with randomly sampled properties "
                "within physically reasonable ranges for stainless steels. Independent properties "
                "were sampled from uniform distributions. Dependent properties (thermal_diffusivity, "
                "melting_temp) were calculated from the independent ones."
            )
        }
    }
    
    return material


def generate_materials_for_experiment(
    experiment_id: int,
    power: float,
    diameter: float,
    feed: float,
    speed: float,
    output_dir: Path,
    n_variants: int = 3,
    master_seed: int = 42,
    materials_dict: dict = None
) -> list:
    """Generate n_variants random materials for a specific experiment

    Args:
        experiment_id: Unique experiment identifier
        power: Laser power (W)
        diameter: Spot diameter (mm)
        feed: Powder feed rate (g/min)
        speed: Scan speed (mm/s)
        output_dir: Directory to save material JSON files (used only if materials_dict is None)
        n_variants: Number of material variants to generate
        master_seed: Master random seed
        materials_dict: Optional dict to accumulate materials (for single-file mode)

    Returns:
        List of generated material names
    """

    # Create a unique seed for this experiment
    exp_seed = master_seed + experiment_id
    rng = np.random.default_rng(exp_seed)

    # Base name encodes the process parameters
    base_name = f"random_P{int(power)}W_D{diameter:.2f}mm_F{feed:.1f}gmin_V{int(speed)}mms"

    material_names = []

    for variant_id in range(n_variants):
        # Each variant gets its own seed derived from experiment seed
        variant_seed = exp_seed + variant_id * 10000
        variant_rng = np.random.default_rng(variant_seed)

        # Generate material
        material = generate_random_material(base_name, variant_id, variant_seed, variant_rng)
        material_name = material["name"]
        material_names.append(material_name)

        if materials_dict is not None:
            # Single-file mode: accumulate in dictionary
            materials_dict[material_name] = material
        else:
            # Multi-file mode: save each material separately
            output_file = output_dir / f"{material_name}.json"
            with open(output_file, 'w') as f:
                json.dump(material, f, indent=4)

    return material_names


def generate_parameter_combinations(
    powers: list,
    diameters: list,
    feeds: list,
    speeds: list
) -> list:
    """Generate all parameter combinations

    Returns:
        List of tuples (experiment_id, power, diameter, feed, speed)
    """
    combinations = []
    experiment_id = 0

    for power in powers:
        for diameter in diameters:
            for feed in feeds:
                for speed in speeds:
                    combinations.append((experiment_id, power, diameter, feed, speed))
                    experiment_id += 1

    return combinations


def main():
    parser = argparse.ArgumentParser(
        description='Generate random material property files for DED-LB experiments'
    )
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for material JSON files')
    parser.add_argument('--n-variants', type=int, default=3,
                       help='Number of material variants per parameter combination')
    parser.add_argument('--master-seed', type=int, default=42,
                       help='Master random seed for reproducibility')
    parser.add_argument('--full-design', action='store_true',
                       help='Use full factorial design (11×5×5×10=2750), otherwise test design (4×2×2×2=32)')
    parser.add_argument('--multi-file', action='store_true',
                       help='Generate individual JSON files instead of single file (not recommended for scratch filesystems)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define parameter ranges
    if args.full_design:
        powers = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
        diameters = [0.5, 0.825, 1.15, 1.475, 1.8]
        feeds = [2.0, 2.5, 3.0, 3.5, 4.0]
        speeds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        design_name = "full"
    else:
        powers = [600, 900, 1200, 1500]
        diameters = [0.5, 1.8]
        feeds = [2.0, 4.0]
        speeds = [2, 20]
        design_name = "test"

    # Generate all combinations
    combinations = generate_parameter_combinations(powers, diameters, feeds, speeds)
    total_experiments = len(combinations)
    total_materials = total_experiments * args.n_variants

    print(f"{'='*80}")
    print(f"Random Material Generation - {design_name.upper()} Design")
    print(f"{'='*80}")
    print(f"Parameter ranges:")
    print(f"  Powers: {len(powers)} values - {powers}")
    print(f"  Diameters: {len(diameters)} values - {diameters}")
    print(f"  Feeds: {len(feeds)} values - {feeds}")
    print(f"  Speeds: {len(speeds)} values - {speeds}")
    print(f"")
    print(f"Total parameter combinations: {total_experiments}")
    print(f"Material variants per combination: {args.n_variants}")
    print(f"Total materials to generate: {total_materials}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Master seed: {args.master_seed}")
    print(f"Output mode: {'Multiple files' if args.multi_file else 'Single file (default)'}")
    print(f"{'='*80}\n")

    # Generate materials for each combination
    all_material_names = []

    if not args.multi_file:
        # Single-file mode (DEFAULT): accumulate all materials in dict, write once at end
        all_materials_dict = {}

        for i, (exp_id, power, diameter, feed, speed) in enumerate(combinations):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Generating materials for experiment {i+1}/{total_experiments}...")

            material_names = generate_materials_for_experiment(
                experiment_id=exp_id,
                power=power,
                diameter=diameter,
                feed=feed,
                speed=speed,
                output_dir=output_dir,
                n_variants=args.n_variants,
                master_seed=args.master_seed,
                materials_dict=all_materials_dict
            )
            all_material_names.extend(material_names)

        # Write all materials to single file
        output_file = output_dir / "all_materials.json"
        print(f"\nWriting all materials to single file: {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_materials_dict, f, indent=2)

        print(f"Single file size: {output_file.stat().st_size / (1024*1024):.2f} MB")

    else:
        # Multi-file mode: write each material to separate file
        for i, (exp_id, power, diameter, feed, speed) in enumerate(combinations):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Generating materials for experiment {i+1}/{total_experiments}...")

            material_names = generate_materials_for_experiment(
                experiment_id=exp_id,
                power=power,
                diameter=diameter,
                feed=feed,
                speed=speed,
                output_dir=output_dir,
                n_variants=args.n_variants,
                master_seed=args.master_seed,
                materials_dict=None
            )
            all_material_names.extend(material_names)

    print(f"\n{'='*80}")
    print(f"Material Generation Complete")
    print(f"{'='*80}")
    print(f"Generated {len(all_material_names)} materials")
    print(f"Saved to: {output_dir.absolute()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:  # No arguments provided
        sys.argv.extend(['--output-dir', './configuration/materials', '--full-design'])
    main()