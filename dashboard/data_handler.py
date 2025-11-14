import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional


class DataHandler:
    """Handler for loading and processing simulation data"""

    def __init__(self):
        self.experiment_dir = None
        self.build_mesh_dir = None
        self.thermal_plots_dir = None
        self.voxel_temps_dir = None
        self.simulation_data = None

        # Cache for loaded data
        self.mesh_cache = {}
        self.thermal_cache = {}
        self.temperature_cache = {}

    def set_experiment_directory(self, exp_dir):
        """Set the directory containing simulation results"""
        self.experiment_dir = Path(exp_dir)

        # Set subdirectories
        self.build_mesh_dir = self.experiment_dir / "build_mesh"
        self.thermal_plots_dir = self.experiment_dir / "thermal_plots"
        self.voxel_temps_dir = self.experiment_dir / "voxel_temps"

        # Try to load simulation data
        try:
            self.simulation_data = pd.read_csv(self.experiment_dir / "simulation_data.csv")
            print(f"Loaded simulation data from {self.experiment_dir}")
            return True
        except Exception as e:
            print(f"Error loading simulation data: {e}")
            return False

    def get_available_steps(self):
        """Get list of available simulation steps"""
        if self.simulation_data is not None:
            return self.simulation_data['step'].unique().tolist()
        return []

    def get_step_data(self, step):
        """Get data for a specific simulation step"""
        if self.simulation_data is not None:
            step_data = self.simulation_data[self.simulation_data['step'] == step]
            if not step_data.empty:
                return step_data.iloc[0].to_dict()
        return {}

    def get_build_mesh(self, step):
        """Get build mesh for a specific step"""
        if self.build_mesh_dir is None:
            return None

        # Check cache first
        if step in self.mesh_cache:
            return self.mesh_cache[step]

        # Find matching file
        mesh_file = self.build_mesh_dir / f"build_state_step{step:04d}.stl"

        if mesh_file.exists():
            try:
                import trimesh
                mesh = trimesh.load(mesh_file)

                # Cache the result (only store key data to save memory)
                self.mesh_cache[step] = {
                    'vertices': np.array(mesh.vertices),
                    'faces': np.array(mesh.faces),
                    'normals': np.array(mesh.face_normals)
                }

                return self.mesh_cache[step]
            except Exception as e:
                print(f"Error loading mesh for step {step}: {e}")
                return None

        return None

    def get_thermal_data(self, step):
        """Get thermal data slices for a specific step"""
        if self.experiment_dir is None:
            return None

        # Check cache first
        if step in self.thermal_cache:
            return self.thermal_cache[step]

        try:
            # Load thermal slices
            temps_dir = self.experiment_dir / "temperatures"

            xy_slice = np.load(temps_dir / f"xy_slice_step{step:04d}.npy")
            xz_slice = np.load(temps_dir / f"xz_slice_step{step:04d}.npy")
            yz_slice = np.load(temps_dir / f"yz_slice_step{step:04d}.npy")

            # Cache the result
            self.thermal_cache[step] = {
                'xy': xy_slice,
                'xz': xz_slice,
                'yz': yz_slice
            }

            return self.thermal_cache[step]
        except Exception as e:
            print(f"Error loading thermal data for step {step}: {e}")
            return None

    def get_temperature_volume(self, step):
        """Get full temperature volume for a specific step"""
        if self.voxel_temps_dir is None:
            return None

        # Check cache first
        if step in self.temperature_cache:
            return self.temperature_cache[step]

        # Find matching file
        temp_file = self.voxel_temps_dir / f"voxel_temps_step{step:04d}.npy"

        if temp_file.exists():
            try:
                temp_data = np.load(temp_file)

                # Cache the result (only if not too large)
                if temp_data.size < 1e7:  # Only cache if less than ~80MB
                    self.temperature_cache[step] = temp_data

                return temp_data
            except Exception as e:
                print(f"Error loading temperature volume for step {step}: {e}")
                return None

        return None

    def get_thermal_plots(self, step):
        """Get thermal plot images for a specific step"""
        if self.thermal_plots_dir is None:
            return None

        # Find matching files
        view_types = ['top', 'front', 'side']
        result = {}

        for view in view_types:
            image_file = self.thermal_plots_dir / f"thermal{step:04d}_{view}_view.png"
            if image_file.exists():
                result[view] = str(image_file)

        return result if result else None

    def get_simulation_params(self):
        """Get overall simulation parameters"""
        params_file = self.experiment_dir / "simulation_params.csv"

        if params_file.exists():
            try:
                params = pd.read_csv(params_file, index_col=0, header=None).squeeze("columns").to_dict()
                return params
            except Exception as e:
                print(f"Error loading simulation parameters: {e}")
                return {}

        return {}

    def get_melt_pool_history(self):
        """Get melt pool dimension history"""
        if self.simulation_data is not None:
            melt_pool_data = {
                'step': self.simulation_data['step'].tolist(),
                'width': self.simulation_data['melt_pool.width'].tolist(),
                'length': self.simulation_data['melt_pool.length'].tolist(),
                'depth': self.simulation_data['melt_pool.depth'].tolist(),
                'max_temp': self.simulation_data['temperature.max_temp'].tolist()
            }
            return melt_pool_data
        return {}

    def get_clad_history(self):
        """Get clad dimension history"""
        if self.simulation_data is not None:
            clad_data = {
                'step': self.simulation_data['step'].tolist(),
                'width': self.simulation_data['clad.width'].tolist(),
                'height': self.simulation_data['clad.height'].tolist(),
                'wetting_angle': self.simulation_data['clad.wetting_angle'].tolist()
            }
            return clad_data
        return {}

    def clear_cache(self):
        """Clear all cached data"""
        self.mesh_cache.clear()
        self.thermal_cache.clear()
        self.temperature_cache.clear()
        print("Cache cleared")

    def extract_position_data(self):
        """Extract position data for path visualization"""
        if self.simulation_data is not None:
            position_data = {
                'x': self.simulation_data['position.x'].tolist(),
                'y': self.simulation_data['position.y'].tolist(),
                'z': self.simulation_data['position.z'].tolist(),
                'layer': self.simulation_data['build.layer'].tolist(),
                'track': self.simulation_data['build.track'].tolist()
            }
            return position_data
        return {}