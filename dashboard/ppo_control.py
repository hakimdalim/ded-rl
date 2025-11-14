import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


ADAPTABLE_PARAMS = {
    'laser_power': (
        600,
        1600
    ),
    'scan_speed': (
        2.0 / 1000,
        20.0 / 1000
    ),
    'powder_feed_rate': (
        2.0  * (1 / 1000) * (1 / 60),
        4.0  * (1 / 1000) * (1 / 60),
    ),
}

OBS_KEYS = {
    'step_dict': {
        'melt_pool.width': (0.0, np.inf),    # Cannot be negative
        'melt_pool.length': (0.0, np.inf),   # Cannot be negative
        'melt_pool.depth': (0.0, np.inf),    # Cannot be negative
        'clad.width': (0.0, np.inf),         # Cannot be negative
        'clad.height': (0.0, np.inf),        # Cannot be negative
        'clad.wetting_angle': (0.0, 2 * np.pi),  # Angle in radians
        'profile.width': (0.0, np.inf),      # Cannot be negative
        'profile.max_profile_height': (0.0, np.inf),  # Cannot be negative
        #'profile.height_at_x': (0.0, np.inf),  # Cannot be negative
        'temperature.voxel.z.-0': (0.0, np.inf),   # Temperature in Kelvin
        'temperature.voxel.z.-5': (0.0, np.inf),   # Temperature in Kelvin
        'temperature.voxel.z.-10': (0.0, np.inf),  # Temperature in Kelvin
        'temperature.voxel.z.-15': (0.0, np.inf),  # Temperature in Kelvin
        'temperature.voxel.z.-20': (0.0, np.inf),  # Temperature in Kelvin
    },
    'params': {
        'laser_power': (600, 1600),
        'scan_speed': (2.0 / 1000, 20.0 / 1000),
        'powder_feed_rate': (2.0 * (1 / 1000) * (1 / 60), 4.0 * (1 / 1000) * (1 / 60)),
    }
}



class PPOController:
    """Controller class to interface a trained PPO model with the DED simulation dashboard"""

    def __init__(self, model_path=None):
        """Initialize the PPO controller with optional model path"""
        self.model = None
        self.model_path = model_path
        self.active = False
        self.obs_keys = None
        self.param_keys = None
        self.last_action = None
        self.last_observation = None
        self.param_ranges = {
            'laser_power': (600.0, 1600.0),
            'scan_speed': (2.0 / 1000, 20.0 / 1000),  # m/s
            'powder_feed_rate': (2.0 / (60 * 1000), 4.0 / (60 * 1000))  # kg/s
        }

        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load a trained PPO model from file"""
        try:
            # Set device to CPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load the model
            self.model = PPO.load(model_path, device=device)
            self.model_path = model_path

            # Set to evaluation mode
            self.model.policy.set_training_mode(False)

            # Extract observation and action space information
            self._extract_model_info()

            print(f"Successfully loaded PPO model from {model_path}")

            # After loading the model
            if self.model:
                # Print observation space details for debugging
                if hasattr(self.model, 'observation_space'):
                    print("Model observation space:", self.model.observation_space)
                if hasattr(self.model.policy, 'observation_space'):
                    print("Policy observation space:", self.model.policy.observation_space)
            return True
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            self.model = None
            return False

    def _extract_model_info(self):
        """Extract information about observation and action spaces from the model"""
        if not self.model:
            return

        # Extract observation keys if available
        try:
            # For Dict observation space
            if hasattr(self.model, 'observation_space') and isinstance(self.model.observation_space, spaces.Dict):
                self.obs_keys = list(self.model.observation_space.spaces.keys())
            elif hasattr(self.model.policy, 'observation_space') and isinstance(self.model.policy.observation_space,
                                                                                spaces.Dict):
                self.obs_keys = list(self.model.policy.observation_space.spaces.keys())

            # For MultiInput policy, extract feature extractors
            if hasattr(self.model.policy, 'features_extractor') and isinstance(self.model.policy.features_extractor,
                                                                               BaseFeaturesExtractor):
                if hasattr(self.model.policy.features_extractor, 'extractors'):
                    self.obs_keys = list(self.model.policy.features_extractor.extractors.keys())
        except Exception as e:
            print(f"Warning: Could not extract observation keys: {e}")

        # Extract parameter keys from action space
        try:
            # For Dict action space
            if hasattr(self.model, 'action_space') and isinstance(self.model.action_space, spaces.Dict):
                self.param_keys = list(self.model.action_space.spaces.keys())
            elif hasattr(self.model.policy, 'action_space') and isinstance(self.model.policy.action_space, spaces.Dict):
                self.param_keys = list(self.model.policy.action_space.spaces.keys())
            else:
                # For Box action space (normalized), use default parameter keys
                self.param_keys = list(self.param_ranges.keys())
        except Exception as e:
            print(f"Warning: Could not extract action keys: {e}")
            # Default to using standard parameter keys
            self.param_keys = list(self.param_ranges.keys())

    def activate(self, active=True):
        """Activate or deactivate the PPO controller"""
        if active and self.model is None:
            print("Cannot activate: No model loaded")
            return False

        self.active = active and self.model is not None
        return self.active

    def is_active(self):
        """Check if the controller is active"""
        return self.active and self.model is not None

    def _prepare_observation(self, simulation_metrics, current_params):
        """
        Prepare the observation dictionary for the PPO model

        Args:
            simulation_metrics (dict): Current simulation metrics
            current_params (dict): Current process parameters

        Returns:
            dict: Observation dictionary in the format expected by the model
        """
        observation = {}

        # Add metrics from step_dict
        if simulation_metrics:
            for key, value in simulation_metrics.items():
                if key not in OBS_KEYS['step_dict']:
                    continue
                obs_key = f"step_dict.{key}".replace('.', '#')
                #obs_key = f"step_dict.{key}"
                # Handle potential missing keys or NaN values
                if value is not None and not np.isnan(value):
                    observation[obs_key] = np.array([float(value)], dtype=np.float32)
                else:
                    observation[obs_key] = np.array([0.0], dtype=np.float32)

        # Add current parameters
        if current_params:
            for key, value in current_params.items():
                if key in self.param_ranges:
                    obs_key = f"params.{key}".replace('.', '#')
                    observation[obs_key] = np.array([float(value)], dtype=np.float32)

        # Store observation for diagnostics
        self.last_observation = observation
        return observation

    def predict_parameters(self, simulation_metrics, current_params):
        """
        Predict optimal parameters based on current simulation state

        Args:
            simulation_metrics (dict): Current simulation metrics
            current_params (dict): Current process parameters

        Returns:
            dict: Updated parameters dict with PPO model's recommendations
        """
        if not self.is_active() or self.model is None:
            return current_params

        try:
            # Prepare observation
            observation = self._prepare_observation(simulation_metrics, current_params)

            # Get model prediction (deterministic for production use)
            action, _ = self.model.predict(observation, deterministic=True)

            # Store last action for diagnostics
            self.last_action = action

            # Convert normalized actions to actual parameter values if using Box action space
            if isinstance(action, np.ndarray) and len(action) == len(self.param_keys):
                # For Box action space, denormalize values
                updated_params = current_params.copy()
                for i, param_name in enumerate(self.param_keys):
                    if param_name in self.param_ranges:
                        low, high = self.param_ranges[param_name]
                        # Convert from [-1, 1] to [low, high]
                        norm_action = np.clip(action[i], -1.0, 1.0)
                        orig_action = 0.5 * (norm_action + 1.0) * (high - low) + low
                        updated_params[param_name] = float(orig_action)
            else:
                # For Dict action space, extract values directly
                updated_params = current_params.copy()
                for param_name, action_value in action.items():
                    if param_name in updated_params:
                        updated_params[param_name] = float(action_value[0])

            # Apply safety limits to ensure parameters are within valid ranges
            for param_name, (min_val, max_val) in self.param_ranges.items():
                if param_name in updated_params:
                    updated_params[param_name] = np.clip(updated_params[param_name], min_val, max_val)

            return updated_params

        except Exception as e:
            print(f"Error during parameter prediction: {e}")
            import traceback
            traceback.print_exc()
            # Return unchanged parameters in case of error
            return current_params

    def get_info(self):
        """Get diagnostic information about the controller"""
        info = {
            "active": self.active,
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "observation_keys": self.obs_keys,
            "parameter_keys": self.param_keys,
            "last_action": self.last_action,
        }
        return info