"""Component registry for extensible architecture."""
from typing import Dict, Type, Callable, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ballbot_gym.rewards.base import BaseReward


class ComponentRegistry:
    """
    Central registry for all extensible components.
    
    This registry allows components (rewards, terrains, policies, etc.) to be
    registered and retrieved by name, enabling a plugin-like architecture where
    new components can be added without modifying core code.
    
    Example:
        ```python
        # Register a component
        ComponentRegistry.register_reward("my_reward", MyRewardClass)
        
        # Retrieve and instantiate
        reward = ComponentRegistry.get_reward("my_reward", **config)
        
        # List available components
        available = ComponentRegistry.list_rewards()
        ```
    """
    
    # Class-level dictionaries to store registered components
    _rewards: Dict[str, Type['BaseReward']] = {}
    _terrains: Dict[str, Callable] = {}
    _policies: Dict[str, Type] = {}
    _sensors: Dict[str, Type] = {}
    
    @classmethod
    def register_reward(cls, name: str, reward_class: Type['BaseReward']) -> None:
        """
        Register a reward function class.
        
        Args:
            name: Unique name identifier for the reward function
            reward_class: Class that inherits from BaseReward
            
        Raises:
            ValueError: If name is already registered or class doesn't inherit from BaseReward
        """
        # Import locally to avoid circular import
        from ballbot_gym.rewards.base import BaseReward
        
        if name in cls._rewards:
            raise ValueError(
                f"Reward '{name}' is already registered. "
                f"Available rewards: {list(cls._rewards.keys())}"
            )
        if not issubclass(reward_class, BaseReward):
            raise ValueError(
                f"Reward class must inherit from BaseReward, got {reward_class}"
            )
        cls._rewards[name] = reward_class
    
    @classmethod
    def get_reward(cls, name: str, **kwargs) -> 'BaseReward':
        """
        Get and instantiate a reward function by name.
        
        Args:
            name: Name of the registered reward function
            **kwargs: Arguments to pass to reward function constructor
            
        Returns:
            Instantiated reward function object
            
        Raises:
            ValueError: If reward name is not registered
        """
        if name not in cls._rewards:
            available = list(cls._rewards.keys())
            raise ValueError(
                f"Unknown reward: '{name}'. Available rewards: {available}"
            )
        return cls._rewards[name](**kwargs)
    
    @classmethod
    def list_rewards(cls) -> List[str]:
        """List all registered reward function names."""
        return list(cls._rewards.keys())
    
    @classmethod
    def register_terrain(cls, name: str, terrain_fn: Callable) -> None:
        """
        Register a terrain generator function.
        
        Args:
            name: Unique name identifier for the terrain generator
            terrain_fn: Callable that generates terrain (takes n: int, **kwargs) -> np.ndarray
            
        Raises:
            ValueError: If name is already registered
        """
        if name in cls._terrains:
            raise ValueError(
                f"Terrain '{name}' is already registered. "
                f"Available terrains: {list(cls._terrains.keys())}"
            )
        if not callable(terrain_fn):
            raise ValueError(f"Terrain must be callable, got {type(terrain_fn)}")
        cls._terrains[name] = terrain_fn
    
    @classmethod
    def get_terrain(cls, name: str) -> Callable:
        """
        Get a terrain generator function by name.
        
        Args:
            name: Name of the registered terrain generator
            
        Returns:
            Terrain generator function
            
        Raises:
            ValueError: If terrain name is not registered
        """
        if name not in cls._terrains:
            available = list(cls._terrains.keys())
            raise ValueError(
                f"Unknown terrain: '{name}'. Available terrains: {available}"
            )
        return cls._terrains[name]
    
    @classmethod
    def list_terrains(cls) -> List[str]:
        """List all registered terrain generator names."""
        return list(cls._terrains.keys())
    
    @classmethod
    def register_policy(cls, name: str, policy_class: Type) -> None:
        """
        Register a policy class.
        
        Args:
            name: Unique name identifier for the policy
            policy_class: Policy class (typically a feature extractor or policy network)
            
        Raises:
            ValueError: If name is already registered
        """
        if name in cls._policies:
            raise ValueError(
                f"Policy '{name}' is already registered. "
                f"Available policies: {list(cls._policies.keys())}"
            )
        cls._policies[name] = policy_class
    
    @classmethod
    def get_policy(cls, name: str) -> Type:
        """
        Get a policy class by name.
        
        Args:
            name: Name of the registered policy
            
        Returns:
            Policy class
            
        Raises:
            ValueError: If policy name is not registered
        """
        if name not in cls._policies:
            available = list(cls._policies.keys())
            raise ValueError(
                f"Unknown policy: '{name}'. Available policies: {available}"
            )
        return cls._policies[name]
    
    @classmethod
    def list_policies(cls) -> List[str]:
        """List all registered policy names."""
        return list(cls._policies.keys())
    
    @classmethod
    def register_sensor(cls, name: str, sensor_class: Type) -> None:
        """
        Register a sensor class.
        
        Args:
            name: Unique name identifier for the sensor
            sensor_class: Sensor class
            
        Raises:
            ValueError: If name is already registered
        """
        if name in cls._sensors:
            raise ValueError(
                f"Sensor '{name}' is already registered. "
                f"Available sensors: {list(cls._sensors.keys())}"
            )
        cls._sensors[name] = sensor_class
    
    @classmethod
    def get_sensor(cls, name: str) -> Type:
        """
        Get a sensor class by name.
        
        Args:
            name: Name of the registered sensor
            
        Returns:
            Sensor class
            
        Raises:
            ValueError: If sensor name is not registered
        """
        if name not in cls._sensors:
            available = list(cls._sensors.keys())
            raise ValueError(
                f"Unknown sensor: '{name}'. Available sensors: {available}"
            )
        return cls._sensors[name]
    
    @classmethod
    def list_sensors(cls) -> List[str]:
        """List all registered sensor names."""
        return list(cls._sensors.keys())
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (mainly for testing)."""
        cls._rewards.clear()
        cls._terrains.clear()
        cls._policies.clear()
        cls._sensors.clear()

