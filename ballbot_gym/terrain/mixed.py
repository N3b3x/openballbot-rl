"""Mixed/composite terrain generator for ballbot environment."""
import numpy as np
from typing import Optional, List, Dict, Any
from ballbot_gym.core.factories import create_terrain


def generate_mixed_terrain(
    n: int,
    components: List[Dict[str, Any]],
    blend_mode: str = "additive",
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate terrain by combining multiple terrain types.
    
    Creates complex terrain by blending multiple terrain generators,
    useful for realistic terrain simulation and advanced training scenarios.
    
    Args:
        n: Grid size (number of rows/columns, should be odd)
        components: List of component configs, each with:
            - "type": terrain type name
            - "weight": blending weight (0-1)
            - "config": terrain-specific config dict
        blend_mode: Blending mode ("additive", "max", or "weighted")
        seed: Random seed (passed to components if they use it)
        
    Returns:
        1D array of height values, shape (n*n,), normalized to [0, 1]
        
    Raises:
        AssertionError: If n is not odd or parameters are invalid
        ValueError: If blend_mode is invalid or components are malformed
    """
    assert n % 2 == 1, "n should be odd for heightfield symmetry"
    assert len(components) > 0, "components list cannot be empty"
    assert blend_mode in ["additive", "max", "weighted"], "blend_mode must be 'additive', 'max', or 'weighted'"
    
    # Validate and prepare components
    terrain_components = []
    weights = []
    
    for comp in components:
        if not isinstance(comp, dict):
            raise ValueError(f"Component must be a dict, got {type(comp)}")
        
        comp_type = comp.get("type")
        weight = comp.get("weight", 1.0)
        comp_config = comp.get("config", {})
        
        if comp_type is None:
            raise ValueError("Component must have 'type' key")
        
        # Create terrain generator
        terrain_config = {
            "type": comp_type,
            "config": comp_config
        }
        
        # Add seed to config if not present
        if "seed" not in terrain_config["config"] and seed is not None:
            terrain_config["config"]["seed"] = seed
        
        terrain_gen = create_terrain(terrain_config)
        terrain_components.append(terrain_gen)
        weights.append(weight)
    
    # Generate all component terrains
    component_terrains = []
    for terrain_gen in terrain_components:
        terrain_data = terrain_gen(n, seed=seed)
        component_terrains.append(terrain_data.reshape(n, n))
    
    # Blend terrains according to blend_mode
    if blend_mode == "additive":
        # Simple weighted sum
        terrain = np.zeros((n, n))
        total_weight = sum(weights)
        for comp_terrain, weight in zip(component_terrains, weights):
            terrain += comp_terrain * (weight / total_weight)
            
    elif blend_mode == "max":
        # Take maximum at each point
        terrain = np.zeros((n, n))
        for comp_terrain, weight in zip(component_terrains, weights):
            terrain = np.maximum(terrain, comp_terrain * weight)
            
    else:  # weighted
        # Weighted average
        terrain = np.zeros((n, n))
        total_weight = sum(weights)
        for comp_terrain, weight in zip(component_terrains, weights):
            terrain += comp_terrain * weight
        terrain = terrain / total_weight
    
    # Clip to [0, 1]
    # We do NOT re-normalize because components should control physical height
    terrain = np.clip(terrain, 0.0, 1.0)
    
    return terrain.flatten()

