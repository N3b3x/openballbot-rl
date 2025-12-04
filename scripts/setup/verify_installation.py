"""Verify openballbot-rl installation."""
from pathlib import Path
import sys

def test_imports():
    """Test that all packages can be imported."""
    try:
        import ballbot_gym
        print("✓ ballbot_gym imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ballbot_gym: {e}")
        return False
    
    try:
        import ballbot_rl
        print("✓ ballbot_rl imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ballbot_rl: {e}")
        return False
    
    return True

def test_environment_registration():
    """Test that environment is registered with Gymnasium."""
    try:
        import gymnasium as gym
        import ballbot_gym
        
        env = gym.make("ballbot-v0.1")
        print("✓ Environment registered and created successfully")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False

def test_cli_commands():
    """Test that CLI commands are available."""
    from importlib import import_module
    
    try:
        # Test that entry points resolve
        from ballbot_rl.training.train import main as train_main
        from ballbot_rl.evaluation.evaluate import main as eval_main
        print("✓ CLI entry points resolve correctly")
        return True
    except Exception as e:
        print(f"✗ Failed to resolve CLI entry points: {e}")
        return False

if __name__ == "__main__":
    print("Verifying openballbot-rl installation...\n")
    
    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_environment_registration()
    all_passed &= test_cli_commands()
    
    if all_passed:
        print("\n✓ All checks passed!")
        sys.exit(0)
    else:
        print("\n✗ Some checks failed!")
        sys.exit(1)

