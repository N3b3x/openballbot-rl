"""Learning rate schedules for training."""


def lr_schedule(progress_remaining):
    """
    Learning rate schedule for PPO training.
    
    Args:
        progress_remaining: Goes from 1 (beginning) to 0 (end)
    
    Returns:
        float: Learning rate for the current training progress
    """
    if progress_remaining > 0.7:
        return 1e-4
    elif progress_remaining < 0.7 and progress_remaining > 0.5:
        return 5e-5
    else:
        return 1e-5

