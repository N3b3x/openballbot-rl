"""Interactive utilities for training."""


def confirm(question):
    """
    Prompt user for yes/no confirmation.
    
    Args:
        question (str): Question to ask the user.
    
    Returns:
        bool: True if user answered 'y', False if 'n'.
    """
    inpt = ""
    while inpt != 'y' and inpt != 'n':
        inpt = input(question + " [y/N]: ").strip().lower()
    return inpt == 'y'

