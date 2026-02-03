"""
my_functions.py - Example module for demonstrating imports

This module contains simple functions to demonstrate how to:
1. Create reusable Python code
2. Import functions from other files
3. Document your code with docstrings

Usage:
    from modules import my_functions
    result = my_functions.add_numbers(5, 3)
    
    # Or import specific functions
    from modules.my_functions import greet, calculate_area
"""

import math


def greet(name: str) -> str:
    """
    Return a greeting message.
    
    Parameters
    ----------
    name : str
        The name to greet
        
    Returns
    -------
    str
        A greeting message
        
    Examples
    --------
    >>> greet("World")
    'Hello, World!'
    """
    return f"Hello, {name}!"


def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers together.
    
    Parameters
    ----------
    a : float
        First number
    b : float
        Second number
        
    Returns
    -------
    float
        The sum of a and b
        
    Examples
    --------
    >>> add_numbers(5, 3)
    8
    >>> add_numbers(2.5, 3.5)
    6.0
    """
    return a + b


def subtract_numbers(a: float, b: float) -> float:
    """
    Subtract b from a.
    
    Parameters
    ----------
    a : float
        First number
    b : float
        Second number
        
    Returns
    -------
    float
        The result of a - b
    """
    return a - b


def multiply_numbers(a: float, b: float) -> float:
    """
    Multiply two numbers.
    
    Parameters
    ----------
    a : float
        First number
    b : float
        Second number
        
    Returns
    -------
    float
        The product of a and b
    """
    return a * b


def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle.
    
    Parameters
    ----------
    radius : float
        The radius of the circle
        
    Returns
    -------
    float
        The area of the circle (pi * r^2)
        
    Examples
    --------
    >>> calculate_area(1)
    3.141592653589793
    >>> round(calculate_area(5), 2)
    78.54
    """
    return math.pi * radius ** 2


def calculate_circumference(radius: float) -> float:
    """
    Calculate the circumference of a circle.
    
    Parameters
    ----------
    radius : float
        The radius of the circle
        
    Returns
    -------
    float
        The circumference of the circle (2 * pi * r)
    """
    return 2 * math.pi * radius


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """
    Convert temperature from Fahrenheit to Celsius.
    
    Parameters
    ----------
    fahrenheit : float
        Temperature in Fahrenheit
        
    Returns
    -------
    float
        Temperature in Celsius
        
    Examples
    --------
    >>> fahrenheit_to_celsius(32)
    0.0
    >>> fahrenheit_to_celsius(212)
    100.0
    """
    return (fahrenheit - 32) * 5 / 9


def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert temperature from Celsius to Fahrenheit.
    
    Parameters
    ----------
    celsius : float
        Temperature in Celsius
        
    Returns
    -------
    float
        Temperature in Fahrenheit
    """
    return celsius * 9 / 5 + 32


# This code only runs when the file is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    # Test the functions
    print("Testing my_functions.py")
    print("-" * 30)
    
    print(greet("Workshop"))
    print(f"5 + 3 = {add_numbers(5, 3)}")
    print(f"Area of circle with r=5: {calculate_area(5):.2f}")
    print(f"32°F = {fahrenheit_to_celsius(32)}°C")
