"""Implementation of MathAPI."""


class MathAPI:
    """A simple math API for BFCL evaluation."""

    def __init__(self):
        pass

    def _load_scenario(self, config):
        # MathAPI is stateless, so no scenarios to load
        pass

    def add(self, a, b):
        """Add two numbers"""
        return {"result": a + b}

    def subtract(self, a, b):
        """Subtract b from a"""
        return {"result": a - b}

    def multiply(self, a, b):
        """Multiply two numbers"""
        return {"result": a * b}

    def divide(self, a, b):
        """Divide a by b"""
        if b == 0:
            return {"error": "Cannot divide by zero"}
        return {"result": a / b}

    def square_root(self, a):
        """Calculate the square root of a number"""
        if a < 0:
            return {"error": "Cannot calculate square root of negative number"}
        return {"result": a**0.5}

    def power(self, base, exponent):
        """Calculate base raised to the power of exponent"""
        return {"result": base**exponent}
