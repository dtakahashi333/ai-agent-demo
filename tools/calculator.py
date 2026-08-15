# tools/calculator.py


def calculator(a: float, b: float, operation: str):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    raise ValueError(f"Unknown operation: {operation}")
