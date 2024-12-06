from math import sqrt, isclose
import numpy as np
import sympy as sp

import sympy as sp

class Optimizer:
    def __init__(self, f):
        """
        Initialize the optimizer with the numeric function.

        Args:
            f (callable): A numeric function to optimize.
        """
        self.f = f
    def GoldenSearch(self, xL: float, xR: float, epsilon: float = 1e-10, max_iter: int = 100, verbose: bool = True) -> tuple:
        """
        Perform optimization using the Golden Section Search method.

        Args:
            xL (float): The left bound of the search interval.
            xR (float): The right bound of the search interval.
            epsilon (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.
            verbose (bool): Whether to print detailed logs.

        Returns:
            tuple: (xm, extreme_value, iterations)
        """
        if xL >= xR:
            raise ValueError("xL must be less than xR.")
        if epsilon <= 0:
            raise ValueError("Epsilon must be greater than 0.")

        f = self.f
        golden_ratio = (sqrt(5) - 1) / 2
        output = []

        for i in range(max_iter):
            diff = xR - xL
            x1 = xL + (1 - golden_ratio) * diff
            x2 = xR - (1 - golden_ratio) * diff

            fx1 = f(x1)
            fx2 = f(x2)

            if verbose:
                output.append(
                    f"Iteration {i+1}: Interval = [{xL:.6f}, {xR:.6f}], "
                    f"x1 = {x1:.6f}, x2 = {x2:.6f}, f(x1) = {fx1:.6f}, f(x2) = {fx2:.6f}"
                )

            # Check convergence
            if abs(xR - xL) < epsilon:
                xm = (xL + xR) / 2
                extreme_value = f(xm)
                if verbose:
                    output.append(f"\nConverged: xm = {xm:.6f}, f(xm) = {extreme_value:.6f}")
                    for line in output:
                        print(line)
                return xm, extreme_value, i + 1

            # Update interval
            if fx1 < fx2:
                xL = x1
            else:
                xR = x2

        # If max iterations reached
        xm = (xL + xR) / 2
        extreme_value = f(xm)
        if verbose:
            output.append(f"\nMax iterations reached: xm = {xm:.6f}, f(xm) = {extreme_value:.6f}")
            for line in output:
                print(line)
        return xm, extreme_value, max_iter

    def NewtonRaphson(self, x0: float, epsilon: float = 1e-6, max_iter: int = 100) -> tuple:
        """
        Perform optimization using the Newton-Raphson method.

        Args:
            x0 (float): Initial guess for the root.
            epsilon (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.

        Returns:
            tuple: A tuple containing:
                - root (float): The root of the function.
                - iterations (int): The number of iterations performed.
                - logs (list of str): Logs of each iteration showing the intermediate values.
        """
        x = sp.symbols("x")
        f_sympy = sp.Lambda(x, self.f(x))  # Convert the function to a sympy expression
        f_prime_sympy = sp.diff(f_sympy(x), x)  # Symbolic derivative
        f_prime = sp.lambdify(x, f_prime_sympy, "numpy")  # Convert to numeric

        logs = []
        current_x = x0

        for i in range(max_iter):
            fx = self.f(current_x)
            fpx = f_prime(current_x)

            # Check if derivative is near zero to avoid division by zero
            if abs(fpx) < 1e-12:
                logs.append(f"Iteration {i+1}: Derivative near zero. No solution found.")
                return None, i + 1, logs

            next_x = current_x - fx / fpx
            logs.append(
                f"Iteration {i+1}: x = {current_x:.6f}, f(x) = {fx:.6f}, f'(x) = {fpx:.6f}, next_x = {next_x:.6f}"
            )

            # Convergence check
            if abs(next_x - current_x) < epsilon:
                logs.append(f"Converged after {i+1} iterations: root = {next_x:.6f}, f(root) = {self.f(next_x):.6f}")
                return next_x, i + 1, logs

            current_x = next_x

        logs.append(f"Max iterations reached. Approximate root = {current_x:.6f}, f(root) = {self.f(current_x):.6f}")
        return current_x, max_iter, logs
    


if __name__ == '__main__':
    # Define the function to optimize
    def test_function(x):
        return (x - 5)**2 + 3

    # Create an Optimizer instance
    optimizer = Optimizer(f=test_function)

    # Perform Golden Section Search
    xm, extreme_value, iterations = optimizer.GoldenSearch(xL=4, xR=8, epsilon=1e-6, max_iter=100)

    # Print results
    print("\nResults:")
    print(f"xm = {xm}, f(xm) = {extreme_value}, iterations = {iterations}")

    # Define the numeric function
    def test_function(x):
        return x**3 - 6*x**2 + 11*x - 6  # Roots at x=1, x=2, x=3

    # Create an instance of the Optimizer
    optimizer = Optimizer(test_function)

    # Perform Newton-Raphson
    x0 = 3.5  # Initial guess
    root, iterations, logs = optimizer.NewtonRaphson(x0=x0, epsilon=1e-6, max_iter=100)

    # Print results
    print("\nResults:")
    print(f"Root = {root}, Iterations = {iterations}\n")

    # Print detailed logs
    print("Logs:")
    for log in logs:
        print(log)

