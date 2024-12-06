import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

class RootFinder:
    def __init__(self, f, df=None):
        """
        Initialize the root finder with the given numeric function.

        Args:
            f (callable): Numeric function for which the root is to be found.
            df (callable, optional): Derivative of the function f. Required for Newton-Raphson.
        """
        self.f = f
        self.df = df  # Derivative of f (required for Newton-Raphson)

    def newton_raphson(self, x0, tol=1e-6, max_iter=100):
        """
        Perform the Newton-Raphson method.

        Args:
            x0 (float): Initial guess for the root.
            tol (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.

        Returns:
            tuple: (root, history) where:
                - root: The estimated root of the function.
                - history: A list of x values at each iteration.

        Raises:
            ValueError: If the derivative is not provided.
        """
        if self.df is None:
            raise ValueError("Derivative function (df) is required for Newton-Raphson.")

        x = x0
        history = [x0]

        for i in range(max_iter):
            fx = self.f(x)
            dfx = self.df(x)

            if abs(dfx) < 1e-12:
                raise ValueError(f"Derivative near zero at iteration {i+1}. No solution found.")

            x_new = x - fx / dfx
            history.append(x_new)

            if abs(x_new - x) < tol:
                return x_new, history

            x = x_new

        return x, history

    def plot_newton_raphson(self, history, start=-3, end=3):
        """
        Visualize the Newton-Raphson iterations.

        Args:
            history (list): A list of x values from the Newton-Raphson method.
            start (float): Start of the x-axis range for the plot.
            end (float): End of the x-axis range for the plot.
        """
        x_vals = np.linspace(start, end, 400)
        y_vals = self.f(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
        plt.axhline(0, color='black', lw=1)

        for i, x in enumerate(history[:-1]):
            tangent_line = self.df(x) * (x_vals - x) + self.f(x)
            plt.plot(x_vals, tangent_line, '--', color='gray', alpha=0.6)
            plt.scatter(x, self.f(x), color='red', zorder=5)
            plt.text(x, self.f(x), f'Iter {i}', fontsize=9, color='red')

        plt.scatter(history[-1], self.f(history[-1]), color='green', zorder=5,
                    label=f'Root approximation: {history[-1]:.6f}')

        plt.title('Newton-Raphson Method Visualization')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.show()


    def bisection_visualize(self, a, b, tol=1e-6, max_iter=100):
        """
        Find the root of the function using the bisection method with visualization support.

        Args:
            a (float): Left endpoint of the interval.
            b (float): Right endpoint of the interval.
            tol (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.

        Returns:
            tuple: (root, history) where:
                - root: The estimated root of the function.
                - history: A list of tuples (a, b, c, f(a), f(b), f(c)) for each iteration.
        """
        f = self.f
        if f(a) * f(b) >= 0:
            raise ValueError("f(a) and f(b) must have opposite signs to apply the bisection method.")

        iter_count = 0
        history = []  # To store (a, b, c, f(a), f(b), f(c)) for each iteration

        while (b - a) / 2 > tol and iter_count < max_iter:
            c = (a + b) / 2
            history.append((a, b, c, f(a), f(b), f(c)))

            if abs(f(c)) < tol:  # If f(c) is close to 0, stop
                break
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c

            iter_count += 1

        self.root = (a + b) / 2
        return self.root, history

    def plot_bisection(self, history, start=None, end=None):
        """
        Visualize the bisection method progress.

        Args:
            history (list): A list of tuples (a, b, c, f(a), f(b), f(c)) from the bisection method.
            start (float, optional): Start of the x-axis range for the plot.
            end (float, optional): End of the x-axis range for the plot.
        """
        f = self.f
        if not history:
            raise ValueError("No history provided for visualization.")

        # Use the first and last interval for the x range if not specified
        if start is None:
            start = history[0][0]
        if end is None:
            end = history[0][1]

        x = np.linspace(start, end, 400)
        y = f(x)

        plt.figure(figsize=(12, 8))
        plt.plot(x, y, label='f(x)', color='blue', lw=2)
        plt.axhline(0, color='black', lw=1, linestyle='--')

        for i, (a, b, c, fa, fb, fc) in enumerate(history):
            color = plt.cm.viridis(i / len(history))  # Gradually change color
            plt.fill_between(x, 0, y, where=(x >= a) & (x <= b), color=color, alpha=0.2)
            plt.scatter([a, b], [fa, fb], color='red', label='Interval Endpoints' if i == 0 else None)
            plt.scatter(c, fc, color='green', s=50, zorder=5,
                        label='Midpoint (c)' if i == 0 else None)

        # Highlight the final root
        root_c, _, _, _, _, root_fc = history[-1]
        plt.scatter(self.root, root_fc, color='purple', s=100, zorder=6, label=f"Root Estimate: c={self.root:.6f}")

        plt.title("Bisection Method Visualization")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def secant_method(self, x0, x1, tol=1e-6, max_iter=100):
        """
        Perform the Secant Method to find the root of the function.

        Args:
            x0 (float): Initial guess for the root.
            x1 (float): Second initial guess for the root.
            tol (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.

        Returns:
            tuple: (root, history) where:
                - root: The estimated root of the function.
                - history: A list of tuples (x, f(x)) at each iteration.
        """
        history = [(x0, self.f(x0)), (x1, self.f(x1))]  # Store history of (x, f(x))

        for _ in range(max_iter):
            f_x0, f_x1 = self.f(x0), self.f(x1)

            if abs(f_x1) < tol:
                return x1, history  # Convergence achieved

            # Compute the next approximation
            x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
            history.append((x2, self.f(x2)))

            # Update x0 and x1 for the next iteration
            x0, x1 = x1, x2

        return x1, history

    def plot_secant_method(self, history, start=-3, end=3):
        """
        Visualize the Secant Method iterations.

        Args:
            history (list): A list of tuples (x, f(x)) from the Secant Method.
            start (float): Start of the x-axis range for the plot.
            end (float): End of the x-axis range for the plot.
        """
        x_vals = np.linspace(start, end, 400)
        y_vals = self.f(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
        plt.axhline(0, color='black', lw=1)

        for i in range(1, len(history)):
            x0, fx0 = history[i-1]
            x1, fx1 = history[i]
            plt.plot([x0, x1], [fx0, fx1], 'g--', label=f'Secant line {i}' if i == 1 else "")
            plt.scatter(x0, fx0, color='red')
            plt.scatter(x1, fx1, color='green')

        plt.title('Secant Method for f(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def inverse_interpolation(self, x0, x1, x2, tol=1e-6, max_iter=100):
        """
        Perform the Inverse Interpolation Method to find the root of the function.

        Args:
            x0 (float): First initial guess for the root.
            x1 (float): Second initial guess for the root.
            x2 (float): Third initial guess for the root.
            tol (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.

        Returns:
            tuple: (root, history) where:
                - root: The estimated root of the function.
                - history: A list of tuples (x, f(x)) at each iteration.
        """
        history = [(x0, self.f(x0)), (x1, self.f(x1)), (x2, self.f(x2))]  # Store history of (x, f(x))

        for _ in range(max_iter):
            f_x0, f_x1, f_x2 = self.f(x0), self.f(x1), self.f(x2)

            # Compute the next approximation using inverse interpolation
            x_new = (
                x0 * f_x1 * f_x2 / ((f_x0 - f_x1) * (f_x0 - f_x2)) +
                x1 * f_x0 * f_x2 / ((f_x1 - f_x0) * (f_x1 - f_x2)) +
                x2 * f_x0 * f_x1 / ((f_x2 - f_x0) * (f_x2 - f_x1))
            )
            history.append((x_new, self.f(x_new)))

            # Check for convergence
            if abs(self.f(x_new)) < tol:
                return x_new, history

            # Update x0, x1, x2 for the next iteration
            x0, x1, x2 = x1, x2, x_new

        return x_new, history

    def plot_inverse_interpolation(self, history, start=0, end=4):
        """
        Visualize the Inverse Interpolation Method iterations.

        Args:
            history (list): A list of tuples (x, f(x)) from the Inverse Interpolation Method.
            start (float): Start of the x-axis range for the plot.
            end (float): End of the x-axis range for the plot.
        """
        x_vals = np.linspace(start, end, 400)
        y_vals = self.f(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label='$f(x)$', color='blue')
        plt.axhline(0, color='black', lw=1)

        for i in range(1, len(history)):
            x_prev, f_prev = history[i-1]
            x_curr, f_curr = history[i]
            plt.plot([x_prev, x_curr], [f_prev, f_curr], 'g--', label=f'Interpolation {i}' if i == 1 else "")
            plt.scatter(x_curr, f_curr, color='red')

        plt.title('Inverse Interpolation Method for Root Finding')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
if __name__ == "__main__":
    # Define the function
    def test_function(x):
        return x**3 - 4*x**2 + 6*x - 24  # Root exists between 3 and 5

    # Create an instance of RootFinder
    finder = RootFinder(test_function)

    # Find the root using bisection
    root, history = finder.bisection_visualize(a=3, b=5, tol=1e-6, max_iter=50)

    # Print the result
    print(f"Estimated root: {root:.6f}")

    # Visualize the bisection method
    finder.plot_bisection(history, start=2, end=6)
    import numpy as np

    # 복잡한 함수 정의
    def test_function(x):
        """
        Test function: f(x) = e^x - 3sin(x) + ln(x + 2) - 5
        """
        return np.exp(x) - 3 * np.sin(x) + np.log(x + 2) - 5

    # Create an instance of RootFinder
    finder = RootFinder(test_function)

    # Define the interval [a, b] where f(a) * f(b) < 0
    a = 0  # Left endpoint
    b = 2  # Right endpoint

    # Perform bisection method
    root, history = finder.bisection_visualize(a, b, tol=1e-6, max_iter=100)

    # Print the estimated root
    print(f"Estimated root: {root:.6f}")

    # Visualize the bisection process
    finder.plot_bisection(history, start=0, end=2)
    
    def test_function(x):
        return np.exp(-x / 2) * (4 - x) - 2

    def test_function_derivative(x):
        return -0.5 * np.exp(-x / 2) * (4 - x) - np.exp(-x / 2)
    # Create an instance of RootFinder
    finder = RootFinder(test_function, test_function_derivative)

    # Initial guess for Newton-Raphson
    x0 = 1.0

    # Solve using Newton-Raphson
    root, history = finder.newton_raphson(x0, tol=1e-6, max_iter=100)

    # Print the result
    print(f"Estimated root: {root:.6f}")

    # Visualize the iterations
    finder.plot_newton_raphson(history, start=0, end=3)
    def test_function(x):
        return np.exp(-x / 2) * (4 - x) - 2
    # Create an instance of RootFinder
    finder = RootFinder(test_function)

    # Initial guesses for the Secant Method
    x0 = 0.0
    x1 = 1.0

    # Solve using Secant Method
    root, history = finder.secant_method(x0, x1, tol=1e-6, max_iter=100)

    # Print the result
    print(f"Estimated root: {root:.6f}")

    # Visualize the iterations
    finder.plot_secant_method(history, start=-1, end=3)
    def test_function(x):
        return np.exp(-x / 2) * (4 - x) - 2
    # Create an instance of RootFinder
    finder = RootFinder(test_function)

    # Initial guesses for the Inverse Interpolation Method
    x0 = 0.5
    x1 = 1.0
    x2 = 2.0

    # Solve using Inverse Interpolation Method
    root, history = finder.inverse_interpolation(x0, x1, x2, tol=1e-6, max_iter=100)

    # Print the result
    print(f"Estimated root: {root:.6f}")

    # Visualize the iterations
    finder.plot_inverse_interpolation(history, start=0, end=4)
