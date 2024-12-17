import numpy as np

class Linear_Equation:
    """
    A class for solving systems of linear equations using different numerical methods.
    """

    def __init__(self, augmented_matrix: np.array):
        """
        Initialize the Linear_Equation instance with the augmented matrix.

        Args:
            augmented_matrix (np.array): A 2D NumPy array representing the augmented matrix
                                         [A|b] of size (n, n+1).
        """
        if len(augmented_matrix.shape) != 2 or augmented_matrix.shape[1] != augmented_matrix.shape[0] + 1:
            raise ValueError("The augmented matrix must have n rows and n+1 columns.")
        self.augmented_matrix = augmented_matrix

    def Gaussian_Elimination(self) -> tuple:
        """
        Solve a system of linear equations using Gaussian Elimination.

        Returns:
            tuple:
                - logs (list of str): Logs of each step during forward elimination and back substitution.
                - x (np.array): Solution vector to the linear system (length n).
        """
        logs = []
        augmented_matrix = self.augmented_matrix.astype(np.float64)
        n = augmented_matrix.shape[0]

        # Forward elimination
        for i in range(n):
            max_row = np.argmax(abs(augmented_matrix[i:, i])) + i
            if augmented_matrix[max_row, i] == 0:
                raise ValueError("Matrix is singular and cannot be solved.")
            augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]
            logs.append(f"Step {i+1}: Pivoting row {i} with row {max_row}:\n{augmented_matrix}")

            for j in range(i + 1, n):
                factor = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
                logs.append(f"Step {i+1}.{j}: Eliminating row {j}, factor = {factor}:\n{augmented_matrix}")

        # Back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i+1:], x[i+1:])) / augmented_matrix[i, i]
            logs.append(f"Back substitution: x[{i}] = {x[i]}")

        return logs, x

    def Gauss_Jordan(self) -> tuple:
        """
        Solve a system of linear equations using the Gauss-Jordan elimination method.

        Returns:
            tuple:
                - x (np.array): Solution vector to the linear system (length n).
                - logs (list of str): Logs of each step during the process.
        """
        n = self.augmented_matrix.shape[0]
        augmented_matrix = self.augmented_matrix.astype(np.float64)
        logs = []

        for j in range(n):
            if augmented_matrix[j, j] == 0:
                raise ZeroDivisionError("Pivot element is zero. The system cannot be solved.")
            augmented_matrix[j] /= augmented_matrix[j, j]
            logs.append(f"Step {j+1}: Normalize row {j}:\n{augmented_matrix}")

            for i in range(n):
                if i != j:
                    factor = augmented_matrix[i, j]
                    augmented_matrix[i] -= factor * augmented_matrix[j]
                    logs.append(f"Step {j+1}.{i}: Eliminating row {i}, factor = {factor}:\n{augmented_matrix}")

        x = augmented_matrix[:, -1]
        return x, logs

    def Gauss_Seidel(self, epsilon: float, max_iter: int) -> tuple:
        """
        Solve a system of linear equations using the Gauss-Seidel iteration method.

        Args:
            epsilon (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.

        Returns:
            tuple:
                - x (np.array): Solution vector of length n.
                - logs (list of str): Logs of each iteration showing the intermediate values of x.
        """
        A = self.augmented_matrix[:, :-1]
        b = self.augmented_matrix[:, -1]
        n = len(b)
        x = np.zeros(n)  # Initial guess
        logs = []  # To store logs for each iteration

        for k in range(max_iter):
            x_new = np.copy(x)
            iteration_log = [f"Iteration {k + 1}:"]

            for i in range(n):
                sum_ = sum(A[i, j] * x_new[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - sum_) / A[i, i]
                iteration_log.append(f"x[{i}] = {x_new[i]:.6f}")

            logs.append("\n".join(iteration_log))

            # Convergence check
            if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
                logs.append(f"Converged after {k + 1} iterations.")
                return x_new, k + 1, logs
            x = x_new

        logs.append(f"Did not converge after {max_iter} iterations.")
        return x, max_iter, logs

class TridiagonalMatrix:
    """
    A class for solving a tridiagonal system of linear equations using the Thomas Algorithm (TDMA).
    """

    def __init__(self, a: np.array, b: np.array, c: np.array, r: np.array):
        """
        Initialize the TridiagonalMatrix instance.

        Args:
            a (np.array): Subdiagonal elements of the tridiagonal matrix (length n-1).
            b (np.array): Main diagonal elements of the tridiagonal matrix (length n).
            c (np.array): Superdiagonal elements of the tridiagonal matrix (length n-1).
            r (np.array): Right-hand side vector (length n).
        """
        if len(a) != len(b) - 1 or len(c) != len(b) - 1 or len(r) != len(b):
            raise ValueError("Invalid input sizes: Ensure a, b, c, and r have compatible lengths.")
        self.a = a.astype(np.float64)
        self.b = b.astype(np.float64)
        self.c = c.astype(np.float64)
        self.r = r.astype(np.float64)

    def TDMA(a, b, c, r):
        """
        TDMA (Thomas Algorithm) for solving tridiagonal systems.
        
        Parameters:
            a (np.array): lower diagonal (length n-1)
            b (np.array): main diagonal (length n)
            c (np.array): upper diagonal (length n-1)
            r (np.array): right-hand side vector (length n)
        
        Returns:
            logs: List of intermediate steps for debugging.
            x: Solution vector (length n).
        """
        n = len(b)
        x = np.zeros(n)
        logs = []

        # Forward elimination
        for i in range(1, n):
            factor = a[i-1] / b[i-1]
            b[i] -= factor * c[i-1]
            r[i] -= factor * r[i-1]
            logs.append(f"Step {i}: Forward Elimination:\n"
                        f"a = {a}\n"
                        f"b = {b}\n"
                        f"c = {c}\n"
                        f"r = {r}\n")

        # Back substitution
        x[-1] = r[-1] / b[-1]
        logs.append(f"Back Substitution: x[{n-1}] = {x[-1]}")

        for i in range(n-2, -1, -1):
            x[i] = (r[i] - c[i] * x[i + 1]) / b[i]
            logs.append(f"Back Substitution: x[{i}] = {x[i]}")

        return logs, x
