from qiskit.quantum_info import DensityMatrix, Pauli, SuperOp, Operator, Choi, PTM
import numpy as np

def is_ppt(choi):
    """Check if the Choi matrix is PPT (Positive Partial Transpose)"""
    # Convert to DensityMatrix object
    dm = DensityMatrix(choi.data)
    # Compute partial transpose over the second subsystem (qargs=[1])
    dm_pt = dm.partial_transpose(qargs=[1])
    # Check positive semi-definiteness with numerical tolerance
    return all(np.linalg.eigvalsh(dm_pt.data) >= 0)


#We define a function to check if a matrix becomes PPT after a certain number of iterations (Calling it Eventually PPT if that's the case)
def is_eppt(choi, app):
    """Check if the Choi matrix is EPPT (Eventually PPT), i.e EEB in out case, after a certain number of applications"""
    # Compose the Choi matrix with itself a certain number of times
    for _ in range(app):
        choi = choi.compose(choi)
    # Check if the resulting Choi matrix is PPT
    return is_ppt(choi)



def is_es_qubit(choi):
    """Check if a qubit channel is ES (Esntanglement Saving)"""
    # First we compute the Pauli Transfer Matrix of the state and then extract the relevant sub,atrix and vector
    ptm = PTM(SuperOp(choi)).data
    M = ptm[1:4, 1:4]
    c = ptm[1:4, 0]

    # Check determinant, as in the qubit case this would mean that the channel es EB direfctly.
    if np.linalg.det(M) == 0:
        return False

    # Check if the channel is unital, i.e if c = 0
    unital = np.array_equal(c, np.zeros(3))

    # Divide in unital and non unital case
    if unital:       
        eigenvalues = np.linalg.eigvals(M)
        return any(eig == -1 for eig in eigenvalues)
    else:
        # For the non unital case, we check if (M - I) n = -c has a solution with ||n|| = 1 (pure state)
        A = M - np.eye(3)
        b = -c
        # Solve A @ n = b
        n, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if residuals.size == 0 or np.all(residuals == 0):  # Exact solution exists
            # Check if n is a unit vector 
            if np.isclose(np.linalg.norm(n), 1):
                return True
        return False
