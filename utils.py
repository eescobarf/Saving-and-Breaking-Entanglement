from qiskit.quantum_info import DensityMatrix
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