import numpy as np

# https://eprint.iacr.org/2018/1041.pdf (E2DM)

def sigma_diagonal_vector(d: int, k:int) -> np.array:
    """Creates the k-th diagonal for the sigma operator
    for matrices of dimension dxd."""
    
    u = np.arange(d**2)
    if k >= 0:
        index = (u - d*k >= 0) & (u < d*k + d - k)
    else:
        index = (u - d*(d+k) >= -k ) & (u - d*(d+k)< d)
    u[index] = 1
    u[~index] = 0
    return u

def tau_diagonal_vector(d: int, k:int) -> np.array:
    """Creates the k-th diagonal for the tau operator
    for matrices of dimension dxd."""
    
    u = np.zeros(d**2)
    for i in range(d):
        l = (k + d * i)
        u[l] = 1
    return u

def row_diagonal_vector(d,k):
    v_k = np.arange(d**2)
    index = (v_k % d) < (d - k)
    v_k[index] = 1
    v_k[~index] = 0
    
    v_k_d = np.arange(d**2)
    index = ((v_k_d % d) >= (d -k)) & ((v_k_d % d) < d)
    v_k_d[index] = 1
    v_k_d[~index] = 0
    return v_k, v_k_d

def column_diagonal_vector(d,k):
    v_k = np.ones(d**2)
    return v_k

class MatrixMultiplicator:
    """Base class to create a matrix multiplicator operator."""
    def __init__(self, d, create_zero=None, 
                sigma_diagonal_vector = sigma_diagonal_vector, 
                tau_diagonal_vector=tau_diagonal_vector,
                row_diagonal_vector=row_diagonal_vector,
                column_diagonal_vector=column_diagonal_vector,
                rotate=None, add=None, pmult=None, cmult=None):
        
        if create_zero is None:
            create_zero = lambda: np.zeros(d*d)
        self.d = d
        self.create_zero = create_zero
        self.sigma_diagonal_vector = sigma_diagonal_vector
        self.tau_diagonal_vector = tau_diagonal_vector
        self.row_diagonal_vector = row_diagonal_vector
        self.column_diagonal_vector = column_diagonal_vector
        
        if not rotate:
            rotate = lambda x,k: np.roll(x, -k)
        if not add:
            add = lambda x,y: x+y
        if not pmult:
            pmult = lambda x,y: x*y
        if not cmult:
            cmult = lambda x,y: x*y
            
        self.rotate, self.add, self.pmult, self.cmult = rotate, add, pmult, cmult
    
    def sigma_lin_transform(self, input):
        
        sigma = []
        d = self.d
    
        for k in range(-d+1,d):
            sigma.append(self.sigma_diagonal_vector(d,k))
        
        output = self.create_zero()
        
        for sigma_vector,k in zip(sigma,range(-d+1,d)):
            output = self.add(output, self.pmult(self.rotate(input,k), sigma_vector))
        return output
    
    def tau_lin_transform(self, input):

        tau = []
        d = self.d

        for k in range(d):
            tau.append(self.tau_diagonal_vector(d,k))
            
        output = self.create_zero()
        
        for tau_vector,k in zip(tau,range(d)):
            output = self.add(output, self.pmult(self.rotate(input,k * d), tau_vector))
        return output
    
    def row_lin_transform(self, input, k):
        
        d = self.d
        v_k, v_k_d = self.row_diagonal_vector(d, k)
        
        output = self.create_zero()
        
        output = self.add(output, self.pmult(self.rotate(input, k), v_k))
        output = self.add(output, self.pmult(self.rotate(input, k-d), v_k_d))

        return output
    
    def column_lin_transform(self, input, k):
        
        d = self.d
        v_k = self.column_diagonal_vector(d, k)
        
        output = self.create_zero()
        
        output = self.add(output, self.pmult(self.rotate(input, d*k),v_k))

        return output
    
    def matmul(self, A, B):
        
        d = self.d

        sigma_A = self.create_zero()
        sigma_A = self.sigma_lin_transform(A)

        tau_B = self.create_zero()
        tau_B = self.tau_lin_transform(B)

        output = self.cmult(sigma_A, tau_B)

        for k in range(1,d):
            shift_A = self.row_lin_transform(sigma_A, k)
            shift_B = self.column_lin_transform(tau_B, k)

            output = self.add(output, self.cmult(shift_A, shift_B))
        
        return output

def encode_matrices_to_vector(matrix):
    shape = matrix.shape
    assert len(shape) == 3, "Non tridimensional tensor"
    assert shape[1] == shape[2], "Non square matrices"
    
    g = shape[0]
    d = shape[1]
    n = g * (d ** 2)
    
    output = np.zeros(n)
    for l in range(n):
        k = l % g
        i = (l // g) // d
        j = (l // g) % d
        output[l] = matrix[k,i,j]
        
    return output

def decode_vector_to_matrices(vector, d):
    n = len(vector)
    g = n // (d ** 2)
    
    output = np.zeros((g, d, d))
    
    for k in range(g):
        for i in range(d):
            for j in range(d):
                output[k,i,j] = vector[g * (d*i + j) +k]
    return output

def encode_matrix_to_vector(matrix: np.array) -> np.array:
    """Encodes a d*d matrix to a vector of size d*d"""
    shape = matrix.shape
    assert len(shape) == 2 and shape[0] == shape[1], "Non square matrix"
    d = shape[0]
    output = np.zeros(d**2)
    for l in range(d**2):
        i = l // d
        j = l % d
        output[l] = matrix[i,j]
    return output

def decode_vector_to_matrix(vector):
    n = len(vector)
    d = np.sqrt(n)
    assert len(vector.shape) == 1 and d.is_integer(), "Non square matrix"
    d = int(d)
    
    output = np.zeros((d,d))
    
    for i in range(d):
        for j in range(d):
            output[i,j] = vector[d*i + j]
    return output

def weave(vector, g):
    output = np.zeros(len(vector) * g)
    for i in range(len(vector)):
        output[i*g:(i+1)*g] = vector[i]
    return output