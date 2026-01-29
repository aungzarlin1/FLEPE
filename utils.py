import torch 
from torch import Tensor 
import torch_sparse
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.typing import OptTensor 
from sklearn.cluster import KMeans

import torch
import scipy.sparse
import scipy.sparse.linalg
import numpy as np



def get_sparse_diagonal(sparse_tensor: torch.Tensor) -> torch.Tensor:
  """
  Extracts the diagonal values from a sparse COO tensor.

  Args:
    sparse_tensor: A sparse COO tensor.

  Returns:
    A tensor containing the diagonal values.
  """
  if not sparse_tensor.is_coalesced():
    sparse_tensor = sparse_tensor.coalesce()
  indices = sparse_tensor.indices()
  values = sparse_tensor.values()
  row_indices = indices[0]
  col_indices = indices[1]
  diagonal_mask = row_indices == col_indices
  diagonal_values = values[diagonal_mask]
  return diagonal_values

def flow_laplacian(edge_index, method, vertex_importance = True, edge_weight: OptTensor = None):

    # print(f"Method : {method}, Vertex Importance : {vertex_importance}")
    edge_index, _ = remove_self_loops(edge_index)
    # edge_index = dataset.edge_index
    M = edge_index.size(1)  # Number of edges
    N = edge_index.max().item() + 1  # Number of nodes

    edge_weight = torch.ones(M, dtype=torch.float32)

    rows = torch.cat([edge_index[0], edge_index[1]])
    cols = torch.cat([torch.arange(M), torch.arange(M)])
    values = torch.cat([torch.ones(M), -torch.ones(M)])

    B = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (N, M), dtype=torch.float32).coalesce()

    Babs = torch.sparse_coo_tensor(B.indices(), B.values().abs(), B.shape)
    # print('1. B calculated')


    sigma = torch.sparse.mm(Babs, torch.ones((M, 1))).squeeze()
    # sigma = torch.sparse_coo_tensor(torch.arange(N).repeat(2), sigma.repeat(2), (N, N))
    

    sigma = torch.sparse_coo_tensor(
        torch.stack([torch.arange(N), torch.arange(N)]),  # Diagonal indices
        sigma,  # Values for the diagonal
        (N, N)  # Shape
    ).coalesce()

    # print('2. sigma calculated')


    if vertex_importance:
        nu_vals = torch.sparse.mm(Babs, edge_weight.abs().unsqueeze(1)).squeeze() / edge_weight.abs().sum()
    else:
        nu_vals = torch.ones(N)

    nu = torch.sparse_coo_tensor(
        torch.stack([torch.arange(N), torch.arange(N)]),  # Diagonal indices
        nu_vals,  # Values for the diagonal
        (N, N)  # Shape
    ).coalesce()

    # print('3. nu calculated')

    del Babs

    Le = torch.sparse.mm(B.T, torch.sparse.mm(nu, B))

    # Compute dual matrices
    Le_diag = get_sparse_diagonal(Le)

    # print('4. Le calculated')

    dualW = Le - torch.sparse_coo_tensor(
        torch.stack([torch.arange(M), torch.arange(M)]),  # Diagonal indices
        Le_diag,  # Values on the diagonal
        (M, M)  # Shape
    ).coalesce()

    # print('5. dualW calculated')


    dualD_values = torch.abs(dualW).sum(axis=1).values()
    dualD = torch.sparse_coo_tensor(
        torch.stack([torch.arange(M), torch.arange(M)]),  # Diagonal indices
        dualD_values,  # Diagonal values
        (M, M)  # Shape
    ).coalesce()

    # print('6. dualD calculated')

    if method == 'DPE':
        Lf = dualD + dualW
    elif method == 'PRE':
        Lf = dualD - dualW
    elif method == 'RGE':
        Lf = dualD - torch.abs(dualW)
    else:
        raise ValueError("Invalid method. Choose from 'DPE', 'PRE', or 'RGE'.")
    # print("Lf calculated")

    Lf = 0.5 * (Lf + Lf.T)
    # print("7. Flow Laplacian calulated")

    Bout = torch.sparse_coo_tensor(B.indices(), (B.values() > 0).float(), B.shape).coalesce()
    Bin = torch.sparse_coo_tensor(B.indices(), (B.values() < 0).float(), B.shape).coalesce()

    Dout = torch.sparse.mm(Bout, edge_weight.abs().unsqueeze(1)).squeeze()
    Din = torch.sparse.mm(Bin, edge_weight.abs().unsqueeze(1)).squeeze()

    sigma = sigma * nu
    tmp1 = sigma.values() / Dout
    tmp1[tmp1 == float("Inf")] = 0
    tmp2 = sigma.values() / Din
    tmp2[tmp2 == float("Inf")] = 0

    Fdiag_vals = 0.5 * edge_weight.abs() * (Bout.T @ tmp1 + Bin.T @ tmp2)

    del Bout, Bin, Dout, Din, sigma
    Fdiag_vals = torch.sqrt(Fdiag_vals)
    Fdiag_vals[Fdiag_vals > 0] = 1 / Fdiag_vals[Fdiag_vals > 0]


    Fdiag = torch.sparse_coo_tensor(torch.stack([torch.arange(M), torch.arange(M)]), Fdiag_vals, (M, M)).coalesce()
    # print('8. Fdiag calculated')

    normLf = torch.sparse.mm(Fdiag, torch.sparse.mm(Lf, Fdiag))
    normLf = 0.5 * (normLf + normLf.T)  # Ensure symmetry
    # print("Normalized flow laplacian calculated")

    return Lf,normLf

def eigen_decomposition(normlf):
    normlf = normlf.to_dense()
    L, V = torch.linalg.eig(normlf)

    L, V = L.real, V.real

    sorted_indices = np.argsort(L)  # Get the indices that would sort eigenvalues
    sorted_L = L[sorted_indices]  # Sort the eigenvalues
    sorted_V = V[:, sorted_indices]  # Sort the eigenvectors by corresponding eigenvalue order
    
    return sorted_V

def eigen_decomposition_scipy(sparse_matrix,k):
    # Convert PyTorch sparse tensor to SciPy sparse matrix
    sparse_matrix_coo = sparse_matrix.to_sparse().coalesce()  # Ensure the sparse matrix is in COO format
    scipy_sparse_matrix = scipy.sparse.coo_matrix(
        (sparse_matrix_coo.values().numpy(),
        (sparse_matrix_coo.indices()[0].numpy(), sparse_matrix_coo.indices()[1].numpy())),
        shape=sparse_matrix.shape
    )

    try:
        # Try computing eigenvalues with 'smallest magnitude'
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(scipy_sparse_matrix, k=k, which='SM')
    except Exception as e:
        print('eigs encounters singular matrix. Trying sigma = 1e-9.')
        # Retry with sigma = 1e-9
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(scipy_sparse_matrix, k=k, sigma=1e-9)

    # Use SciPy's eig to compute the eigenvalues and eigenvectors
    # eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(scipy_sparse_matrix,k = k, which='SM')
    eigvals, eigvecs = eigenvalues.real, eigenvectors.real
    sorted_indices = np.argsort(eigvals)  # Get the indices that would sort eigenvalues
    sorted_eigenvalues = eigvals[sorted_indices]  # Sort the eigenvalues
    sorted_eigenvectors = eigvecs[:, sorted_indices]  # Sort the eigenvectors by corresponding eigenvalue order
    DR = torch.tensor(sorted_eigenvectors)  # Convert sorted eigenvectors to a PyTorch tensor  
    return DR


