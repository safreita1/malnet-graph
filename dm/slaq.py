# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Main SLaQ interface for approximating graph descritptors NetLSD and VNGE."""
import scipy
import numpy as np
from scipy.sparse.base import spmatrix


def laplacian(adjacency, normalized=True):
    """Computes the sparse Laplacian matrix given sparse adjacency matrix as input.
    Args:
      adjacency (spmatrix): Input adjacency matrix of a graph.
      normalized (bool): If True, return the normalized version of the Laplacian.
    Returns:
      spmatrix: Sparse Laplacian matrix of the graph.
    """
    degree = np.squeeze(np.asarray(adjacency.sum(axis=1)))
    if not normalized:
        return scipy.sparse.diags(degree) - adjacency
    with np.errstate(divide='ignore'):  # Ignore the warning for divide by 0 case.
        degree = 1. / np.sqrt(degree)
    degree[degree == np.inf] = 0
    degree = scipy.sparse.diags(degree)
    return scipy.sparse.eye(adjacency.shape[0], dtype=np.float32) - degree @ adjacency @ degree


def lanczos_m(matrix, lanczos_steps, nvectors):
    """Implementation of Lanczos algorithm for sparse matrices.
    Lanczos algorithm computes symmetric m x m tridiagonal matrix T
    and matrix V with orthogonal rows constituting the basis of the
    Krylov subspace K_m(matrix, x), where x is an arbitrary starting unit vector.
    This implementation parallelizes `nvectors` starting vectors.
    The notation follows https://en.wikipedia.org/wiki/Lanczos_algorithm.
    Arguments:
      matrix (spmatrix): Sparse input matrix.
      lanczos_steps (int): Number of Lanczos steps.
      nvectors (int): Number of random vectors.
    Returns:
      T (np.ndarray): A (nvectors x m x m) tensor, T[i, :, :] is the ith symmetric
      tridiagonal matrix.
      V (np.ndarray): A (n x m x nvectors) tensor, V[:, :, i] is the ith matrix
      with orthogonal rows.
    """
    start_vectors = np.random.randn(matrix.shape[0], nvectors).astype(
        np.float32)  # Initialize random vectors in columns (n x nvectors).
    V = np.zeros((start_vectors.shape[0], lanczos_steps, nvectors), dtype=np.float32)
    T = np.zeros((nvectors, lanczos_steps, lanczos_steps), dtype=np.float32)

    np.divide(start_vectors, np.linalg.norm(start_vectors, axis=0), out=start_vectors)  # Normalize each column.
    V[:, 0, :] = start_vectors

    # First Lanczos step.
    w = matrix @ start_vectors
    alpha = np.einsum('ij,ij->j', w, start_vectors)
    w -= alpha[None, :] * start_vectors
    beta = np.einsum('ij,ij->j', w, w)
    np.sqrt(beta, beta)

    T[:, 0, 0] = alpha
    T[:, 0, 1] = beta
    T[:, 1, 0] = beta

    np.divide(w, beta[None, :], out=w)
    V[:, 1, :] = w

    t = np.zeros((lanczos_steps, nvectors), dtype=np.float32)

    # Further Lanczos steps.
    for i in range(1, lanczos_steps):
        old_vectors = V[:, i - 1, :]
        start_vectors = V[:, i, :]

        w = matrix @ start_vectors
        w -= beta[None, :] * old_vectors
        np.einsum('ij,ij->j', w, start_vectors, out=alpha)
        T[:, i, i] = alpha

        if i < lanczos_steps - 1:
            w -= alpha[None, :] * start_vectors
            # Orthogonalize columns of V.
            np.einsum('ijk,ik->jk', V, w, out=t)
            w -= np.einsum('ijk,jk->ik', V, t)
            np.einsum('ij,ij->j', w, w, out=beta)
            np.sqrt(beta, beta)
            np.divide(w, beta[None, :], out=w)

            T[:, i, i + 1] = beta
            T[:, i + 1, i] = beta
            V[:, i + 1, :] = w

            if (np.abs(beta) > 1e-6).sum() == 0:
                break
    return T, V


def slq(matrix, m, nvectors, functions, scales=np.ones(1)):
    """Stochastic Lanczos Quadrature approximation of given matrix functions.
    Arguments:
      matrix (spmatrix): Sparse input matrix.
      m (int): Number of Lanczos steps.
      nvectors (int): Number of random vectors.
      functions (List[Callable[np.ndarray, np.ndarray]]): A list of functions over
        the matrix spectrum.
      scales (np.ndarray): An array of scales to parametrize the functions. By
        default no scaling of the spectrum is used.
    Returns:
      traces (np.ndarray): a (nvectors x m x m) tensor, T[i, :, :] is the ith
      symmetric
      tridiagonal matrix.
    """
    T, _ = lanczos_m(matrix, m, nvectors)
    eigenvalues, eigenvectors = np.linalg.eigh(T)

    traces = np.zeros((len(functions), len(scales)))
    for i, function in enumerate(functions):
        expeig = function(np.outer(scales, eigenvalues)).reshape(len(scales), nvectors, m)
        sqeigv1 = np.power(eigenvectors[:, 0, :], 2)
        traces[i, :] = matrix.shape[-1] * (expeig * sqeigv1).sum(axis=-1).mean(axis=-1)

    return traces


def _slq_red_var_netlsd(matrix, lanczos_steps, nvectors, timescales):
    """Computes unnormalized NetLSD signatures of a given matrix.
    Uses the control variates method to reduce the variance of NetLSD estimation.
    Args:
      matrix (sparse matrix): Input adjacency matrix of a graph.
      lanczos_steps (int): Number of Lanczos steps.
      nvectors (int): Number of random vectors for stochastic estimation.
      timescales (np.ndarray): Timescale parameter for NetLSD computation. Default
        value is the one used in both NetLSD and SLaQ papers.
    Returns:
      np.ndarray: Approximated NetLSD descriptors.
      """
    functions = [np.exp, lambda x: x]
    traces = slq(matrix, lanczos_steps, nvectors, functions, -timescales)
    subee = traces[0, :] - traces[1, :] / np.exp(timescales)
    sub = -timescales * matrix.shape[0] / np.exp(timescales)
    return np.array(subee + sub)


def _slq_red_var_vnge(matrix, lanczos_steps, nvectors):
    """Approximates Von Neumann Graph Entropy (VNGE) of a given matrix.
    Uses the control variates method to reduce the variance of VNGE estimation.
    Args:
      matrix (sparse matrix): Input adjacency matrix of a graph.
      lanczos_steps (int): Number of Lanczos steps.
      nvectors (int): Number of random vectors for stochastic estimation.
    Returns:
      float: Approximated von Neumann graph entropy.
    """
    functions = [lambda x: -np.where(x > 0, x * np.log(x), 0), lambda x: x]
    traces = slq(matrix, lanczos_steps, nvectors, functions).ravel()
    return traces[0] - traces[1] + 1


def vnge(adjacency, lanczos_steps=10, nvectors=10, seed=1):
    """Computes Von Neumann Graph Entropy (VNGE) using SLaQ.
    Args:
      adjacency (scipy.sparse.base.spmatrix): Input adjacency matrix of a graph.
      lanczos_steps (int): Number of Lanczos steps. Setting lanczos_steps=10 is
        the default from SLaQ.
      nvectors (int): Number of random vectors for stochastic estimation. Setting
        nvectors=10 is the default values from the SLaQ paper.
    Returns:
      float: Approximated VNGE.
    """
    np.random.seed(seed)
    if adjacency.nnz == 0:  # By convention, if x=0, x*log(x)=0.
        return 0
    density = laplacian(adjacency, False)
    density.data /= np.sum(density.diagonal()).astype(np.float32)

    return _slq_red_var_vnge(density, lanczos_steps, nvectors)


def netlsd(adjacency, timescales=np.logspace(-2, 2, 256), lanczos_steps=10, nvectors=10, normalization=None, seed=1):
    """Computes NetLSD descriptors using SLaQ.
    Args:
      adjacency (sparse matrix): Input adjacency matrix of a graph.
      timescales (np.ndarray): Timescale parameter for NetLSD computation. Default
        value is the one used in both NetLSD and SLaQ papers.
      lanczos_steps (int): Number of Lanczos steps. Setting lanczos_steps=10 is
        the default from SLaQ.
      nvectors (int): Number of random vectors for stochastic estimation. Setting
        nvectors=10 is the default values from the SLaQ paper.
      normalization (str): Normalization type for NetLSD.
    Returns:
      np.ndarray: Approximated NetLSD descriptors.
    """
    np.random.seed(seed)
    lap = laplacian(adjacency, True)
    hkt = _slq_red_var_netlsd(lap, lanczos_steps, nvectors,
                              timescales)  # Approximated Heat Kernel Trace (hkt).
    if normalization is None:
        return hkt
    n = lap.shape[0]
    if normalization == 'empty':
        return hkt / n
    elif normalization == 'complete':
        return hkt / (1 + (n - 1) * np.exp(-timescales))
    elif normalization is None:
        return hkt
    else:
        raise ValueError("Unknown normalization type: expected one of [None, 'empty', 'complete'], got", normalization)


def netlsd_naive(adjacency, timescales=np.logspace(-2, 2, 256)):
    """Computes NetLSD with full eigendecomposition, in a naïve way.
    Args:
      adjacency (spmatrix): Input sparse adjacency matrix of a graph.
      timescales (np.ndarray): A 1D array with the timescale parameter of NetLSD.
        Default value is the one used in both NetLSD and SLaQ papers.
    Returns:
      np.ndarray: NetLSD descriptors of the graph.
    """
    lap = laplacian(adjacency)
    lambdas, _ = scipy.linalg.eigh(lap.todense())
    return np.exp(-np.outer(timescales, lambdas)).sum(axis=-1)


def vnge_naive(adjacency):
    """Computes Von Neumann Graph Entropy (VNGE) with full eigendecomposition.
    Args:
      adjacency (spmatrix): Input sparse adjacency matrix of a graph.
    Returns:
      float: Von Neumann entropy of the graph.
    """
    density = laplacian(adjacency, normalized=False)
    density.data /= np.sum(density.diagonal())
    eigenvalues, _ = scipy.linalg.eigh(density.todense())
    return -np.where(eigenvalues > 0, eigenvalues * np.log(eigenvalues), 0).sum()