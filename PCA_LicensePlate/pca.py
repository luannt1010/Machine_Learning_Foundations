import numpy as np

def center_data(X):
    n = X.shape[0]
    mean_vec = np.mean(X, axis=0)
    mean_matrix = np.reshape(np.repeat(mean_vec, n),
                             (X.shape), order="F")
    return X - mean_matrix, mean_vec
def get_covariance_matrix(X):
    centered_data, _ = center_data(X)
    n = X.shape[0]
    cov_matrix = centered_data.T @ centered_data / (n - 1)
    return cov_matrix

def cal_eigen(cov_matrix):
    return np.linalg.eig(cov_matrix)
#
# def scale(eigenvectors):
#     norm_val = np.linalg.norm(eigenvectors, axis=0)
#     n = eigenvectors.shape[0]
#     scaled_matrix = eigenvectors / np.reshape(np.repeat(norm_val, n),
#                                               (eigenvectors.shape), order="F")
#     return scaled_matrix

def reduce_dimension(X, V, k):
    eigenvector_scaled = V[:, :k]
    centered_data, mean = center_data(X)
    X_pca = centered_data @ eigenvector_scaled
    return X_pca, mean

def perform_pca(X, k):
    cov_matrix = get_covariance_matrix(X)
    eigenvalues, eigenvectors = cal_eigen(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    X, mean = reduce_dimension(X, eigenvectors, k)
    return X, eigenvectors, mean

def reconstruct_image(Xred, eigenvecs, mean):
    X_reconstructed = Xred.dot(eigenvecs[:,:Xred.shape[1]].T) + mean
    return X_reconstructed




