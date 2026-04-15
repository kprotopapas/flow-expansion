import torch

def rbf_kernel(X, gamma=1.0):
    """
    Compute the RBF kernel (Gaussian kernel) between all pairs of samples in X.
    """
    sq_dists = torch.cdist(X, X, p=2) ** 2
    return torch.exp(-gamma * sq_dists)


def vendi_score(features, gamma=1.0):
    """
    Calculate the Vendi score for a set of features.
    
    Arguments:
    ---------
    features : torch.Tensor
        Feature tensor of shape (n_samples, n_features).

    Returns:
    -------
    vendi_score : float
        The Vendi score.
    """

    norm_features = features / features.norm(dim=1, keepdim=True)
    
    # Compute the covariance matrix
    K = norm_features @ norm_features.T / features.shape[0]
    K_rbf = rbf_kernel(features, gamma) 


    # Compute the eigenvalues and eigenvectors
    eigenvalues, _ = torch.linalg.eig(K)
    eigenvalues = eigenvalues.real  # Take the real part of the eigenvalues

    eig_rbf, _ = torch.linalg.eig(K_rbf)
    eig_rbf = eig_rbf.real  # Take the real part of the eigenvalues

    # Sort the eigenvalues in descending order
    
    # Compute the Vendi score
    vendi_score = torch.exp(torch.sum(-eigenvalues[eigenvalues>0]*torch.log(eigenvalues[eigenvalues>0])))
    vendi_score = vendi_score.item()
    
    vendi_score_rbf = torch.exp(torch.sum(-eig_rbf[eig_rbf>0]*torch.log(eig_rbf[eig_rbf>0])))
    # round to 3 decimal places
    vendi_score_rbf = vendi_score_rbf.item()
    vendi_score = round(vendi_score, 3)
    vendi_score_rbf = round(vendi_score_rbf, 3)
    return vendi_score, vendi_score_rbf



