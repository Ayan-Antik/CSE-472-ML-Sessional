import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
from sklearn.decomposition import PCA

def plot_data():
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1],s=10, c='b', marker='o', alpha=0.5)
    plt.title('Data')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
# initialize parameters
def init_params(X, K):

    N, D = X.shape                                                  
    mu = np.zeros((K, D)) # mean
    sigma = np.zeros((K, D, D)) # covariance
    pi = np.zeros(K) # weight (3,)
    gamma = np.zeros((N, K)) # responsibility of shape (datapoints, clusters)(1500, 3)
    for i in range(K):
        mu[i] = X[np.random.choice(N)] 
        sigma[i] = np.eye(D) 
        pi[i] = 1.0 / K 

    return mu, sigma, pi, gamma

# compute the log likelihood
def log_likelihood(X, mu, sigma, pi):


    N = X.shape[0]
    K = pi.shape[0]
    log_likelihood = np.zeros(N)
    reg_cov = 1e-6 * np.eye(X.shape[1])

    for j in range(K):
        log_likelihood += pi[j] * multivariate_normal.pdf(X, mu[j], sigma[j] + reg_cov, allow_singular=True)
    return np.sum(np.log(log_likelihood))

def animate(X, mu, sigma, gamma, x, y, XY):
    plt.ion()
    fig = plt.figure(figsize=(10, 8), num=1)
    ax1 = fig.add_subplot(111)
    ax1.set_title('Contour Plot')   
    ax1.scatter(X[:,0], X[:,1], c=[sns.color_palette()[i] for i in np.argmax(gamma, axis=1)], s=10, alpha=0.5)
    for m, c in zip(mu, sigma):
        c += 1e-6*np.eye(X.shape[1])
        multi_normal = multivariate_normal(mean=m,cov=c)
        ax1.contour(x, y ,multi_normal.pdf(XY), colors='black', alpha=0.3, extend = 'min')
        ax1.scatter(m[0],m[1],c='grey',zorder=10,s=100)
        
    plt.pause(0.0002)
    fig.clf()
    plt.show()

def EM_Algorithm(iteration, X, mu, sigma, pi, gamma, draw = False):

    
    N = X.shape[0]
    K = pi.shape[0]
   
    log_like = log_likelihood(X, mu, sigma, pi)
    if draw:
        
        x = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
        y = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
        x, y = np.meshgrid(x, y) 
        XY = np.dstack((x, y))
        # print(XY.shape) # (2250000, 2)
        # print(mu.shape, sigma.shape) # (3, 2), (3, 2, 2)
        
        # plt.ion()
        

    # np.seterr(divide = 'ignore', invalid='ignore')
    for _ in range(iteration):
        
        reg_cov = 1e-6 * np.eye(X.shape[1])
        for j in range(K):
            gamma[:, j] = pi[j] * multivariate_normal.pdf(X, mu[j], sigma[j] + reg_cov, allow_singular=True)
        gamma /= np.sum(gamma, axis=1).reshape(-1, 1)

        # M-step
        for j in range(K):
            Nk = np.sum(gamma[:, j])
            mu[j] = np.sum(gamma[:, [j]] * X, axis=0) / Nk
            sigma[j] = np.dot((gamma[:, j] * (X - mu[j]).T), (X - mu[j])) / Nk + reg_cov
            pi[j] = Nk / N

            if draw:
                animate(X, mu, sigma, gamma, x, y, XY)

        # compute log likelihood
        new_log_like = log_likelihood(X, mu, sigma, pi)
        if np.abs(new_log_like - log_like) < 0.01:
            return mu, sigma, pi, gamma, log_like
        log_like = new_log_like
    return mu, sigma, pi, gamma, log_like

def find_convergence_point(ll_array, threshold=0.01):
    for i in range(1, len(ll_array)):
        if abs((ll_array[i] - ll_array[i-1]) / ll_array[i-1]) < threshold:
            return i
    return -1   

def doPCA(X, K, mu, sigma):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    plot_sigma = np.zeros((K, 2, 2))
    for i in range(K):
        plot_sigma[i] = np.dot(np.dot(pca.components_, sigma[i]), pca.components_.T)
    # plot_mu = pca.transform(mu)
    plot_mu = np.zeros((K, 2))
    for i in range(K):
        plot_mu[i] = np.dot(pca.components_, (mu[i] - pca.mean_))
    return X, plot_mu, plot_sigma



if __name__ == '__main__':
    file = sys.argv[1]
    X = np.loadtxt(file)
    
    log_likelihoods = []
    
    for K in range(1, 11):
        mu, sigma, pi, gamma = init_params(X, K)
        max_iter = 100
        mu, sigma, pi, gamma, log_like = EM_Algorithm(max_iter, X, mu, sigma, pi, gamma)
        log_likelihoods.append(log_like)
        print(f'K = {K}, log likelihood = {log_like}\n')

    #plot K vs log likelihood
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 11), log_likelihoods)
    plt.xlabel('K')
    plt.ylabel('Log likelihood')
    plt.show()
    k_star = find_convergence_point(log_likelihoods)
    # k_star = 3
    print(f'Convergence point: {k_star}')
    mu, sigma, pi, gamma = init_params(X, k_star)
    
    if X.shape[1] != 2:
        X, mu, sigma = doPCA(X, k_star, mu, sigma)

    mu, sigma, pi, gamma, log_like = EM_Algorithm(max_iter, X, mu, sigma, pi, gamma, draw=True)
    

    
    

