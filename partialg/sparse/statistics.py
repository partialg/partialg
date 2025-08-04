
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

def SBDError(matrix_size, sample_size, block_eigensolver, T=0, N=1 ):
    ''' 
    Compute error of random Hermitian matrices of size matrix_size up to sample_size.
    T : Translation factor
    N : Multiplier of spectrum
    '''
    tested_evs  = []
    ref_evs     = []
    #
    for i in range(sample_size):
        M   = sp.sparse.csc_array(np.random.rand(*matrix_size))
        M   = M @ M.T.conjugate()         # Building Hermitian matrix
        #
        # Fitting spectrum of M to domain (0, 1)
        np_evs  = np.sort( sp.sparse.linalg.eigs(M, sigma=0)[0] )
        summand = np_evs[0]
        norm    = (np_evs[-1] - summand)
        M       = (M - summand*np.eye(M.shape[0]) ) /norm   # Set from0 to 1
        M       = M*N + T*sp.sparse.eye(M.shape[0])                # Rescale by N and translate by T
        #
        ref_evs.append( (np.min( sp.sparse.linalg.eigs(M, sigma=0)[0] )/N -T )*norm + summand )
        #
        M2 = block_eigensolver(M)[0]
        M2 = M2 @ M2.T.conjugate()   # Forcing Hermiticity
        #
        tested_evs.append( (np.sqrt( np.min( sp.sparse.linalg.eigs( M2, sigma=0 )[0] ) )/N -T )*norm + summand )
    #
    ref_evs    = np.array(ref_evs)
    tested_evs = np.array(tested_evs)
    error      = np.abs( ref_evs - tested_evs )
    ratio      = error / ref_evs 
    #
    return {'error':error, 'mean_error': np.mean(error), 'ratio':ratio, 'mean_ratio':np.mean(ratio), 
            'error_std': np.std(error), 'ratio_std': np.std(ratio),
            'matrix_size':matrix_size, 'lower':T, 'upper':N+T }


def SBDErrorPlot(data, saveas=False):
    ''' 
    Plot outputs of SBDError.
    '''
    error       = data['error']
    mean_error  = data['mean_error']
    std         = data['error_std']
    matrix_size = data['matrix_size']
    lower       = data['lower']
    upper       = data['upper']
    #
    fig, ax = plt.subplots(figsize=(4,4))
    #
    dev1_ = np.round(-abs(mean_error) - std,6)
    dev1  = np.round(abs(mean_error) + std,6)
    dev2_ = np.round(-abs(mean_error) - 2*std,6)
    dev2  = np.round(abs(mean_error) + 2*std,6)
    dev3_ = np.round(-abs(mean_error) - 3*std,6)
    dev3  = np.round(abs(mean_error) + 3*std,6)
    #
    ax.axvline( mean_error, color='darkgreen')
    ax.axvspan( mean_error-std, mean_error+std, color='tab:green', alpha=0.3)
    ax.axvspan( mean_error-2*std, mean_error+2*std, color='tab:green', alpha=0.3)
    ax.axvspan( mean_error-3*std, mean_error+3*std, color='tab:green', alpha=0.3)
    #
    ax.hist( error, bins=50, edgecolor='black', density=False)
    #
    s1 = '$1\sigma$ interval: $\lambda_\mathrm{actual} \\approx \lambda_\mathrm{sbd} \,_{' + f'{ dev1_}' + '}^{+' + f'{dev1}' + '}$'
    s2 = '$2\sigma$ interval: $\lambda_\mathrm{actual} \\approx \lambda_\mathrm{sbd} \,_{' + f'{ dev2_}' + '}^{+' + f'{dev2}' + '}$'
    s3 = '$3\sigma$ interval: $\lambda_\mathrm{actual} \\approx \lambda_\mathrm{sbd} \,_{' + f'{ dev3_}' + '}^{+' + f'{dev3}' + '}$'
    #
    ax.text(0.99, 0.99, s=s1 + '\n' + s2 + '\n' + s3,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    #
    # ax.set_title(f'Error distribution for ground states from space [{lower}, {upper}]')
    ax.set_ylabel('Frequency (Arbitrary Unit)')
    ax.set_xlabel('Error')
    plt.tight_layout()
    #
    if type(saveas) == str:
        plt.savefig(saveas+'.png', dpi=800, format='png')
    else:
        plt.show()
