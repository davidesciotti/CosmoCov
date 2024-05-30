import time
from matplotlib import cm
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm
sys.path.append('/home/davide/Documenti/Lavoro/Programmi/Spaceborne')
import spaceborne.my_module as mm

def get_cov(filename):

	data = np.genfromtxt(filename)
	ndata = int(np.max(data[:,0]))+1

	print("Dimension of cov: %dx%d"%(ndata,ndata))

	ndata_min = int(np.min(data[:,0]))
	cov_g = np.zeros((ndata,ndata))
	cov_ng = np.zeros((ndata,ndata))
	for i in tqdm(range(ndata_min,data.shape[0])):
		cov_g[int(data[i,0]),int(data[i,1])] =data[i,8]
		cov_g[int(data[i,1]),int(data[i,0])] =data[i,8]
		cov_ng[int(data[i,0]),int(data[i,1])] =data[i,9]
		cov_ng[int(data[i,1]),int(data[i,0])] =data[i,9]

	return cov_g, cov_ng, ndata


def cov2corr(covariance):
    """ Taken from 
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    """

    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


# if __name__ == '__main__':

# covfile = 'output/out_cov_C01/C01_ssss_--_cov_Ntheta40_Ntomo10_1'  # first block, to test the script
covfile = 'cov_ssss_--_full'  # first block, to test the script

check_posdef = False
zbins = 10
chunk_size = 20_000
deg_to_rad = np.pi/180
deg_to_arcmin = 60
arcmin_to_deg = deg_to_arcmin**-1
theta_unit = 'rad'
# covfile = 'cov_test'
order_file = 'output/out_cov_C01/order_C01_i_8400-8439'
order = np.genfromtxt(order_file)
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
ind = mm.build_full_ind('triu', 'row-major', zbins)  # TODO paly around with these?
ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]

# check theta values
theta_values_out = order[:, 1]
theta_bins = len(theta_values_out)
theta_values_in = np.geomspace(0.1, 10, theta_bins)*deg_to_rad

plt.plot(theta_values_out, label='theta_values_out', marker='.')
plt.plot(theta_values_in, label='theta_values_in', marker='.')
plt.legend()

theta_indices = {theta: idx for idx, theta in enumerate(theta_values_out)}

# default load
c_g, c_ng, ndata = get_cov(covfile)	
cov = c_ng+c_g

# my load
cov_g_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
cov_ng_6d = np.zeros((theta_bins, theta_bins, zbins, zbins, zbins, zbins))
# let't load it in a dataframe, as done for OneCovariance, to check the ordering
column_names = ['idx_1', 'idx_2', 'theta1', 'theta2', 'zi', 'zj', 'zk', 'zl', 'covg', 'covng']
start = time.perf_counter()
for df_chunk in pd.read_csv(f'{covfile}', delim_whitespace=True, names=column_names, comment='#', chunksize=chunk_size):

    # Use apply to vectorize the extraction of probes
    # extracted_probes = df_chunk['#obs'].apply(lambda x: extract_probes(x, probe_names))
    # probe_idxs = np.array([[probe_idx_dict[probe[0]], probe_idx_dict[probe[1]]] for probe in extracted_probes])

    theta1_indices = df_chunk['theta1'].map(theta_indices).values
    theta2_indices = df_chunk['theta2'].map(theta_indices).values

    z_indices = df_chunk[['zi', 'zj', 'zk', 'zl']].values

    # index_tuple = (probe_idxs[:, 0], probe_idxs[:, 1], theta1_indices, theta2_indices,
                #    z_indices[:, 0], z_indices[:, 1], z_indices[:, 2], z_indices[:, 3])

    index_tuple = (theta1_indices, theta2_indices,
                   z_indices[:, 0], z_indices[:, 1], z_indices[:, 2], z_indices[:, 3])

    cov_g_6d[index_tuple] = df_chunk['covg'].values
    cov_ng_6d[index_tuple] = df_chunk['covng'].values

print('df optimized loaded in ', time.perf_counter() - start, ' seconds')

# let's try to revert to 2D
cov_4d = mm.cov_6D_to_4D(cov_g_6d, theta_bins, zpairs_cross, ind_cross)
cov_2d = mm.cov_4D_to_2D(cov_4d, block_index='ij')
mm.matshow(cov_2d, log=True, abs_val=True)



# load signal
cl_input_folder = '/home/davide/Documenti/Lavoro/Programmi//CLOE_validation/output/v2.0.2/C01'
xi_gg_2d = np.genfromtxt(f'{cl_input_folder}/xi-ij-GG-PyCCL-C01.dat')
xi_gl_2d = np.genfromtxt(f'{cl_input_folder}/xi-ij-GL-PyCCL-C01.dat')
xi_pp_2d = np.genfromtxt(f'{cl_input_folder}/xi-ij-Lplus-PyCCL-C01.dat')
xi_mm_2d = np.genfromtxt(f'{cl_input_folder}/xi-ij-Lminus-PyCCL-C01.dat')

theta_xi_deg = xi_gg_2d[:, 0]
theta_arcmin = theta_xi_deg * deg_to_arcmin
theta_rad = theta_xi_deg * deg_to_rad

if theta_unit == 'arcmin':
    theta_arr = theta_arcmin
elif theta_unit == 'deg':
    theta_arr = theta_xi_deg
elif theta_unit == 'rad':
	theta_arr = theta_xi_deg * deg_to_rad
else:
    raise ValueError('theta unit not recognised')

xi_gg_2d = xi_gg_2d[:, 1:]
xi_gl_2d = xi_gl_2d[:, 1:]
xi_pp_2d = xi_pp_2d[:, 1:]
xi_mm_2d = xi_mm_2d[:, 1:]

xi_gg_3D = mm.cl_2D_to_3D_symmetric(xi_gg_2d, theta_bins, zpairs_auto, zbins)
xi_gl_3D = mm.cl_2D_to_3D_asymmetric(xi_gl_2d, theta_bins, zbins=zbins, order='row-major')
xi_pp_3D = mm.cl_2D_to_3D_symmetric(xi_pp_2d, theta_bins, zpairs_auto, zbins)
xi_mm_3D = mm.cl_2D_to_3D_symmetric(xi_mm_2d, theta_bins, zpairs_auto, zbins)

# load OC uncertainties

thetas_arcmin_oc = np.array([  6.        ,   6.7520135 ,   7.59828104,   8.55061602,
         9.62231246,  10.8283306 ,  12.18550573,  13.71278318,
        15.43148285,  17.36559675,  19.54212393,  21.99144742,
        24.7477583 ,  27.849533  ,  31.34007045,  35.26809643,
        39.68844385,  44.66281808,  50.2606584 ,  56.56010731,
        63.64910131,  71.6265985 ,  80.60395996,  90.70650424,
       102.07525679, 114.86891857, 129.2660814 , 145.46772102,
       163.70000259, 184.21743779, 207.30643768, 233.28931082,
       262.5287625 , 295.4329579 , 332.4612198 , 374.13044048,
       421.0222972 , 473.79137211, 533.17428976, 600.        ])
thetas_rad_oc = thetas_arcmin_oc * arcmin_to_deg * deg_to_rad

# import OC variances
variance_w_oc = np.genfromtxt('/home/davide/Documenti/Lavoro/Programmi/CosmoCov/covs/output/OneCovariance_2PCF_variances/variance_xim.dat')
variance_w_oc_3d = np.zeros((theta_bins, zbins, zbins))
count = 0
for theta_i in range(theta_bins):
	for zi in range(zbins):
		for zj in range(zi, zbins):
			variance_w_oc_3d[theta_i, zi, zj] = variance_w_oc[count]
			count += 1


colors = cm.rainbow(np.linspace(0, 1, zbins))
for zi in range(zbins):
	uncert_w_cc = np.sqrt(np.diag(cov_g_6d[:, :, zi, zi, zi, zi]))
	plt.loglog(theta_values_out, xi_mm_3D[:, zi, zi], label=f'z{zi}', c=colors[zi])
	plt.loglog(theta_values_out, uncert_w_cc, c=colors[zi], ls=':')
	plt.loglog(thetas_rad_oc, np.sqrt(variance_w_oc_3d[:, zi, zi]), c=colors[zi], ls='--')
plt.title('w')

# Create the initial legend for plot lines
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles, labels=labels)

# Custom legend entries
custom_lines = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='2PCF'),
    Line2D([0], [0], color='black', lw=2, linestyle=':', label='CosmoLike unc.'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='OneCovariance unc.')
]

# Add the custom legend entries
handles.extend(custom_lines)
labels.extend(['2PCF', 'CosmoCov unc.', 'OneCovariance unc.'])
plt.legend(handles=handles, labels=labels, loc='center right', bbox_to_anchor=(1.4, 0.5))

plt.xlabel(f'theta [{theta_unit}]')


cmap = 'viridis'

# compute correlation matrix - slow
# pp_norm = np.zeros((ndata,ndata))
# for i in tqdm(range(ndata)):
# 	for j in range(ndata):
# 		pp_norm[i][j] = cov[i][j]/ np.sqrt(cov[i][i]*cov[j][j])

# compute correlation matrix - fast
pp_norm = cov2corr(cov)

print("Plotting correlation matrix ...")

plot_path = covfile+'_plot.pdf'
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# plt.axvline(x=10*20,color='black')
# plt.axvline(x=20*20,color='black')
# plt.axvline(x=40*20,color='black')
# plt.axhline(y=10*20,color='black')
# plt.axhline(y=20*20,color='black')
# plt.axhline(y=40*20,color='black')
im3 = ax.imshow(pp_norm, cmap=cmap, vmin=-1, vmax=1)
im3 = ax.imshow(np.log10(cov), cmap=cmap)
fig.colorbar(im3, orientation='vertical')
# ax.text(65, -15, r'$\xi_+^{ij}(\theta)$', fontsize=14)
# ax.text(265, -15, r'$\xi_-^{ij}(\theta)$', fontsize=14)
# ax.text(565, -15, r'$\gamma_t^{ij}(\theta)$', fontsize=14)
# ax.text(815, -15, r'$w^{i}(\theta)$', fontsize=14)

# ax.text(905, 95, r'$\xi_+$', fontsize=14)
# ax.text(905, 295, r'$\xi_-$', fontsize=14)
# ax.text(905, 595, r'$\gamma_t$', fontsize=14)
# ax.text(905, 845, r'$w$', fontsize=14)
plt.savefig(plot_path,dpi=2000)
plt.show()
print("Plot saved as %s"%(plot_path))

if check_posdef:
	b = np.sort(LA.eigvals(cov))
	print("min, max eigenvalues cov: %e, %e"%(np.min(b), np.max(b)))
	if(np.min(b)<=0.):
		print("non-positive eigenvalue encountered! Covariance Invalid!")
	
	print("Covariance is postive definite!")
