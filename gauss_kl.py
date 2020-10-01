import numpy as np
import  torch
import torch.utils
import torch.utils.data

def kl_divergence(mu1, mu2, sigma_1, sigma_2):
	sigma_diag_1 = np.eye(sigma_1.shape[0]) * sigma_1
	sigma_diag_2 = np.eye(sigma_2.shape[0]) * sigma_2

	sigma_diag_2_inv = np.linalg.inv(sigma_diag_2)

	kl = 0.5 * (np.log(np.linalg.det(sigma_diag_2) / np.linalg.det(sigma_diag_1))
				- mu1.shape[0] + np.trace(np.matmul(sigma_diag_2_inv, sigma_diag_1))
				+ np.matmul(np.matmul(np.transpose(mu2 - mu1), sigma_diag_2_inv), (mu2 - mu1))
				)



	return kl
def kl_div_std_gauss(mu1, sigma_1):
	sigma_diag_1 = np.eye(sigma_1.shape[0]) * sigma_1
	# print ("a ", - mu1.shape[0] + np.trace(sigma_diag_1)+np.matmul(np.transpose(mu1),   mu1))
	# print ("b ", -np.log( np.linalg.det(sigma_diag_1)))
	kl = 0.5 * (-np.log( np.linalg.det(sigma_diag_1))
				- mu1.shape[0] + np.trace(sigma_diag_1)+np.matmul(np.transpose(mu1),   mu1)
				)



	return kl
def kl_div_std_gauss_torch(mu1, sigma_1):
	tt = mu1.mul(mu1)
	t2 = torch.sum(tt, axis=2)

	zz = torch.sum(sigma_1, axis=2)
	yy= -torch.sum(torch.log(sigma_1),axis=2)


	kl = 0.5 * (yy	- mu1.shape[2] + zz+t2)



	return kl

if __name__ =="__main__":
	x =torch.rand(3,1,5)
	y= torch.rand(3,1,5)
	x0=x.detach().numpy()
	y0 =y.detach().numpy()

	print ("hhh p", kl_div_std_gauss_torch(x, y))
	print ("hhh ",kl_div_std_gauss(x0[0,0,:], y0[0,0,:]))
	print ("hhh ",kl_div_std_gauss(x0[1,0,:], y0[1,0,:]))
	print ("hhh ",kl_div_std_gauss(x0[2,0,:], y0[2,0,:]))

	exit(44)
	print (torch.log(x).shape)
	exit(23)
	zz =torch.sum(y,axis=2)
	print ("zz=",zz)
	print (np.sum(y0,axis=2))
	exit(45)
	print (np.power( np.linalg.norm(x0[0,0,:]),2))
	print(np.power(np.linalg.norm(x0[1, 0, :]), 2))
	print(np.power(np.linalg.norm(x0[2, 0, :]), 2))

	# print (np.matmul(np.transpose(x0),   x0))
	tt= x.mul(x)
	print (tt)
	t2 =torch.sum(tt,axis=2)
	print(tt.shape,t2.shape)
	print ("t2=",t2,zz)
	print ("sum=",t2+zz)
	exit(43)
	zz =torch.matmul(torch.transpose(x),x)
	print (zz.shape)
	exit(444)
	mm=torch.eve(64,1,77)

	y=torch.rand(2,1,77)
	y1= torch.diag(y,2)
	print (y1)
	exit(44)
	sigma_diag_1 = torch.matmul(torch.eye(y.shape[2]), y)


	mu0= np.asarray([0.,0,0.,0.])
	print (np.diag(mu0))
    # s0=  np.asarray([[3.,0,3,0.],[0,2,0.,0],[0.,0.,2.,0],[0,0,0.,2.]])
    # mu1= np.asarray([0.,0,0,0.])
    # s1=  np.asarray([[3.,0,3,0.],[0,2,0.,0],[0.,0.,2.,0],[0,0,0.,2.]])
    # print (kl_divergence(mu0, mu1, s0, s1))
    # print(kl_div_std_gauss(mu0,s0))