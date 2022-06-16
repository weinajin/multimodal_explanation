import torch
import torch.nn.functional as F
cudaAvailable = False
if torch.cuda.is_available():
	cudaAvailable = True
Tensor = torch.cuda.FloatTensor if cudaAvailable else torch.Tensor

BCELoss = torch.nn.BCELoss()
L1Loss = torch.nn.L1Loss(reduction='sum')
MSELoss = torch.nn.MSELoss()
def realTargetLoss(x):
	target = Tensor(x.shape[0], 1).fill_(1.0)
	return MSELoss(x, target)

def fakeTargetLoss(x):
	target = Tensor(x.shape[0], 1).fill_(0.0)
	return MSELoss(x, target)

def cycleLoss(a, a_):
	loss = MSELoss(a, a_)
	return loss

def VAE_loss(recon_x, x, mu, logvar):
	BCE = reconstruction_function(recon_x, x)  # mse loss
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.sum(KLD_element).mul_(-0.5)
	return BCE + KLD
