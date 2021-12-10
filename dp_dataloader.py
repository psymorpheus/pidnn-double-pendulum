import numpy as np
import torch
from pyDOE import lhs
from torch.nn.modules import loss
from torch.utils.data import Dataset, DataLoader

AAT_test = None
A_test = None
AAT_validation = None
A_validation = None

def set_loss(model, device, batch_size, data = None, A_true = None):
	# Returns relative MSE
	if data is None:
		data = AAT_validation
		A_true = A_validation
	# loss_function = torch.nn.MSELoss(reduction ='mean')
	f_hat = torch.zeros_like(A_true)
	with torch.no_grad():
		prediction = model.forward(data)
	return model.loss_function(prediction, A_true)/model.loss_function(A_true, f_hat)

	
	error = 0
	actual = 0
	size = A_true.shape[0]
	if size<1000: batch_size = size
	for i in range(0, size, batch_size):
		batch_data = data[i:min(i+batch_size,size), :]
		batch_A_true = A_true[i:min(i+batch_size,size), :]
		with torch.no_grad():
			batch_A_predicted = model.forward(batch_data)
		error += torch.sum((batch_A_predicted-batch_A_true)**2)/size
		actual += torch.sum((batch_A_true)**2)/size
		
	return error/actual


def dataloader(config, device):
	""" N_u = training data / boundary points for data driven training
	N_f = collocation points for differential for differential driven training
	"""

	N_u = config['num_datadriven']
	N_f = config['num_collocation']

	data = np.genfromtxt(config['datadir'] + config['datafile'], delimiter=',')
	data = np.array(data, dtype=np.float32)

	T1T2range = data[range(0,data.shape[0],config['TRAIN_ITERATIONS'])][:,[1,2]]
	trange = data[:config['TRAIN_ITERATIONS'], [0]]
	AAT_true = data

	lb = np.min(data[:,:3], axis=0)
	ub = np.max(data[:,:3], axis=0)

	idx_train = np.random.choice(AAT_true.shape[0], N_u, replace=False)
	AAT_u_train = AAT_true[idx_train, :3]
	A_u_train = AAT_true[idx_train, 3:]

	idx_basecases = range(0, data.shape[0], config['TRAIN_ITERATIONS'])

	if N_f > 0:
		AAT_f_train = lb + (ub-lb)*lhs(3, N_f)		# 3 for 3 dimensional input data
		AAT_f_train = np.vstack((AAT_f_train, AAT_u_train))     # Taken all basecase points in collocation as well
		AAT_f_train = np.vstack([AAT_f_train, AAT_true[idx_basecases][:,:3]])
	else:
		AAT_f_train = AAT_u_train

	""" Adding noise if taking internal points """
	if not config['training_is_border']:
		A_u_train = A_u_train + config['noise'] * np.std(A_u_train) * np.random.randn(A_u_train.shape[0], A_u_train.shape[1])

	AAT_u_train = torch.from_numpy(AAT_u_train).float().to(device)
	A_u_train = torch.from_numpy(A_u_train).float().to(device)
	AAT_f_train = torch.from_numpy(AAT_f_train).float().to(device)
	lb = torch.from_numpy(lb).float().to(device)
	ub = torch.from_numpy(ub).float().to(device)

	global AAT_test, A_test, AAT_validation, A_validation
	AAT_test = np.delete(AAT_true[:, :3], idx_train, axis=0)
	A_test = np.delete(AAT_true[:, 3:], idx_train, axis=0)

	# Takes validation and training in 1:1 ratio
	idx_validation = np.random.choice(AAT_test.shape[0], N_u, replace=False)
	AAT_validation = AAT_test[idx_validation, :]
	A_validation = A_test[idx_validation, :]
	AAT_test = np.delete(AAT_test, idx_validation, axis=0)
	A_test = np.delete(A_test, idx_validation, axis=0)

	# Convert all to tensors
	AAT_validation = torch.from_numpy(AAT_validation).float().to(device)
	A_validation = torch.from_numpy(A_validation).float().to(device)
	AAT_test = torch.from_numpy(AAT_test).float().to(device)
	A_test = torch.from_numpy(A_test).float().to(device)

	# global test_dataset, validation_dataset, test_dataloader, validation_dataloader
	# test_dataset = AATDataset(AAT_test, A_test)
	# validation_dataset = AATDataset(AAT_validation, A_validation)
	# test_dataloader = DataLoader(test_dataset, batch_size=mcl.batch_size, shuffle=True, drop_last=True)
	# validation_dataloader = DataLoader(validation_dataset, batch_size=mcl.batch_size, shuffle=True, drop_last=True)

	return AAT_u_train, A_u_train, AAT_f_train, lb, ub

def testloader(config, testfile, model):
	device = torch.device('cuda' if torch.cuda.is_available() and config['CUDA_ENABLED'] else 'cpu')

	data = np.genfromtxt(testfile, delimiter=',')
	data = np.array(data, dtype=np.float32)

	T1T2range = data[range(0,data.shape[0],config['TRAIN_ITERATIONS']),[1,2]]
	trange = data[:config['TRAIN_ITERATIONS'], [0]]
	AAT_true = data

	AAT_test = data[:,:3]
	A_test = data[:,3:]

	# Convert all to tensors
	AAT_test = torch.from_numpy(AAT_test).float().to(device)
	A_test = torch.from_numpy(A_test).float().to(device)
	model.to(device)

	return set_loss(model, device, config['BATCH_SIZE'], AAT_test, A_test)

if __name__=="__main__":
  print("Please call this from pidnn file.")