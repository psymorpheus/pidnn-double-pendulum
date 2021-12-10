from numpy.lib.npyio import save
import torch
import torch.autograd as autograd
import torch.nn as nn

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import dp_dataloader as dpdl

optimizer = None

class FF_Baseline(nn.Module):
	
	def __init__(self, VT_u, X_u, layers, lb, ub, device, config, N_u):
		super().__init__()
		
		self.id = id
		self.device = device

		self.u_b = ub
		self.l_b = lb
		self.N_u = N_u
		self.batch_size = config['BATCH_SIZE']
		self.config = config
	   
		self.VT_u = VT_u
		self.X_u = X_u
		self.layers = layers

		self.activation = nn.ReLU()
		self.builtin_loss_function = nn.MSELoss(reduction ='mean') # removing for being able to batch 
		self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
		self.iter = 0
		self.elapsed = None

		self.iter_history = []
		self.history = None # train, validation

		for i in range(len(layers)-1):
			nn.init.xavier_normal_(self.linears[i].weight.data, gain=nn.init.calculate_gain('relu'))
			nn.init.zeros_(self.linears[i].bias.data)
	
	def loss_function(self, prediction, actual):
		del_x1 = self.config['l1'] * (torch.sin(prediction[:,[0]]) - torch.sin(actual[:,[0]]))
		del_y1 = - self.config['l1'] * (torch.cos(prediction[:,[0]]) - torch.cos(actual[:,[0]]))
		del_x2 = self.config['l1'] * (torch.sin(prediction[:,[0]]) - torch.sin(actual[:,[0]])) \
				+ self.config['l2'] * (torch.sin(prediction[:,[1]]) - torch.sin(actual[:,[1]]))
		del_y2 = - self.config['l1'] * (torch.cos(prediction[:,[0]]) - torch.cos(actual[:,[0]])) \
				- self.config['l2'] * (torch.cos(prediction[:,[1]]) - torch.cos(actual[:,[1]]))
		square_error = del_x1**2 + del_y1**2 + del_x2**2 + del_y2**2
		mse = torch.sum(square_error)/square_error.shape[0]
		return mse

	def forward(self,x):
		if torch.is_tensor(x) != True:
			x = torch.from_numpy(x)    
					  
		# Preprocessing input - sclaed from 0 to 1
		x = (x - self.l_b)/(self.u_b - self.l_b)
		a = x.float()
		
		for i in range(len(self.layers)-2):
			z = self.linears[i](a)
			a = self.activation(z)

		# Activation is not applied to last layer
		a = self.linears[-1](a)
		
		return a

	def batched_mse(self, err):
		size = err.shape[0]
		if size<1000: batch_size = size
		else: batch_size = self.batch_size
		mse = 0
		
		for i in range(0, size, batch_size):
			batch_err = err[i:min(i+batch_size,size), :]
			mse += torch.sum((batch_err)**2)/size

		return mse
	
	def loss(self, x, y):
		prediction = self.forward(x)
		error = prediction - y
		# loss_u = self.batched_mse(error)
		loss_u = self.loss_function(prediction, y)

		# loss_u = self.loss_function(self.forward(x), y)
		return loss_u

	def closure(self):
		""" Called multiple times by optimizers like Conjugate Gradient and LBFGS.
		Clears gradients, compute and return the loss.
		"""
		optimizer.zero_grad()
		loss = self.loss(self.VT_u, self.X_u)
		loss.backward()		# To get gradients

		self.iter += 1

		if self.iter % 100 == 0:
			training_loss = loss.item()
			validation_loss = dpdl.set_loss(self, self.device, self.batch_size).item()
			print(
				'Iter %d, Training: %.5e, Validation: %.5e' % (self.iter, training_loss, validation_loss)
			)
			self.iter_history.append(self.iter)
			current_history = np.array([training_loss, validation_loss])
			if self.history is None: self.history = current_history
			else: self.history = np.vstack([self.history, current_history])

		return loss
	
	def plot_history(self, debug=True):
		""" Saves training (loss_u + loss_f and both separately) and validation losses
		"""
		loss = {}
		if self.history is not None:
			epochs = self.iter_history
			loss['Training'] = np.ndarray.tolist(self.history[:,0].ravel())
			loss['Validation'] = np.ndarray.tolist(self.history[:,1].ravel())
		else:
			epochs = [self.iter]
			loss['Training'] = [1e6]
			loss['Validation'] = [1e6]
		last_training_loss = loss['Training'][-1]
		last_validation_loss = loss['Validation'][-1]

		for loss_type in loss.keys():
			plt.clf()
			plt.plot(epochs, loss[loss_type], color = (63/255, 97/255, 143/255), label=f'{loss_type} loss')
			if (loss_type == 'Validation'): title = f'{loss_type} loss (Relative MSE)\n'
			else : title = f'{loss_type} loss (MSE)\n'

			plt.title(
				title + 
				f'Elapsed: {self.elapsed:.2f}, N_u: {self.N_u},\n Validation: {last_validation_loss:.2f}, Train: {last_training_loss:.2f}'
			)
			plt.xlabel('Epochs')
			plt.ylabel('Loss')
			plt.legend()
			savefile_name = ''
			if debug: savefile_name += 'Debug_'
			savefile_name += 'plot_' + self.config['model_name']
			# if debug: savefile_name += '_' + str(self.N_f) + '_' + str(self.alpha)
			savefile_name += '_' + loss_type
			savefile_name += '.png'
			savedir = self.config['modeldir']
			if debug: savedir += self.config['model_name'] + '/'

			if self.config['SAVE_PLOT']: plt.savefig(savedir + savefile_name)
			plt.close()

def ff_driver(config):
	plt.figure(figsize=(8, 6), dpi=80)
	num_layers = config['num_layers']
	num_neurons = config['neurons_per_layer']

	torch.set_default_dtype(torch.float)
	torch.manual_seed(1234)
	np.random.seed(1234)
	if config['ANOMALY_DETECTION']:
		torch.autograd.set_detect_anomaly(True)
	else:
		torch.autograd.set_detect_anomaly(False)
		torch.autograd.profiler.profile(False)
		torch.autograd.profiler.emit_nvtx(False)

	device = torch.device('cuda' if torch.cuda.is_available() and config['CUDA_ENABLED'] else 'cpu')

	print("Running this on", device)
	if device == 'cuda': 
		print(torch.cuda.get_device_name())

	# layers is a list, not an ndarray
	layers = np.concatenate([[3], num_neurons*np.ones(num_layers), [2]]).astype(int).tolist()

	models = []
	validation_losses = []
	
	N_u = config['num_datadriven']

	VT_u_train, u_train, _, lb, ub = dpdl.dataloader(config, device)


	print(f'++++++++++ N_u:{N_u} ++++++++++')

	model = FF_Baseline(VT_u_train, u_train, layers, lb, ub, device, config, N_u)
	model.to(device)
	# print(model)

	# L-BFGS Optimizer
	global optimizer
	# optimizer = torch.optim.LBFGS(
	# 	model.parameters(), lr=0.01, 
	# 	max_iter = config['EARLY_STOPPING'],
	# 	tolerance_grad = 1.0 * np.finfo(float).eps, 
	# 	tolerance_change = 1.0 * np.finfo(float).eps, 
	# 	history_size = 100
	# )
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=0.001
	)
	prev_loss = -np.inf
	current_loss = np.inf
	
	start_time = time.time()

	# optimizer.step(model.closure)		# Does not need any loop like Adam
	while abs(current_loss-prev_loss)>np.finfo(float).eps and model.iter<config['EARLY_STOPPING']:
		current_loss, prev_loss = model.closure(), current_loss
		optimizer.step()

	elapsed = time.time() - start_time                
	print('Training time: %.2f' % (elapsed))

	validation_loss = dpdl.set_loss(model, device, config['BATCH_SIZE'])
	model.elapsed = elapsed
	model.plot_history()
	model.to('cpu')
	models.append(model)
	validation_losses.append(validation_loss.cpu().item())

	model_id = np.nanargmin(validation_losses) # choosing best model out of the bunch
	model = models[model_id]

	""" Model Accuracy """ 
	error_validation = validation_losses[model_id]
	print('Validation Error of finally selected model: %.5f'  % (error_validation))

	"""" For plotting final model train and validation errors """
	if config['SAVE_PLOT']: model.plot_history(debug=False)

	""" Saving only final model for reloading later """
	if config['SAVE_MODEL']: torch.save(model, config['modeldir'] + config['model_name'] + '.pt')

	all_hyperparameter_models = [[models[md].N_u, validation_losses[md]] for md in range(len(models))]
	all_hyperparameter_models = pd.DataFrame(all_hyperparameter_models)
	all_hyperparameter_models.to_csv(config['modeldir'] + config['model_name'] + '.csv', header=['N_u', 'Validation Error'])

	if device == 'cuda':
		torch.cuda.empty_cache()

# if __name__ == "__main__": 
# 	main_loop(config['num_datadriven'], config['num_collocation'], config['num_layers'], config['neurons_per_layer'], config['num_validation'])
