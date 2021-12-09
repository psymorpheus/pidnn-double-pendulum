import torch
from torch.autograd import grad
import torch.nn as nn

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import dp_dataloader as dpdl

optimizer = None

class PINN(nn.Module):
    
    def __init__(self, id, AAT_u, A_u, AAT_f, layers, lb, ub, device, config, alpha, beta, N_u, N_f):
        super().__init__()
        
        self.id = id
        self.device = device

        self.u_b = ub
        self.l_b = lb
        self.alpha = alpha
        self.beta = beta
        self.N_u = N_u
        self.N_f = N_f
        self.batch_size = config['BATCH_SIZE']
        self.config = config
       
        # AAT = angle, angle, time
        self.AAT_u = AAT_u
        self.A_u = A_u
        self.AAT_f = AAT_f
        self.layers = layers

        self.f_hat = torch.zeros(AAT_f.shape[0]).to(device)

        self.activation = nn.Tanh()
        # self.activation = nn.ReLU()
        self.loss_function = nn.MSELoss(reduction ='mean') # removing for being able to batch 
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0
        self.loss_u = None
        self.loss_f = None
        self.elapsed = None

        self.iter_history = []
        self.history = None # train, loss_u, loss_f, validation

        for i in range(len(layers)-1):
            # Recommended gain value for tanh = 5/3? TODO - done
            # nn.init.xavier_normal_(self.linears[i].weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(self.linears[i].bias.data)
    
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
            mse += torch.sum((batch_err)**2, dim=0)/size

        return mse
    
    def loss_BC(self,x,y):
        """ Loss at boundary and initial conditions """
        prediction = self.forward(x)
        error = prediction - y
        # loss_u = self.batched_mse(error)
        loss_u = self.loss_function(y, prediction)

        # loss_u = self.loss_function(self.forward(x), y)
        return loss_u

    def loss_PDE(self, AAT_f_train):
        """ Loss at collocation points, calculated from Partial Differential Equation
        Note: x_v = x_v_t[:,[0]], x_t = x_v_t[:,[1]]
        """
                        
        # g = AAT_f_train.clone()
        # g.requires_grad = True
        # u = self.forward(g)
        
        # T1 is theta1, T2 is theta2

        # x_v_t = autograd.grad(u,g,torch.ones([AAT_f_train.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        # x_vv_tt = autograd.grad(x_v_t,g,torch.ones(AAT_f_train.shape).to(self.device), retain_graph=True, create_graph=True)[0]
        # x_vvv_ttt = autograd.grad(x_vv_tt,g,torch.ones(AAT_f_train.shape).to(self.device), create_graph=True)[0]
        # x_t = x_v_t[:,[1]]
        # x_tt = x_vv_tt[:,[1]]
        # x_ttt = x_vvv_ttt[:,[1]]
        # f = x_ttt

        # T1_T1T2t = autograd.grad(u[0],g,torch.ones([AAT_f_train.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        # T2_T1T2t = autograd.grad(u[1],g,torch.ones([AAT_f_train.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        # TT_T1T2t = autograd.grad(
        # 	outputs=u,
        # 	inputs=g,
        # 	grad_outputs=torch.ones([AAT_f_train.shape[0], 2]).to(self.device),
        # 	retain_graph=True,
        # 	create_graph=True
        # )[0]

        # ####################### M2
        # T1_T1T2t = autograd.grad(
        # 	outputs=u[:,[0]],
        # 	inputs=g,
        # 	grad_outputs=torch.ones([AAT_f_train.shape[0], 1]).to(self.device),
        # 	retain_graph=True,
        # 	create_graph=True
        # )[0]
        # T1_T1T2t_T1T2t = autograd.grad(
        # 	outputs=T1_T1T2t,
        # 	inputs=g,
        # 	grad_outputs=torch.ones(AAT_f_train.shape).to(self.device),
        # 	create_graph=True
        # )[0]

        # g = AAT_f_train.clone()  
        # g.requires_grad = True
        # u = self.forward(g)
        # T2_T1T2t = autograd.grad(
        # 	outputs=u[:,[1]],
        # 	inputs=g,
        # 	grad_outputs=torch.ones([AAT_f_train.shape[0], 1]).to(self.device),
        # 	retain_graph=True,
        # 	create_graph=True
        # )[0]
        
        # T2_T1T2t_T1T2t = autograd.grad(
        # 	outputs=T2_T1T2t,
        # 	inputs=g,
        # 	grad_outputs=torch.ones(AAT_f_train.shape).to(self.device),
        # 	create_graph=True
        # )[0]
        
        x = AAT_f_train.clone()
        x.requires_grad = True
        pred = self.forward(x)
        T1 = pred[:, 0]
        T2 = pred[:, 1]
        
        grads = torch.ones(T1.shape, device=pred.device) # move to the same device as prediction
        grad_T1 = grad(T1, x, create_graph=True, grad_outputs=grads)[0]
        grad_T2 = grad(T2, x, create_graph=True, grad_outputs=grads)[0]

        # calculate first order derivatives
        T1_T1 = grad_T1[:, 0]
        T1_T2 = grad_T1[:, 1]
        T1_t = grad_T1[:, 2]

        T2_T1 = grad_T2[:, 0]
        T2_T2 = grad_T2[:, 1]
        T2_t = grad_T2[:, 2]

        # calculate second order derivatives
        grad_T1_t = grad(T1_t, x, create_graph=True, grad_outputs=grads)[0]
        grad_T2_t = grad(T2_t, x, create_graph=True, grad_outputs=grads)[0]

        T1_tt = grad_T1_t[:, 2]
        T2_tt = grad_T2_t[:, 2]
        # f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        # f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

        # T1 = u[0]
        # T2 = u[1]
        # T1_t = T1_T1T2t[:,[2]]
        # T2_t = T2_T1T2t[:,[2]]
        # T1_tt = T1_T1T2t_T1T2t[:,[2]]
        # T2_tt = T2_T1T2t_T1T2t[:,[2]]

        T1 += x[:, 1]
        T1 += x[:, 2]

        f1 = self.config['l1'] * (self.config['m1']+self.config['m2']) * T1_tt \
            + self.config['l2'] * self.config['m2'] * torch.sin(T1 - T2) * (T2_t ** 2) \
            + self.config['l2'] * self.config['m2'] * torch.cos(T1 - T2) * T2_tt \
            + self.config['g'] * (self.config['m1']+self.config['m2']) * torch.sin(T1)
        
        f2 = self.config['l1'] * torch.sin(T1 - T2) * (T1_t ** 2) \
            - self.config['l1'] * torch.cos(T1 - T2) * T1_tt \
            - self.config['l2'] * T2_tt \
            - self.config['g'] * torch.sin(T2)
        

        # x_vv_tt = autograd.grad(x_v_t,g,torch.ones(AAT_f_train.shape).to(self.device), retain_graph=True, create_graph=True)[0]
        # x_vvv_ttt = autograd.grad(x_vv_tt,g,torch.ones(AAT_f_train.shape).to(self.device), create_graph=True)[0]
        # x_t = x_v_t[:,[1]]
        # x_tt = x_vv_tt[:,[1]]
        # x_ttt = x_vvv_ttt[:,[1]]
        # f = x_ttt

        # Use alpha beta here itself
        # loss_f1 = self.batched_mse(f1)
        # loss_f2 = self.batched_mse(f2)
        loss_f1 = self.loss_function(self.f_hat, f1)
        loss_f2 = self.loss_function(self.f_hat, f2)
        # if self.iter<1000:
        #     loss_f = 0*loss_f1 + 0*loss_f2
        # else:
        #     loss_f = self.alpha*loss_f1 + self.beta*loss_f2
        loss_f = self.alpha*loss_f1 + self.beta*loss_f2
        return loss_f

    def loss(self,AAT_u_train,A_u_train,AAT_f_train):

        self.loss_u = self.loss_BC(AAT_u_train,A_u_train)
        if self.config['take_differential_points']:
            self.loss_f = self.loss_PDE(AAT_f_train)
        else:
            self.loss_f = torch.tensor(0)
        loss_val = self.loss_u + self.loss_f
        
        return loss_val

    def closure(self):
        """ Called multiple times by optimizers like Conjugate Gradient and LBFGS.
        Clears gradients, compute and return the loss.
        """
        optimizer.zero_grad()
        loss = self.loss(self.AAT_u, self.A_u, self.AAT_f)
        loss.backward()		# To get gradients

        self.iter += 1

        if self.iter % 100 == 0:
            training_loss = loss.item()
            validation_loss = dpdl.set_loss(self, self.device, self.batch_size).item()
            # training_history[self.id].append([self.iter, training_loss, validation_loss])
            print(
                'Iter %d, Training: %.5e, Data loss: %.5e, Collocation loss: %.5e, Validation: %.5e' % (self.iter, training_loss, self.loss_u, self.loss_f, validation_loss)
            )
            self.iter_history.append(self.iter)
            current_history = np.array([training_loss, self.loss_u.item(), self.loss_f.item(), validation_loss])
            if self.history is None: self.history = current_history
            else: self.history = np.vstack([self.history, current_history])

        return loss
    
    def plot_history(self, debug=True):
        """ Saves training (loss_u + loss_f and both separately) and validation losses
        """
        epochs = self.iter_history
        loss = {}
        loss['Training'] = np.ndarray.tolist(self.history[:,0].ravel())
        loss['Data'] = np.ndarray.tolist(self.history[:,1].ravel())
        loss['Collocation'] = np.ndarray.tolist(self.history[:,2].ravel())
        loss['Validation'] = np.ndarray.tolist(self.history[:,3].ravel())
        last_training_loss = loss['Training'][-1]
        last_validation_loss = loss['Validation'][-1]

        for loss_type in loss.keys():
            plt.clf()
            plt.plot(epochs, loss[loss_type], color = (63/255, 97/255, 143/255), label=f'{loss_type} loss')
            if (loss_type == 'Validation'): title = f'{loss_type} loss (Relative MSE)\n'
            else : title = f'{loss_type} loss (MSE)\n'

            plt.title(
                title + 
                f'Elapsed: {self.elapsed:.2f}, Alpha: {self.alpha}, N_u: {self.N_u}, N_f: {self.N_f},\n Validation: {last_validation_loss:.2f}, Train: {last_training_loss:.2f}'
            )
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            savefile_name = ''
            if debug: savefile_name += 'Debug_'
            savefile_name += 'plot_' + self.config['model_name']
            if debug: savefile_name += '_' + str(self.N_f) + '_' + str(self.alpha)
            savefile_name += '_' + loss_type
            savefile_name += '.png'
            savedir = self.config['modeldir']
            if debug: savedir += self.config['model_name'] + '/'

            if self.config['SAVE_PLOT']: plt.savefig(savedir + savefile_name)

def pidnn_driver(config):
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

    # 3 inputs, 2 outputs
    layers = np.concatenate([[3], num_neurons*np.ones(num_layers), [2]]).astype(int).tolist()

    models = []
    validation_losses = []
    
    for i in range(len(config['collocation_multiplier'])):
        N_u = config['num_datadriven']
        N_f = N_u * config['collocation_multiplier'][i]
        AAT_u_train, A_u_train, AAT_f_train, lb, ub = dpdl.dataloader(config,device)
        if not config['take_differential_points']:
            num_alpha = 1
            num_beta = 1
        else:
            num_alpha = len(config['ALPHA'])
            num_beta = len(config['BETA'])

        for i in range(num_alpha):
            for j in range(num_beta):
                alpha = config['ALPHA'][i]
                beta = config['BETA'][j]

                print(f'++++++++++ N_u:{N_u}, N_f:{N_f}, Alpha:{alpha}, Beta:{beta} ++++++++++')

                model = PINN((i,j), AAT_u_train, A_u_train, AAT_f_train, layers, lb, ub, device, config, alpha, beta, N_u, N_f)
                model.to(device)
                # print(model)

                # L-BFGS Optimizer
                global optimizer
                optimizer = torch.optim.LBFGS(
                    model.parameters(), lr=0.01, 
                    max_iter = config['EARLY_STOPPING'],
                    tolerance_grad = 1.0 * np.finfo(float).eps, 
                    tolerance_change = 1.0 * np.finfo(float).eps, 
                    history_size = 100
                )
                # optimizer = torch.optim.Adam(
                #     model.parameters(),
                #     lr=0.01
                # )
                # prev_loss = -np.inf
                # current_loss = np.inf
                
                # start_time = time.time()

                # while abs(current_loss-prev_loss)>np.finfo(float).eps and model.iter<config['EARLY_STOPPING']:
                #     current_loss, prev_loss = model.closure(), current_loss
                #     optimizer.step()
                
                start_time = time.time()
                optimizer.step(model.closure)		# Does not need any loop like Adam
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

    all_hyperparameter_models = [[models[md].N_u, models[md].N_f, models[md].alpha, validation_losses[md]] for md in range(len(models))]
    all_hyperparameter_models = pd.DataFrame(all_hyperparameter_models)
    all_hyperparameter_models.to_csv(config['modeldir'] + config['model_name'] + '.csv', header=['N_u', 'N_f', 'alpha', 'Validation Error'])

    if device == 'cuda':
        torch.cuda.empty_cache()

# if __name__ == "__main__": 
# 	main_loop(config['num_datadriven'], config['num_collocation'], config['num_layers'], config['neurons_per_layer'], config['num_validation'])
