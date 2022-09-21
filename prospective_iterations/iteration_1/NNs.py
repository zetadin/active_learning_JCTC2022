import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset
import numpy as np
import scipy as sp
#import os
from utils import get_FPR_TPR_AUC, get_fixed_ROC_AUC, get_FPR_and_TPR
import warnings
from torch import profiler
from torch.profiler import record_function


class TrainingRecord:
    def __init__(self):
        self.epoch=[]

        self.loss_Train=[]
        self.loss_XVal=[]
        self.loss_Val=[]

        self.RMSD_Train=[]
        self.RMSD_XVal=[]
        self.RMSD_Val=[]

        self.Cor_Train=[]
        self.Cor_XVal=[]
        self.Cor_Val=[]

        self.TP_Train=[]
        self.TP_XVal=[]
        self.TP_Val=[]

        self.FP_Train=[]
        self.FP_XVal=[]
        self.FP_Val=[]

        self.AUC_Train=[]
        self.AUC_XVal=[]
        self.AUC_Val=[]
        
        self.KT_Train=[]
        self.KT_XVal=[]
        self.KT_Val=[]


from enum import Enum
#class Reg_method(Enum):
    #no=0
    #l1=1
    #l2=2
    #dropout=3

    #def __int__(self):
        #return self.value

class Net(nn.Module):
    def __init__(self, inp_width, hl_w=15, nhl=2, learning_rate=1e-3,
                 activation="relu", drop_p=np.array([0,0]), #regmet=Reg_method.no,
                 lr_decay=0, weights_distrib_func=None,
                 high_binder_cutoff=0, weight_decay=0, noise=0,
                 shiftY=False):
        super(Net, self).__init__()
        self.device = torch.device("cpu") #assume initial allocation on cpu
        self.nhl=nhl
        self.hl_w=hl_w
        self.layers=[]
        self.dropouts=[]
        #self.regmet=regmet
        self.init_lr=learning_rate
        self.lr_decay=lr_decay
        self.inp_width=inp_width
        self.weights_distrib_func=weights_distrib_func
        self.high_binder_cutoff=high_binder_cutoff
        self.noise=noise
        self.shiftY=shiftY
        self.use_dropout=False
        if(np.any(drop_p>0)):
            self.use_dropout=True
            self.drop_p=drop_p
        
        self.init_layers()

        self.act_func_name=activation
        if(self.act_func_name=="relu"):
            self.act_func=F.relu
        elif(self.act_func_name=="tanh"):
            self.act_func=torch.nn.Tanh()
        elif(self.act_func_name=="celu"):
            self.act_func=F.celu
        elif(self.act_func_name=="gelu"):
            self.act_func=F.gelu
        elif(self.act_func_name=="gaussian"):
            self.act_func=lambda x: torch.exp(torch.neg(x**2))

        self.loss_mae = nn.L1Loss() #MAE
        self.loss_mse = nn.MSELoss() #MSE
        self.loss_fn = nn.L1Loss() #MAE
        self.loss_fn_weighted = self.weighted_MAE #MAE
#         self.loss_fn = nn.MSELoss() #MSE
#         self.loss_fn = lambda x,y: self.loss_mae(x,y) + 5*torch.sum((torch.log(x+0.1)-torch.log(y+0.01))**2)
#         self.loss_fn = lambda x,y: self.loss_mae(x,y) + 5*torch.mean(torch.log((x-y)**2 + 1.0))
        self.epoch=0
        self.record=TrainingRecord()
        self.best_state=self.state_dict()
        self.best_loss=None #max int value #2147483647
        self.best_epoch=0

        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        #self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)

        self.traing_generator=None
        self.crossvalidation_generator=None
        self.test_generator=None

        self.cache_state()
        
    #init dimentions and names of layers    
    def init_layers(self):
        if(self.nhl==0):
            if(self.use_dropout):
                self.dropouts.append(nn.Dropout(p=self.drop_p[0])) #input dropout
            self.layers.append(nn.Linear(self.inp_width, 1))
        else:
            if(self.use_dropout):
                self.dropouts.append(nn.Dropout(p=self.drop_p[0])) #input dropout
            self.layers.append(nn.Linear(self.inp_width, self.hl_w))
            for i in range(self.nhl):
                if(i<self.nhl-1):
                    if(self.use_dropout):
                        self.dropouts.append(nn.Dropout(p=self.drop_p[1])) #dropout between layers
                    self.layers.append(nn.Linear(self.hl_w, self.hl_w))
                else:
                    if(self.use_dropout):
                        self.dropouts.append(nn.Dropout(p=self.drop_p[1])) #dropout between layers
                    self.layers.append(nn.Linear(self.hl_w, 1))
                    
        for i in range(len(self.layers)):
            #assign references to class attributes so printing the class object shows them
            setattr(self, "layer_{}".format(i), self.layers[i])
            if(self.use_dropout):
                setattr(self, "dropout_{}".format(i), self.dropouts[i])
            #custom init values
            nn.init.xavier_uniform_(self.layers[i].weight, gain=nn.init.calculate_gain('relu'))
        

    #custom loss functions
    def weighted_MSE(self, output, target, weights):
        loss = torch.mean(weights*(output - target)**2)
        return loss

    def weighted_MAE(self, output, target, weights):
        loss = torch.mean(weights*torch.abs(output - target))
        return loss

    #store device locally in the class
    def to(self, dev):
        self.device=dev
        return(super().to(dev))

    def forward(self, x):
#         x=self.act_func(x)
        for i in range(len(self.layers)):
            if(self.use_dropout):
                x=self.dropouts[i](x)
            x=self.layers[i](x)
            if(i<len(self.layers)-1):
                x=self.act_func(x)
        if(self.shiftY):
            if(type(self.shiftY) is tuple):
                x*=self.shiftY[0] # range
                x+=self.shiftY[1] # mean
            else:
                #x+=self.high_binder_cutoff
                x*=1.535
                x+=-9.512
        return x

    def feed_training_batch(self, batch_X, batch_Y):
        def closure():
            self.optimizer.zero_grad()
            if(self.noise>0.):
                batch_new_X=batch_X+torch.randn(batch_X.size()).to(self.device) * self.noise
                output = self(batch_new_X)
            else:
                output = self(batch_X)

            if(not self.weights_distrib_func is None):
                npY=batch_Y.cpu().numpy()
                weights=self.weights_distrib_func(npY[:,0])
                weights=torch.from_numpy(weights.reshape(npY.shape)).to(self.device)
                loss = self.loss_fn_weighted(output, batch_Y, weights)
            else:
                loss = self.loss_fn(output, batch_Y)



            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)

    def get_predictions_from_batches(self, gen, with_np=False):
        P_tensor=torch.zeros([len(gen.dataset), 1], dtype=torch.float, device=self.device)
        Y_tensor=torch.zeros([len(gen.dataset), 1], dtype=torch.float, device=self.device)
        if(with_np):
            P_np=np.zeros((len(gen.dataset)))
            Y_np=np.zeros((len(gen.dataset)))

        #fill arrays
        start=0
        for batch_X, batch_Y in gen:
            batchsize = batch_X.shape[0]
            end = start + batchsize

            # Store Y before transfer to GPU
            if(with_np):
                Y_np[start:end] = batch_Y.numpy()[:,0] #still on CPU

            # Transfer to GPU
            batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)

            # Evaluate
            with torch.no_grad():
                batch_P = self.eval()(batch_X)
            if(batch_P.size() != batch_Y.size()):
                raise(Exception(f"Shapes of predicted and target tensors do not match: batch_P is {batch_P.size()}, batch_Y is {batch_Y.size()}, "))

            # Store the rest of the arrays
            batch_P = batch_P.detach()
            P_tensor[start:end,:] = batch_P
            Y_tensor[start:end,:] = batch_Y
            if(with_np):
                P_np[start:end] = batch_P.cpu().numpy()[:,0]

            #update indeces in arrays
            start = end

        if(with_np):
            return(P_tensor, Y_tensor, P_np, Y_np)
        else:
            return(P_tensor, Y_tensor)

    def evaluate_fit(self):
        with profiler.record_function("evaluate_fit"):

            #fill training arrays
            P_dtrain, Y_dtrain, np_P_dtrain, np_Y_dtrain = self.get_predictions_from_batches(self.training_generator_no_shuffle, with_np=True)
    #         print(np_P_dtrain)
    #         print(np_Y_dtrain)

            #fill XVal arrays
            if(self.crossvalidation_generator is not None):
                P_d_XVal, Y_d_XVal, np_P_d_XVal, np_Y_d_XVal = self.get_predictions_from_batches(self.crossvalidation_generator, with_np=True)
    #         print(np_P_d_XVal)
    #         print(np_Y_d_XVal)



            #compute metrics
            loss = self.loss_fn(P_dtrain, Y_dtrain).detach().tolist()
            if(self.crossvalidation_generator is not None):
                XVal_loss=self.loss_fn(P_d_XVal, Y_d_XVal).detach().tolist()

            self.record.epoch.append(self.epoch)
            self.record.loss_Train.append(loss)     #already detached
            self.record.RMSD_Train.append(torch.sqrt(self.loss_mse(Y_dtrain, P_dtrain)).detach().tolist())
            if(self.crossvalidation_generator is not None):
                self.record.loss_XVal.append(XVal_loss) #already detached
                self.record.RMSD_XVal.append(torch.sqrt(self.loss_mse(Y_d_XVal, P_d_XVal)).detach().tolist())
            try:
                self.record.Cor_Train.append(np.corrcoef(np_Y_dtrain, np_P_dtrain)[0,1])
                self.record.KT_Train.append(sp.stats.kendalltau(np_Y_dtrain.flatten(), np_P_dtrain.flatten())[0])
                if(self.crossvalidation_generator is not None):
                    self.record.Cor_XVal.append(np.corrcoef(np_Y_d_XVal, np_P_d_XVal)[0,1])
                    self.record.KT_XVal.append(sp.stats.kendalltau(np_Y_d_XVal.flatten(), np_P_d_XVal.flatten())[0])
                
            except Exception as e:
                #print(np_Y_dtrain)
                #print(np_P_dtrain)
                #print(np.corrcoef(np_Y_dtrain, np_P_dtrain))
                #print(np_Y_d_XVal)
                #print(np_P_d_XVal)
                #print(np.corrcoef(np_Y_d_XVal, np_P_d_XVal))
                raise(e)

    #         raise()

            #true and false positive rates + AUC
                #XVal
            if(self.crossvalidation_generator is not None):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    #FP, TP, AUC = get_FPR_TPR_AUC(np_Y_d_XVal, np_P_d_XVal, self.high_binder_cutoff)
                    FP, TP = get_FPR_and_TPR(np_Y_d_XVal, np_P_d_XVal, self.high_binder_cutoff)
                    AUC = get_fixed_ROC_AUC(np_Y_d_XVal, np_P_d_XVal, P_err=None, cut=self.high_binder_cutoff)
                self.record.FP_XVal.append(FP)
                self.record.TP_XVal.append(TP)
                self.record.AUC_XVal.append(AUC)

                #Train
            #FP, TP, AUC = get_FPR_TPR_AUC(np_Y_dtrain, np_P_dtrain, self.high_binder_cutoff)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                FP, TP = get_FPR_and_TPR(np_Y_dtrain, np_P_dtrain, self.high_binder_cutoff)
                AUC = get_fixed_ROC_AUC(np_Y_dtrain, np_P_dtrain, P_err=None, cut=self.high_binder_cutoff)
            self.record.FP_Train.append(FP)
            self.record.TP_Train.append(TP)
            self.record.AUC_Train.append(AUC)

            #validation, if available
            if(not self.test_generator is None):
                #fill validation arrays
                P_d_Val, Y_d_Val, np_P_d_Val, np_Y_d_Val = self.get_predictions_from_batches(self.test_generator, with_np=True)

                #calculate metrics
                Val_loss=self.loss_fn(Y_d_Val, P_d_Val).detach().tolist()
                Val_RMSD=torch.sqrt(self.loss_mse(Y_d_Val, P_d_Val)).detach().tolist()
                Val_Cor =np.corrcoef(np_Y_d_Val, np_P_d_Val)[0,1]

                self.record.loss_Val.append(Val_loss)
                self.record.RMSD_Val.append(Val_RMSD)
                self.record.Cor_Val.append(Val_Cor)
                self.record.KT_Val.append(sp.stats.kendalltau(np_Y_d_Val.flatten(), np_P_d_Val.flatten())[0])

                #true and false positive rates + AUC
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    #FP, TP, AUC = get_FPR_TPR_AUC(np_Y_d_Val, np_P_d_Val, self.high_binder_cutoff)
                    FP, TP = get_FPR_and_TPR(np_Y_d_Val, np_P_d_Val, self.high_binder_cutoff)
                    AUC = get_fixed_ROC_AUC(np_Y_d_Val, np_P_d_Val, P_err=None, cut=self.high_binder_cutoff)
                self.record.FP_Val.append(FP)
                self.record.TP_Val.append(TP)
                self.record.AUC_Val.append(AUC)

    #         if(self.best_loss is None or XVal_loss<self.best_loss):
    #             self.best_state=self.state_dict()
    #             self.best_loss=XVal_loss
    #             self.best_epoch=self.epoch

    def train_epoch(self, idxs=None, eval_every=20):
        with profiler.record_function("train_epoch"):
            self.adjust_learning_rate()

            for batch_X, batch_Y in self.training_generator:
                # Transfer to GPU
                batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
                self.feed_training_batch(batch_X, batch_Y)

            if(eval_every>0 and (self.epoch % eval_every == eval_every-1 or self.epoch == 0)):
                self.evaluate_fit()

            self.epoch+=1

    def cache_state(self):
        self.saved_state=(copy.deepcopy(self.state_dict()),
                          copy.deepcopy(self.optimizer.state_dict()))

    def restore_state(self):
        if(self.saved_state is not None):
            backup=(copy.deepcopy(self.state_dict()),
                    copy.deepcopy(self.optimizer.state_dict()))
            self.load_state_dict(self.saved_state[0])
            self.optimizer.load_state_dict(self.saved_state[1])
            self.saved_state=backup
        else:
            raise()

    def adjust_learning_rate(self):
        """Reduce the learning rate by order of magnitude every self.lr_decay epochs."""
        if(type(self.lr_decay) is tuple): # stochastic gradient descent with warm restarts
            #lr_decay_const=lr_decay[0]
            #lr_min=self.init_lr * (0.1 ** (20000 / lr_decay_const))
            lr_min=self.lr_decay[0]
            lr_max=self.init_lr
            lr_decay_period=self.lr_decay[1]
            
            lr=lr_min + 0.5*(lr_max-lr_min)*(1+np.cos(np.pi*(self.epoch%lr_decay_period)/lr_decay_period))
            
            #print(f"Epoch: {self.epoch} \tlearning rate: {lr}")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif(self.lr_decay>0):
            lr = self.init_lr * (0.1 ** (self.epoch / self.lr_decay))
            
            #print(f"Epoch: {self.epoch} \tlearning rate: {lr}")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr









class Net_classifier(Net):
    def __init__(self, inp_width, hl_w=15, nhl=2, learning_rate=1e-3,
                activation="relu", drop_p=np.array([0,0]), #regmet=Reg_method.no,
                lr_decay=0, weights_distrib=None,
                high_binder_cutoff=0, weight_decay=0):
        super(Net, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
