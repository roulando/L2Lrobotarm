import numpy as np
import numpy.random as rd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.framework import function
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell


from collections import namedtuple

import setup_arm
from setup_arm import Dataset

from models import QuantizationInfoTuple
from models import LIFParameterTuple
from models import epropQuanTuple
from models import LIFcell_metaeprop_trainee_LSG
from models import LIFcellWeightBatchOutput
from models import Quantize10

RegStatsTuple = namedtuple('RegStats', (
    'target_trainee_f',
    'target_lsg_f',
    'lambda_trainee',
    'lambda_lsg'
))

class L2Lsetup_doublecell():
  def __init__(self,dataset, 
                    n_trainee_rec, 
                    n_lsg_rec, 
                    lsg_binary_encoding,
                    trainee_par = LIFParameterTuple(thr=0.4, tau=20.,tau_o=20.,damp=0.3,n_ref=0),
                    lsg_par = LIFParameterTuple(thr=0.4, tau=20.,tau_o=20.,damp=0.3,n_ref=0),
                    quantize=None, #{"trainee_quan" ,"trainee_eprop","lsg_quan", "output_groups"}
                    output_groups = 16,
                    lr = 1.5e-3,
                    noise = False, 
                    inner_lr = 0.01,
                    old = False,
                    stochupdate = None,
                    norm = None,
                    reg_par = RegStatsTuple(target_trainee_f=10.,target_lsg_f=10., lambda_trainee=0.,lambda_lsg=0.)):
    

    self.norm = norm
    self.dataset = dataset
    self.optimizer = tf.keras.optimizers.Adam(lr = lr)
    self.inner_lr = inner_lr
    self.trainee_input_per_output = 1#trainee_input_per_output
    self.lsg_input_per_trainee = 1#lsg_input_per_trainee
    self.reg_par = reg_par
    self.lr_decay = 1.
    n_in = dataset.n_in
    n_out = dataset.n_out
    
    self.quantize = quantize is not None
    
    #update_time = 100
    
    #self.update_mask = tf.concat((tf.zeros((1,dataset.n_t-update_time,1)),tf.ones((1,update_time,1)) ),axis = 1)
    
    if only_output:
      if quantize is None:
        groups = None
        n_lsg_in = n_in + (n_out * lsg_binary_encoding) #+ n_trainee_rec
        n_lsg_out = n_trainee_rec
      else:
        groups = n_trainee_rec // output_groups
        n_lsg_in = n_in + (n_out * lsg_binary_encoding) #+ groups
        n_lsg_out = groups
    else:
      groups = n_trainee_rec // output_groups
      n_lsg_in = n_in + (n_out * lsg_binary_encoding) #+ groups
      n_lsg_out = n_out
    
    
    #if quantize is not None:
      #n_lsg_in = n_in + output_groups + (n_out * n_lsg_target)
    #n_lsg_in = n_in +  (n_out * lsg_binary_encoding)
    #n_lsg_out = output_groups    #output_groups
    #else:
    #  #n_lsg_in = n_in + n_trainee_rec + (n_out * n_lsg_target)
    
    self.W_in_trainee  = tf.Variable(name="InputWeight_trainee",     initial_value=(rd.randn(n_in, n_trainee_rec) / np.sqrt(n_in)).astype(np.float32),dtype='float32',trainable = True)
    self.W_rec_trainee = tf.Variable(name="RecurrentWeight_trainee", initial_value=(rd.randn(n_trainee_rec, n_trainee_rec) / np.sqrt(n_trainee_rec)).astype(np.float32),dtype='float32',trainable = True)
    self.W_out_trainee = tf.Variable(name="OutputWeight_trainee",    initial_value=(rd.randn(n_trainee_rec, n_out) / np.sqrt(n_trainee_rec)).astype(np.float32) * (1. -np.exp(-1/20)),dtype='float32',trainable = True)
    
    self.W_in_lsg  = tf.Variable(name="InputWeight_lsg",     initial_value=(rd.randn(n_lsg_in, n_lsg_rec) / np.sqrt(n_lsg_in)).astype(np.float32),dtype='float32',trainable = True)
    self.W_rec_lsg = tf.Variable(name="RecurrentWeight_lsg", initial_value=(rd.randn(n_lsg_rec, n_lsg_rec) / np.sqrt(n_lsg_rec)).astype(np.float32),dtype='float32',trainable = True)
    self.W_out_lsg = tf.Variable(name="OutputWeight_lsg",    initial_value=tf.keras.initializers.GlorotUniform()((n_lsg_rec,n_lsg_out)),dtype='float32',trainable = True)
    self.variables = [self.W_in_trainee,self.W_rec_trainee,self.W_out_trainee, self.W_out_lsg,self.W_in_lsg,self.W_rec_lsg]
    
    self.training_cell = LIFcell_metaeprop_trainee_LSG(n_in, n_trainee_rec, n_out 
                                                ,n_lsg_in, n_lsg_rec, n_lsg_out
                                                ,LIF_par_trainee = trainee_par
                                                , LIF_par_lsg = lsg_par
                                                , lr = inner_lr
                                                , old = old
                                                , groups = groups
                                                , quantize= quantize
                                                , noise = noise
                                                , stochupdate = stochupdate)
    if not quantize:
      test_cell_quan = quantize
    else:
      test_cell_quan = quantize[0]
    self.test_cell = LIFcellWeightBatchOutput(n_trainee_rec,n_out,LIF_par = trainee_par, noise = noise ,quantize=test_cell_quan )
    
    self.training_network = tf.keras.layers.RNN(self.training_cell, return_sequences=True, return_state=True) 
    self.test_network = tf.keras.layers.RNN(self.test_cell, return_sequences=True) 
    
    
    #if quantize is not None:
    #  self.trainee = RSNNModel(n_in, n_trainee_rec,q_info = quantize["trainee_quan"], LIF_par= trainee_par,batch_weight = True, output_groups = output_groups, eprop_q_info = quantize["trainee_eprop"])
    #  self.lsg = RSNNModel(n_lsg_in, n_lsg_rec,q_info = quantize["lsg_quan"], LIF_par= lsg_par ,batch_weight = False)
    #else:  
    #  self.trainee = RSNNModel(n_in, n_trainee_rec,LIF_par=trainee_par,batch_weight = True)
    #  self.lsg = RSNNModel(n_lsg_in, n_lsg_rec,LIF_par= lsg_par,batch_weight = False)
  
  def set_learningrate_decay(self, decay):
    self.lr_decay = decay
  def decrease_learning_rate(self):
    self.optimizer.lr = self.optimizer.lr * self.lr_decay
    
  def set_static_data(self):
    init_position, clock_signal, clock_signal_lsg = self.dataset.getstaticdata()
    self.init_position = init_position
    self.clock_signal = clock_signal
    self.clock_signal_lsg = clock_signal_lsg
  
  def set_weights(self,variables):
    self.W_in_trainee.assign(variables[0])
    self.W_rec_trainee.assign(variables[1])
    self.W_out_trainee.assign(variables[2])
    self.W_out_lsg.assign(variables[3])
    self.W_in_lsg.assign(variables[4])
    self.W_rec_lsg.assign(variables[5])
  
  @tf.function
  def loss(self,output, omega_t, coor_t):
    
    position = self.init_position[:, None, :] +0.01*tf.cumsum(output, axis=1) #some weird scaling ask franz -> 0.02 * 
    
    phi0 = position[..., 0]
    phi1 = position[..., 1] + phi0

    cartesian_x = (tf.cos(phi0) + tf.cos(phi1)) * .5
    cartesian_y = (tf.sin(phi0) + tf.sin(phi1)) * .5
    cartesian = tf.stack((cartesian_x, cartesian_y), -1)

    return .5*(tf.reduce_mean(tf.square(omega_t-output)))+.5*tf.reduce_mean(tf.square(coor_t-cartesian)) , cartesian

  @tf.function    
  def loss_reg(self,spikes,target_freq,lambda_reg):
    return tf.reduce_sum(tf.square(tf.reduce_mean(spikes,axis=(0,1)) - target_freq/1000.))*lambda_reg
  
  #@tf.function
  def inner_loop(self,target_coor_spike ,coor_t ,omega_t,mask):
  
    batch_size=target_coor_spike.shape[0]
    lsg_input = tf.concat((self.clock_signal_lsg ,target_coor_spike), axis = -1)
      
    w_in_tr  = tf.tile(tf.expand_dims(self.W_in_trainee,axis = 0),[batch_size,1,1])
    w_rec_tr = tf.tile(tf.expand_dims(self.W_rec_trainee,axis = 0),[batch_size,1,1])
    w_out_tr = tf.tile(tf.expand_dims(self.W_out_trainee,axis = 0),[batch_size,1,1]) 
     
    self.training_cell.set_init_weights(w_in_tr, w_rec_tr, w_out_tr
                                      ,self.W_in_lsg, self.W_rec_lsg, self.W_out_lsg)
    
    #asdf= self.training_network((self.clock_signal,lsg_input,self.update_mask))
    asdf= self.training_network((self.clock_signal,lsg_input))
    
    trainee_output_training = tf.boolean_mask(asdf[0], mask[0,:,0], axis = 1)
    trainee_spikes_training = asdf[1]
    lsg_spikes_training     = asdf[2]
    membrane_potential_tr   = asdf[3]
    membrane_potemtial_lsg  = asdf[4]
    delta_w_in              = asdf[6] 
    delta_w_rec             = asdf[7]
    

    w_in_trainee_test, w_rec_trainee_test = self.training_cell.reprog_weights(delta_w_in, delta_w_rec)
    
    #print(trainee_output_training[0])
    
    #print(tf.reduce_sum(delta_w_in))
        
    _, cartesian_training =self.loss(trainee_output_training, omega_t, coor_t)
    
    self.test_cell.set_weights(w_in_trainee_test, w_rec_trainee_test ,w_out_tr)
    
    trainee_output_test, trainee_spikes_test= self.test_network(self.clock_signal)
    #print(trainee_spikes_test)
    
    trainee_output_test = tf.boolean_mask(trainee_output_test, mask[0,:,0], axis = 1)
    
    loss_mean, cartesian_test =self.loss(trainee_output_test, omega_t, coor_t)
    
    loss = loss_mean +  self.loss_reg(tf.concat((trainee_spikes_training,trainee_spikes_test), axis = 1),self.reg_par.target_trainee_f,self.reg_par.lambda_trainee) + self.loss_reg(lsg_spikes_training,self.reg_par.target_lsg_f,self.reg_par.lambda_lsg)  
 
    return loss, trainee_output_test, trainee_output_training , cartesian_training,  cartesian_test  ,(omega_t, coor_t), (lsg_spikes_training, trainee_spikes_training,trainee_spikes_test)
  
  def do_training(self):
    target_coor_spike ,coor_t ,omega_t, mask = self.dataset.getdata()

    with tf.GradientTape(persistent=False) as g:
      loss, trainee_output_test, trainee_output_training, cartesian_training,  cartesian_test  , (omega_t, coor_t), spikes = self.inner_loop(target_coor_spike ,coor_t ,omega_t, mask)
    
    grads = g.gradient(loss,self.variables)
    
    norms = []
    for g in grads:
      norms.append(tf.norm(g, ord=2))
    
    
    if self.norm is not None:
      grads = [tf.clip_by_norm(g, self.norm) for g in grads]
    
    
    self.optimizer.apply_gradients(zip(grads, self.variables))
 
    return loss.numpy(), trainee_output_test.numpy(), trainee_output_training.numpy(), cartesian_training,  cartesian_test,(omega_t, coor_t), norms,spikes
  
  def do_testing(self):
    target_coor_spike ,coor_t ,omega_t, mask = self.dataset.getdata()
    loss, trainee_output_test, trainee_output_training , cartesian_training,  cartesian_test ,(omega_t, coor_t),spikes = self.inner_loop(target_coor_spike ,coor_t ,omega_t,mask)
    
    #w_in_f = tf.reduce_mean(w_in_f, axis =0 )
    #w_rec_f = tf.reduce_mean(w_rec_f, axis =0 )
    
    return loss.numpy(), trainee_output_test.numpy(), trainee_output_training.numpy(), cartesian_training,  cartesian_test,(omega_t, coor_t)  ,spikes
    
  """
  def inner_loop(self, target_coor_spike):  
    train_output, trainee_spikes_train, trainee_membrane_potentials, trainee_to_lsg = self.trainee.forward(self.clock_signal,self.W_in_trainee , self.W_rec_trainee ,self.W_out_trainee)
    
    
    #trainee_to_lsg_sync = tf.repeat(trainee_to_lsg, np.full(trainee_to_lsg.shape[1], self.lsg_input_per_trainee), axis=1)
    #L, lsg_spikes, _,_  = self.lsg.forward(tf.concat((self.clock_signal_lsg, trainee_to_lsg_sync ,target_coor_spike), axis = -1), self.W_in_lsg, self.W_rec_lsg, self.W_out_lsg)
    
    L, lsg_spikes, _,_  = self.lsg.forward(tf.concat((self.clock_signal_lsg ,target_coor_spike), axis = -1), self.W_in_lsg, self.W_rec_lsg, self.W_out_lsg)
    
    w_in_inner, w_rec_inner = self.trainee.weightupdate(self.W_in_trainee, self.W_rec_trainee, L[:,self.lsg_input_per_trainee-1::self.lsg_input_per_trainee] ,self.inner_lr, self.clock_signal, trainee_spikes_train, trainee_membrane_potentials, input_per_output = self.trainee_input_per_output)
    
    test_output ,trainee_spikes_test ,_ ,_ = self.trainee.forward(self.clock_signal ,w_in_inner ,w_rec_inner ,self.W_out_trainee)
    
    #if self.trainee_input_per_output > 1:
    #  test_output = test_output[:,::self.trainee_input_per_output]
    #  train_output = train_output[:,::self.trainee_input_per_output]
        
    return test_output, train_output, tf.concat((trainee_spikes_test, trainee_spikes_train), axis = 1) , lsg_spikes ,w_in_inner, w_rec_inner
  
  
  
  
  
  
  def do_testing(self):  #the same as do_training but without gradient
    target_coor_spike ,coor_t ,omega_t = self.dataset.getdata()
    
    test_output, train_output, trainee_spike_train , lsg_spike_train, w_in_f,w_rec_f  = self.inner_loop(target_coor_spike)
    
    loss = self.loss(test_output, omega_t, coor_t) + self.loss_reg(trainee_spike_train,self.reg_par.target_trainee_f,self.reg_par.lambda_trainee) + self.loss_reg(lsg_spike_train,self.reg_par.target_lsg_f,self.reg_par.lambda_lsg)  
    
    return loss.numpy(), test_output.numpy(), train_output.numpy() , (omega_t, coor_t)
  """










































class L2Lsetup():
  def __init__(self,dataset, 
                    n_trainee_rec, 
                    n_lsg_rec, 
                    lsg_binary_encoding,
                    trainee_par = LIFParameterTuple(thr=0.4, tau=20.,tau_o=20.,damp=0.3,n_ref=5),
                    lsg_par = LIFParameterTuple(thr=0.4, tau=20.,tau_o=20.,damp=0.3,n_ref=5),
                    quantize=None, #{"trainee_quan" ,"trainee_eprop","lsg_quan", "output_groups"}
                    output_groups = 16,
                    lr = 1.5e-3,
                    inner_lr = 1e-4,
                    reg_par = RegStatsTuple(target_trainee_f=10.,target_lsg_f=10., lambda_trainee=0.,lambda_lsg=0.)):
    
    self.dataset = dataset
    self.optimizer = tf.keras.optimizers.Adam(lr = lr)
    self.inner_lr = inner_lr
    self.trainee_input_per_output = 1#trainee_input_per_output
    self.lsg_input_per_trainee = 1#lsg_input_per_trainee
    self.reg_par = reg_par
    self.lr_decay = 1.
    n_in = dataset.n_in
    n_out = dataset.n_out
    
    if quantize is not None:
      #n_lsg_in = n_in + output_groups + (n_out * n_lsg_target)
      n_lsg_in = n_in +  (n_out * lsg_binary_encoding)
      n_lsg_out = output_groups    #output_groups
    else:
      #n_lsg_in = n_in + n_trainee_rec + (n_out * n_lsg_target)
      n_lsg_in = n_in + (n_out * lsg_binary_encoding)
      n_lsg_out = n_trainee_rec
    
    self.W_in_trainee  = tf.Variable(name="InputWeight_trainee",     initial_value=(rd.randn(n_in, n_trainee_rec) / np.sqrt(n_in)).astype(np.float32),dtype='float32',trainable = True)
    self.W_rec_trainee = tf.Variable(name="RecurrentWeight_trainee", initial_value=(rd.randn(n_trainee_rec, n_trainee_rec) / np.sqrt(n_trainee_rec)).astype(np.float32),dtype='float32',trainable = True)
    self.W_out_trainee = tf.Variable(name="OutputWeight_trainee",    initial_value=(rd.randn(n_trainee_rec, n_out) / np.sqrt(n_trainee_rec)).astype(np.float32) * (1. -np.exp(-1/20)),dtype='float32',trainable = True)
    
    self.W_in_lsg  = tf.Variable(name="InputWeight_lsg",     initial_value=(rd.randn(n_lsg_in, n_lsg_rec) / np.sqrt(n_lsg_in)).astype(np.float32),dtype='float32',trainable = True)
    self.W_rec_lsg = tf.Variable(name="RecurrentWeight_lsg", initial_value=(rd.randn(n_lsg_rec, n_lsg_rec) / np.sqrt(n_lsg_rec)).astype(np.float32),dtype='float32',trainable = True)
    
    self.W_out_lsg = tf.Variable(name="OutputWeight_lsg",    initial_value=tf.keras.initializers.GlorotUniform()((n_lsg_rec,n_lsg_out)),dtype='float32',trainable = True)
    self.variables = [self.W_in_trainee,self.W_rec_trainee,self.W_out_trainee, self.W_out_lsg,self.W_in_lsg,self.W_rec_lsg]
    
    if quantize is not None:
      self.trainee = RSNNModel(n_in, n_trainee_rec,q_info = quantize["trainee_quan"], LIF_par= trainee_par,batch_weight = True, output_groups = output_groups, eprop_q_info = quantize["trainee_eprop"])
      self.lsg = RSNNModel(n_lsg_in, n_lsg_rec,q_info = quantize["lsg_quan"], LIF_par= lsg_par ,batch_weight = False)
    else:  
      self.trainee = RSNNModel(n_in, n_trainee_rec,LIF_par=trainee_par,batch_weight = True)
      self.lsg = RSNNModel(n_lsg_in, n_lsg_rec,LIF_par= lsg_par,batch_weight = False)
  
  def set_learningrate_decay(self, decay):
    self.lr_decay = decay
  def decrease_learning_rate(self):
    self.optimizer.lr = self.optimizer.lr * self.lr_decay
    
  def set_static_data(self):
    init_position, clock_signal, clock_signal_lsg = self.dataset.getstaticdata()
    self.init_position = init_position
    self.clock_signal = clock_signal
    self.clock_signal_lsg = clock_signal_lsg
    
  def inner_loop(self, target_coor_spike):  
    train_output, trainee_spikes_train, trainee_membrane_potentials, trainee_to_lsg = self.trainee.forward(self.clock_signal,self.W_in_trainee , self.W_rec_trainee ,self.W_out_trainee)
    
    
    #trainee_to_lsg_sync = tf.repeat(trainee_to_lsg, np.full(trainee_to_lsg.shape[1], self.lsg_input_per_trainee), axis=1)
    #L, lsg_spikes, _,_  = self.lsg.forward(tf.concat((self.clock_signal_lsg, trainee_to_lsg_sync ,target_coor_spike), axis = -1), self.W_in_lsg, self.W_rec_lsg, self.W_out_lsg)
    
    L, lsg_spikes, _,_  = self.lsg.forward(tf.concat((self.clock_signal_lsg ,target_coor_spike), axis = -1), self.W_in_lsg, self.W_rec_lsg, self.W_out_lsg)
    
    w_in_inner, w_rec_inner = self.trainee.weightupdate(self.W_in_trainee, self.W_rec_trainee, L[:,self.lsg_input_per_trainee-1::self.lsg_input_per_trainee] ,self.inner_lr, self.clock_signal, trainee_spikes_train, trainee_membrane_potentials, input_per_output = self.trainee_input_per_output)
    
    test_output ,trainee_spikes_test ,_ ,_ = self.trainee.forward(self.clock_signal ,w_in_inner ,w_rec_inner ,self.W_out_trainee)
    
    #if self.trainee_input_per_output > 1:
    #  test_output = test_output[:,::self.trainee_input_per_output]
    #  train_output = train_output[:,::self.trainee_input_per_output]
        
    return test_output, train_output, tf.concat((trainee_spikes_test, trainee_spikes_train), axis = 1) , lsg_spikes ,w_in_inner, w_rec_inner
  
  def set_weights(self,variables):
    self.W_in_trainee = variables[0]
    self.W_rec_trainee = variables[1]
    self.W_out_trainee= variables[2]
    self.W_out_lsg= variables[3]
    self.W_in_lsg= variables[4]
    self.W_rec_lsg= variables[5]
  
  @tf.function
  def loss(self,output, omega_t, coor_t):
    
    position = self.init_position[:, None, :] + 0.02 * tf.cumsum(output, axis=1) #some weird scaling ask franz
    
    phi0 = position[..., 0]
    phi1 = position[..., 1] + phi0

    cartesian_x = (tf.cos(phi0) + tf.cos(phi1)) * .5
    cartesian_y = (tf.sin(phi0) + tf.sin(phi1)) * .5
    cartesian = tf.stack((cartesian_x, cartesian_y), -1)
    
    return .5*(tf.reduce_mean(tf.square(omega_t-output))+.5*tf.reduce_mean(tf.square(coor_t-cartesian)))

  @tf.function    
  def loss_reg(self,spikes,target_freq,lambda_reg):
    return tf.reduce_sum(tf.square(tf.reduce_mean(spikes,axis=(0,1)) - target_freq/1000.))*lambda_reg
  
  def do_training(self):
    target_coor_spike ,coor_t ,omega_t = self.dataset.getdata()
    
    with tf.GradientTape(persistent=False) as g:
      test_output, train_output, trainee_spike_train , lsg_spike_train, w_in_f,w_rec_f  = self.inner_loop(target_coor_spike)
      loss = self.loss(test_output, omega_t, coor_t) + self.loss_reg(trainee_spike_train,self.reg_par.target_trainee_f,self.reg_par.lambda_trainee) + self.loss_reg(lsg_spike_train,self.reg_par.target_lsg_f,self.reg_par.lambda_lsg)  
    
    w_in_f = tf.reduce_mean(w_in_f, axis =0 )
    w_rec_f = tf.reduce_mean(w_rec_f, axis =0 )
    
    grads = g.gradient(loss,self.variables)
    self.optimizer.apply_gradients(zip(grads, self.variables))
    return loss.numpy(), test_output.numpy(), train_output.numpy() , (omega_t, coor_t)
  
  
  def do_testing(self):  #the same as do_training but without gradient
    target_coor_spike ,coor_t ,omega_t = self.dataset.getdata()
    
    test_output, train_output, trainee_spike_train , lsg_spike_train, w_in_f,w_rec_f  = self.inner_loop(target_coor_spike)
    
    loss = self.loss(test_output, omega_t, coor_t) + self.loss_reg(trainee_spike_train,self.reg_par.target_trainee_f,self.reg_par.lambda_trainee) + self.loss_reg(lsg_spike_train,self.reg_par.target_lsg_f,self.reg_par.lambda_lsg)  
    
    return loss.numpy(), test_output.numpy(), train_output.numpy() , (omega_t, coor_t)



