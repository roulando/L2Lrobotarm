import numpy as np
import matplotlib.pyplot as plt

import os
import pathlib
import datetime
import pickle

from models import QuantizationInfoTuple
from models import LIFParameterTuple
from models import epropQuanTuple ,LIFcell_quan ,eprop_quan
from network import L2Lsetup_doublecell

from setup_arm import Dataset, DatasetBrown
from network import L2Lsetup
from network import RegStatsTuple
from time import time
import tensorflow as tf

def main():
  #simulation parameter
  
  tf.random.set_seed(42)
  np.random.seed(666)

  batch_size = 3 #90
    
  iterations = 25000
  result_folder = "Test_small_network_pcm_constraints"
  print_every = 250
  save = True
  save_every  = 250
  plot = True
  load = True
  lr_decay = 0.98
  decay_lr_every = 300
  test_iterations = 1000
  quantize = True
  validate_every  = 1000
  
  noise = True
  
  early_stopping = False
  
  weight_diagnostic = False
  collect_weight_diagnostic_over_n_iter = 250
  
  #data parameter
  n_t = 250  #length of the target movement   
  lsg_binary_encoding = 16 #Bits for one euclidian coordinate of the target movement
  time_steps_per_output = 1
  window_length = 130
  dt = 0.01 # for brown motion
  stoch_update = None  # None, "ignore", "linear" ,"linear_zero", "linear_fallof", "stochupdate_godknows"
  trainee_lsg_con = False
  
  #network parameter
  old_setup = True
  
  only_output = True
  
  n_in = 5            #number neurons for the input signal
  n_trainee_rec = 250 #number of neurons in trainee network
  n_out = 2           #number of output angels
  n_lsg_rec = 200     #number of neurons in the LSG
  lr = 1.5e-3         #outer loop learning rate
  #stoch update none: eprop_lr = 1e-4     #eprop update learning rate
  eprop_lr = 0.001
  
  reg_statsmodels = RegStatsTuple(target_trainee_f=10.,target_lsg_f=10., lambda_trainee=0.25,lambda_lsg=0.25) #Parameter for Regularizer target frequency for Trainee and LSG and lambda scales influence  
  trainee_parameter = LIFParameterTuple(thr=0.6, tau=20.,tau_o=20.,damp=0.3,n_ref=5)  #LIF neurons parameter for Trainee
  lsg_parameter = LIFParameterTuple(thr=0.6, tau=20.,tau_o=20.,damp=0.3,n_ref=5)      #LIF neurons parameter for LSG
  
  output_groups = 16  #number of neurons groups for the learning signals in the Trainee. Also sets number of outputs in the LSG. 16 is max
  clip_norm = None
  
    #Quantization Parameter ...b sets the overall bits. ...a sets the range of values and comma (see models Quantize and UQuantize)
  #  quantization_trainee = QuantizationInfoTuple(W_b = 8., W_a= 1., O_b= 16., O_a= 5.) #W_ sets the Quantization for the weights of the trainee. O_ the quantization of the output
  #  quantization_lsg = QuantizationInfoTuple(W_b = 8., W_a= 1., O_b= 16., O_a= 256.)   #W_ sets the Quantization for the weights of the LSG . O_ the quantization of the output and therefore the learning signals
  #  eprop_quan_parameter =  epropQuanTuple(tr_in_b = 12. ,tr_in_a = 8. ,tr_rec_b = 12. ,tr_rec_a = 8. ) #Sets the Quantization of the Input and Recurrent eligibility traces of the trainee.
  #  quan_par = {'trainee_quan' : quantization_trainee,'trainee_eprop' : eprop_quan_parameter, 'lsg_quan' : quantization_lsg} 
  if quantize:
    trainee_quan = LIFcell_quan(nb_w = 5. , a_w = 1. ,nb_v = 16., a_v = 16. , nb_out = 16., a_out = 32.)
    lsg_quan =     LIFcell_quan(nb_w = 5. , a_w = 1. ,nb_v = 16., a_v = 16. , nb_out = 16., a_out = 256.)
    trace_quan =   eprop_quan(nb_tr = 12., a_tr = 8.)
    quan_par = [trainee_quan,lsg_quan,trace_quan]
  else:
    quan_par = None
  
  #end of Parameter stuff
  
  if save:
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    train_folder = result_folder+"/Training_"+timestamp
    test_folder =  result_folder+"/Test_"+timestamp  
    if not os.path.exists(result_folder):
      os.makedirs(result_folder)
    if not os.path.exists(train_folder):
      os.makedirs(train_folder)
    if not os.path.exists(test_folder):
      os.makedirs(test_folder) 
    log_file = train_folder+"/log.txt"
  else:
    test_folder = None
    train_folder = None
    log_file = None

  #simulation
  dataset = DatasetBrown(batch_size,n_t,n_in, lsg_binary_encoding, time_steps_per_output = time_steps_per_output, window_length = window_length)
  #dataset = Dataset(batch_size,n_t,n_in, lsg_binary_encoding,time_steps_per_output = time_steps_per_output)
  
  
  network = L2Lsetup_doublecell(dataset, 
                    n_trainee_rec, 
                    n_lsg_rec, 
                    lsg_binary_encoding,
                    trainee_par = trainee_parameter, 
                    lsg_par = lsg_parameter , 
                    quantize =  quan_par,
                    output_groups = output_groups,
                    lr = lr,
                    noise = noise,
                    stochupdate = stoch_update,
                    norm = clip_norm,
                    inner_lr = eprop_lr,
                    reg_par = reg_statsmodels,
                    only_output = only_output,
                    old = old_setup)  
              
              
  network.set_static_data()
  network.set_learningrate_decay(lr_decay)
  
  if not load: 
    t_ref = time()
    
    network, _ , norms= training_loop(network,log_file = log_file ,iterations = iterations, weight_diagnostic = weight_diagnostic,collect_weight_diagnostic_over_n_iter=collect_weight_diagnostic_over_n_iter ,save=save  ,plot = plot, path = train_folder, early_stopping = early_stopping ,validate_every=validate_every ,decay_lr_every = decay_lr_every, print_every = print_every , store_every = save_every)
    
    if save:
      pickle.dump( [network.variables], open(result_folder+ "/weights.p", "wb" ) )
      #pickle.dump( [network], open(result_folder+ "/theholeshit.p", "wb" ) )
      #if plot:
      #  plot_norms(norms, store_as = train_folder+'/norms_dude.png')
    print('TRAINING FINISHED IN {:.2g} s'.format(time() - t_ref)) 
  else:
    data = pickle.load(open('weights.p', 'rb'))[0]
    
    network.set_weights(data)
  
  test_loop(network, iterations = 50, plot = plot, path = test_folder)

def plot_norms(norms, store_as = None):
  fig, ax = plt.subplots(len(norms))
  fig.set_figheight(20)
  fig.set_figwidth(20)
  
  for a,norm in zip(ax, norms):
    a.plot(norm)
  
  if store_as is None:
    plt.show()
  else:
    plt.savefig(store_as)
  plt.clf()
  plt.close()
    
    
def plot_spikes(spikes,store_as = None):
  fig, ax = plt.subplots(3)
  fig.set_figheight(10)
  fig.set_figwidth(10)
  
  raster_plot(ax[0],spikes[0][0])
  ax[0].title.set_text('LSG Training Spikes')
  raster_plot(ax[1],spikes[1][0])
  ax[1].title.set_text('Trainee Training Spikes')
  raster_plot(ax[2],spikes[2][0])
  ax[2].title.set_text('Trainee Testing Spikes')
  
  plt.tight_layout()
  
  if store_as is None:
    plt.show()
  else:
    plt.savefig(store_as)
  plt.clf()
  plt.close()
  
  
  del fig
  del ax 

def raster_plot(ax,spikes,linewidth=1.2,**kwargs):

  n_t,n_n = spikes.shape
  
  #print(np.where(spikes)[0])
  
  event_times,event_ids = spikes.numpy().nonzero()
  
  #print(event_times)
  
  max_spike = 10000
  event_times = event_times[:max_spike]
  event_ids = event_ids[:max_spike]

  for n,t in zip(event_ids,event_times):
    ax.vlines(t, n + 0., n + 1., linewidth=linewidth, **kwargs)

  ax.set_ylim([0 + .5, n_n + .5])
  ax.set_xlim([0, n_t])
  ax.set_yticks([0, n_n])
  del event_times
  del event_ids


def training_loop(network, log_file = None ,iterations = 20000 ,weight_diagnostic = False, collect_weight_diagnostic_over_n_iter=250, save = False , plot = False, path = None, early_stopping = False ,validate_every = None ,decay_lr_every = 1000, print_every = 250 , store_every = 250, figure_name = "this_is_a_figure" ):
  losses = []
  norms_over_training = []
  reset_diag= True
  if early_stopping:
    ignore_first_n = 3
    alpha = 0.9
    old_loss = 0.
    loss_diff_momentum = 0.
  
  for i in range(iterations):
    
    if i != 0 and i % decay_lr_every == 0 :
      network.decrease_learning_rate()
    
    loss, output_test, output_train,cartesian_training,  cartesian_test , target, norms , _ = network.do_training()
        
    norms_over_training.append(norms)
    
    #if i == 25000:
      #network.lsg.set_output_quantize()
      #network.ln.set_output_quantize()
    if i % print_every  == 0 or i == iterations-1:
      print('-'*50)    
      print("Iteration: "+str(i))
      print(loss)
      
      if log_file is not None:
        with open(log_file, "a") as file_object:
          file_object.write('-'*50)
          file_object.write('\n')
          file_object.write("Iteration: "+str(i))
          file_object.write('\n')
          file_object.write(str(loss))
          file_object.write('\n')
    losses.append(loss)
    
    if validate_every is not None:
      if i % validate_every  == 0 :
        if save:
          validation_folder = path+"/Validation_"+str(i)  
          if not os.path.exists(validation_folder):
            os.makedirs(validation_folder)
        else: 
          validation_folder = None
        val_loss = test_loop(network, iterations = 100, plot = plot, path = validation_folder)
        if save:
          pickle.dump( [network.variables], open(validation_folder+ "/weights.p", "wb" ) )
          #pickle.dump( [network], open(validation_folder+ "/theholeshit.p", "wb" ) )
        
        print(val_loss)
        if log_file is not None:
          with open(log_file, "a") as file_object:
            file_object.write("Val Loss: ")
            file_object.write(str(val_loss))
            file_object.write('\n')
          
        
        if early_stopping:
          if i == 0:
            old_loss = val_loss            
          else:
            loss_diff_momentum = alpha*loss_diff_momentum + (1.-alpha)*(old_loss-val_loss)
            
            old_loss = val_loss
            
    if path is not None:
      if ( i % store_every  == 0 or i == iterations-1):
        if plot:
          
          #do_plot((target[0], output_train ,output_test), output_train.shape[1] ,store_as = path+'/'+figure_name+"_"+str(i)+'.png')  
          do_plot_coordinates((target[1], cartesian_training,  cartesian_test), (target[0], output_train ,output_test), cartesian_training.shape[1] ,store_as = path+'/'+figure_name+"_"+str(i)+'.png')  
          
        pickle.dump( losses, open(path+"/losses.p", "wb" ) )
        pickle.dump( norms_over_training, open(path+"/norms.p", "wb" ) )
    if early_stopping and ( i > ignore_first_n) :
      if loss_diff_momentum < 0:
        print("Dude we are done at Iteration: "+str(i))
        if log_file is not None:
          with open(log_file, "a") as file_object:
            file_object.write("Dude we are done at Iteration: ")
            file_object.write(str(i))
            file_object.write('\n')
        break
        
  return network, losses, norms_over_training


def test_loop(network, iterations = 1000, plot = False, n_plots = 40,path = None, figure_name = "this_is_a_figure"):
  losses = []
  
  print("")
  print("Testing:")
  print('-'*50)
  
  for i in range(iterations):

    loss, output_test, output_train,  cartesian_training,  cartesian_test,target ,spikes = network.do_testing()
    
    losses.append(loss)

    if path is not None:
      if plot and i<n_plots:
        #do_plot((target[0], output_train ,output_test),output_train.shape[1],store_as = path+'/'+figure_name+"_"+str(i)+'.png')  
        do_plot_coordinates((target[1], cartesian_training,  cartesian_test), (target[0], output_train ,output_test) ,cartesian_training.shape[1],store_as = path+'/'+figure_name+"_"+str(i)+'.png')  
        plot_spikes(spikes, store_as= path+'/spikes_'+str(i)+'.png')
        
        if i < 3:
          pickle.dump( [spikes[0][0],spikes[1][0],spikes[2][0]], open(path+"/spikes_just_in_case_"+str(i)+"_.p", "wb" ) )
          
        
  print("Loss: "+str(np.mean(np.stack(losses)))+"+-"+str(np.std(np.stack(losses))))
  losses_av = np.mean(np.stack(losses))
  if path is not None:
    temp = np.stack(losses)
    pickle.dump( [np.mean(temp), np.std(temp)], open(path+"/test_loss.p", "wb" ) )
  
  return losses_av   
    
def do_plot_coordinates(data_coor, data_angles , n_t , store_as = None ):  
  
  asdf = data_coor[1]
  t = np.arange(0,n_t)
  fig, ax = plt.subplots(3,2)
  fig.set_figheight(10)
  fig.set_figwidth(10)
  ax[0][0].set_title('Training coordinates 1')
  ax[0][0].plot(np.asarray(asdf)[0,:,0],np.asarray(asdf)[0,:,1], label= 'network')
  ax[0][0].plot(np.asarray(data_coor[0])[0,:,0],np.asarray(data_coor[0])[0,:,1], label = 'target')
  
  ax[0][0].set_xlabel('x')
  ax[0][0].set_ylabel('y')
  ax[0][0].legend()
  
  asdf = data_coor[2]
  ax[0][1].set_title('Test coordinates 1')
  #ax[1][0].plot(t, np.asarray(asdf)[0,t,0], np.asarray(data[0])[0,t,0])
  ax[0][1].plot(np.asarray(asdf)[0,:,0],np.asarray(asdf)[0,:,1], label= 'network')
  ax[0][1].plot(np.asarray(data_coor[0])[0,:,0],np.asarray(data_coor[0])[0,:,1], label= 'target')
  ax[0][1].set_xlabel('x')
  ax[0][1].set_ylabel('y')
  ax[0][1].legend()

  
  asdf = data_angles[1]
  t = np.arange(0,n_t)
  ax[1][0].set_title('Training trial Angular velocities 1')
  ax[1][0].plot(t, np.asarray(asdf)[0,t,0], label= 'network')
  ax[1][0].plot(t, np.asarray(data_angles[0])[0,t,0], label = 'target')
  ax[1][0].set_xlabel('Time ms')
  ax[1][0].set_ylabel('Angular velocity rad')
  ax[1][0].legend()
  
  ax[1][1].set_title('Training Trial Angular Velocities 2')
  ax[1][1].plot(t, np.asarray(asdf)[0,t,1], label= 'network')
  ax[1][1].plot(t, np.asarray(data_angles[0])[0,t,1], label= 'target')
  ax[1][1].set_xlabel('Time ms')
  ax[1][1].set_ylabel('Angular velocity rad')
  ax[1][1].legend()
  
  asdf = data_angles[2]
  ax[2][0].set_title('Test Trial Angular Velocities 1')
  #ax[1][0].plot(t, np.asarray(asdf)[0,t,0], np.asarray(data[0])[0,t,0])
  ax[2][0].plot(t, np.asarray(asdf)[0,t,0], label= 'network')
  ax[2][0].plot(t, np.asarray(data_angles[0])[0,t,0], label= 'target')
  ax[2][0].set_xlabel('Time ms')
  ax[2][0].set_ylabel('Angular velocity rad')
  ax[2][0].legend()
  ax[2][1].set_title('Test Trial Angular Velocities 2')
  #ax[1][1].plot(t, np.asarray(asdf)[0,t,1], np.asarray(data[0])[0,t,1])
  ax[2][1].plot(t, np.asarray(asdf)[0,t,1], label= 'network')
  ax[2][1].plot(t, np.asarray(data_angles[0])[0,t,1], label= 'target')
  ax[2][1].legend()
  ax[2][1].set_xlabel('Time ms')
  ax[2][1].set_ylabel('Angular velocity rad')
  
  plt.tight_layout()
  
  if store_as is None:
    plt.show()
  else:
    plt.savefig(store_as)
    plt.clf()
    plt.close()
  


  

def do_plot(data , n_t , store_as = None ):  
  asdf = data[1]
  t = np.arange(0,n_t)
  fig, ax = plt.subplots(2,2)
  fig.set_figheight(10)
  fig.set_figwidth(10)
  ax[0][0].set_title('Training trial Angular velocities 1')
  ax[0][0].plot(t, np.asarray(asdf)[0,t,0], label= 'network')
  ax[0][0].plot(t, np.asarray(data[0])[0,t,0], label = 'target')
  ax[0][0].set_xlabel('Time ms')
  ax[0][0].set_ylabel('Angular velocity rad')
  ax[0][0].legend()
  
  ax[0][1].set_title('Training Trial Angular Velocities 2')
  ax[0][1].plot(t, np.asarray(asdf)[0,t,1], label= 'network')
  ax[0][1].plot(t, np.asarray(data[0])[0,t,1], label= 'target')
  ax[0][1].set_xlabel('Time ms')
  ax[0][1].set_ylabel('Angular velocity rad')
  ax[0][1].legend()
  
  asdf = data[2]
  ax[1][0].set_title('Test Trial Angular Velocities 1')
  #ax[1][0].plot(t, np.asarray(asdf)[0,t,0], np.asarray(data[0])[0,t,0])
  ax[1][0].plot(t, np.asarray(asdf)[0,t,0], label= 'network')
  ax[1][0].plot(t, np.asarray(data[0])[0,t,0], label= 'target')
  ax[1][0].set_xlabel('Time ms')
  ax[1][0].set_ylabel('Angular velocity rad')
  ax[1][0].legend()
  ax[1][1].set_title('Test Trial Angular Velocities 2')
  #ax[1][1].plot(t, np.asarray(asdf)[0,t,1], np.asarray(data[0])[0,t,1])
  ax[1][1].plot(t, np.asarray(asdf)[0,t,1], label= 'network')
  ax[1][1].plot(t, np.asarray(data[0])[0,t,1], label= 'target')
  ax[1][1].legend()
  ax[1][1].set_xlabel('Time ms')
  ax[1][1].set_ylabel('Angular velocity rad')
  plt.tight_layout()
  if store_as is None:
    plt.show()
  else:
    plt.savefig(store_as)
    plt.close()


if __name__ == '__main__':
    main()

