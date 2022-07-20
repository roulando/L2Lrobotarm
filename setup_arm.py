
#import torch
import tensorflow as tf

import numpy as np
import numpy.random as rd
import sys
#from torch.distributions import Bernoulli
import os

import matplotlib.pyplot as plt

class Brownian():
  def __init__(self,batch_size, dims, n_t, dt ,delta):
    self.batch_size = batch_size
    self.dims = dims
    self.n_t = n_t
    self.delta = delta
    self.dt = dt
    self.x0 =  0.

  def gen_random_walk(self):
    w = tf.ones((self.n_t,self.batch_size,self.dims))*self.x0
    def brown(w,x):
      yi = tf.random.normal((self.batch_size,self.dims), mean=0.0, stddev=self.delta*tf.math.sqrt(tf.cast(self.dt,tf.float32)))
      return w+(yi)
    return tf.transpose(tf.scan(brown, w, initializer = w[0]), perm = [1,0,2])

class TrajectoriesBrown():
  def __init__(self, n_batch, n_t, length_arm=0.5 , window_length=50 ,seed=3000):
    self.window_length =window_length
    self.n_t = n_t
    self.n_batch = n_batch
    self.random_state = np.random.RandomState(seed=seed)
    self.length_arm = length_arm
    self.n_out = 2
    self.intervall = 0.01
    self.start =  tf.tile(tf.expand_dims(tf.constant(np.array([0., np.pi / 2], np.float32)),axis=0), [self.n_batch,1] )      #tf.constant([[[0.,0.5]]])

    self.noise = Brownian(n_batch, self.n_out, n_t, self.intervall ,1.)
   
  def getdata(self):
    angular_velocities = self.noise.gen_random_walk()
    
    padded_angular_vel = tf.concat((tf.reverse(angular_velocities[:,0:self.window_length-1],axis =[1]), angular_velocities  , tf.reverse(angular_velocities[:,-self.window_length+1:],axis =[1])), axis = 1)
    window = tf.signal.hann_window(self.window_length)
    
    window = window/tf.reduce_sum(window)
    
    angular_velocities = tf.nn.convolution(padded_angular_vel, tf.tile(tf.expand_dims(tf.expand_dims(window,axis = -1),axis = -1),[1,1,padded_angular_vel.shape[-1]]),padding='VALID')[:,int(self.window_length//2):(int(self.window_length//2)+self.n_t)]
    angular_velocities_scaled = tf.clip_by_value(angular_velocities, -np.pi,np.pi)
  
    angles_scaled = tf.expand_dims(self.start,axis = 1) +self.intervall* tf.math.cumsum(angular_velocities_scaled, axis = 1)
    
    cartesian_x_scaled = tf.expand_dims((tf.cos(angles_scaled[:,:,0]) + tf.cos(angles_scaled[:,:,1]+angles_scaled[:,:,0])) * self.length_arm, axis = -1)
    cartesian_y_scaled = tf.expand_dims((tf.sin(angles_scaled[:,:,0]) + tf.sin(angles_scaled[:,:,1]+angles_scaled[:,:,0])) * self.length_arm, axis = -1)
    cartesian_scaled = tf.concat((cartesian_x_scaled,cartesian_y_scaled), axis = -1)
    
    return cartesian_scaled, angular_velocities_scaled

class DatasetBrown():
  def __init__(self,batch_size,n_t,n_in, n_encode, time_steps_per_output = 1 ,window_length=50):
    self.this_is_a_dataset = TrajectoriesBrown(batch_size, n_t, window_length=window_length) 
    
    self.batch_size = batch_size
    self.n_out = 2
    self.n_t = n_t
    self.n_in = n_in
    self.n_encode = n_encode
    self.time_steps_per_output = time_steps_per_output
    
  def getdata(self):
    coor,omega  = self.this_is_a_dataset.getdata()
    
    x_b = np.cast['int64']((coor[:,:,0]+1)*2**(self.n_encode-1))
    x_b = (((x_b[:,:,None] & (1 << np.arange(self.n_encode)))) > 0).astype(int)
            
    y_b = np.cast['int64']((coor[:,:,1]+1)*2**(self.n_encode-1))

    y_b = (((y_b[:,:,None] & (1 << np.arange(self.n_encode)))) > 0).astype(int)
    coor_encode = tf.concat((tf.constant(x_b),tf.constant(y_b)) , -1)
    
    coor_encode_padded = tf.repeat(coor_encode, np.full(coor_encode.shape[1], self.time_steps_per_output), axis=1)
    
    mask = np.zeros((1, 1 ,self.time_steps_per_output, 1))
    mask[0][0][self.time_steps_per_output-1] = 1
    mask = tf.reshape(tf.tile(tf.constant(mask), [1, self.n_t, 1 , 1]), (1, self.n_t* self.time_steps_per_output, 1))
    
    return tf.cast(coor_encode_padded, tf.float32 ), tf.cast(coor, tf.float32), tf.cast(omega, tf.float32), tf.cast(mask, tf.bool)
    
  def getstaticdata(self):
    
    clock_signal = tf.tile(tf.expand_dims(tf.constant(givemeclocksignal(self.n_t*self.time_steps_per_output, self.n_in,self.n_in*self.time_steps_per_output,1,10)), axis = 0), [self.batch_size,1,1] )  #seq_len, n_in, step_len,step_group, spike_every     #plus expand to bach size
    
    init_position = self.this_is_a_dataset.start 
    return tf.cast(init_position, tf.float32 ),tf.cast(clock_signal, tf.float32), tf.cast(clock_signal, tf.float32)

def givemeclocksignal(seq_len, n_in, step_len,step_group, spike_every):
  step_len = step_len * spike_every
  input_spikes = np.zeros((int(seq_len), n_in))
  n_of_steps = seq_len//step_len
  if n_of_steps == 0:
    n_of_steps = 1
  
  neuron_pointer = 0 
  step_pointer = 0
  
  for i in range(0,seq_len ,n_of_steps):  # clock signal like
    for j in range(step_group):   
      local_step_pointer = step_pointer
      for k in range(step_len):       
        if k % spike_every == 0:
          input_spikes[local_step_pointer][neuron_pointer]=1
        local_step_pointer+=1        
        if local_step_pointer >= seq_len:
          if j+1 == step_group:
            
            return input_spikes
          else:
            break      
      neuron_pointer+=1    
      if neuron_pointer>= n_in:
        neuron_pointer = 0     
      if j+1 == step_group:
        step_pointer = local_step_pointer
 
  return input_spikes


def spike_encode(input_component, minn, maxx, n_input_code=100, max_rate_hz=200, dt=1, n_dt_per_step=None):
    if 110 < n_input_code < 210:  # 100
        factor = 20
    elif 90 < n_input_code < 110:  # 100
        factor = 10
    elif 15 < n_input_code < 25:  # 20
        factor = 4
    else:
        factor = 2

    sigma_tuning = (maxx - minn) / n_input_code * factor
    mean_line = tf.cast(tf.linspace(minn - 2. * sigma_tuning, maxx + 2. * sigma_tuning, n_input_code), tf.float32)
    max_rate = max_rate_hz / 1000
    max_prob = max_rate * dt

    step_neuron_firing_prob = max_prob * tf.exp(-(mean_line[None, None, :] - input_component[..., None]) ** 2 /
                                                (2 * sigma_tuning ** 2))

    if n_dt_per_step is not None:
        spike_code = tf.distributions.Bernoulli(probs=step_neuron_firing_prob, dtype=tf.bool).sample(n_dt_per_step)
        dims = len(spike_code.get_shape())
        r = list(range(dims))
        spike_code = tf.transpose(spike_code, r[1:-1] + [0, r[-1]])
    else:
        spike_code = tf.compat.v1.distributions.Bernoulli(probs=step_neuron_firing_prob, dtype=tf.bool).sample()

    spike_code = tf.cast(spike_code, tf.float32)
    return spike_code




    
    #print(cartesian- filtered_tensor)
    
    #print(cartesian[0])
    
    #plt.scatter(cartesian[0,:,0],cartesian[0,:,1])
    #plt.scatter(cartesian[1,:,0],cartesian[1,:,1])
    
    #plt.scatter(cartesian_scaled[0,:,0]+2.,cartesian_scaled[0,:,1])
    #plt.scatter(cartesian_scaled[1,:,0]+0.5,cartesian_scaled[1,:,1])
    
    
    #plt.scatter(cartesian[2,:,0],cartesian[2,:,1])
    #plt.scatter(cartesian[0,:,0]+.2,cartesian[0,:,1]+.2)
    #plt.scatter(filtered_tensor[0,:,0],filtered_tensor[0,:,1])
    #plt.scatter(filtered_tensor[1,:,0],filtered_tensor[1,:,1])
    #plt.scatter(filtered_tensor[2,:,0],filtered_tensor[2,:,1])
    
    
    
    
    #x_dir = tf.random.uniform((self.n_batch,1))#-0.5
    #y_dir = tf.random.uniform((self.n_batch,1), minval=0.5)
    #coordinates = tf.concat((x_dir,y_dir),axis=1)
    
    #magni = tf.expand_dims(tf.math.sqrt(tf.reduce_sum(coordinates**2, axis = 1)),axis = -1)
    #unit_coordinates = coordinates/magni  
    
    #total_len = self.length_arm+ self.length_arm
    
    #stop = tf.random.uniform((self.n_batch,1), minval=0, maxval=self.intervall)

    #end_range =   total_len - stop
    
    #coordinates = tf.cast(tf.tile(self.start,[self.n_t,self.n_batch,1]),tf.float32)   #x is index 0 y is index 1
    
    #step_len =  (end_range-tf.math.sqrt(tf.reduce_sum(self.start**2)))/self.n_t*unit_coordinates
    #initializer = tf.zeros_like(coordinates[0])
    #coordinates = tf.scan(lambda a,x: a + step_len, coordinates)
    #coordinates = tf.transpose(coordinates, perm=[1, 0, 2])
      
    #angles2 = tf.math.acos( (coordinates[:,:,0]**2+coordinates[:,:,1]**2-self.length_arm**2 - self.length_arm**2 ) / (2*self.length_arm*self.length_arm))
    #angles1 = tf.expand_dims(tf.math.atan(coordinates[:,:,1]/coordinates[:,:,0]) - tf.math.atan((self.length_arm*tf.math.sin(angles2))/(self.length_arm+self.length_arm*tf.math.cos(angles2))),axis=-1)
    #angles2 = tf.expand_dims(angles2, axis= -1)
    
    #angles_pack_for_mapfun_1 = tf.concat((angles1[:,:-1],angles1[:,1:]),axis = -1)
    #angular_velocities_1 = tf.expand_dims(tf.transpose(tf.map_fn(fn=lambda x: x[:,1]-x[:,0] , elems= tf.transpose(angles_pack_for_mapfun_1, perm=[1, 0, 2]) ),perm=[1, 0]),axis = -1)
    
    #angles_pack_for_mapfun_2 = tf.concat((angles2[:,:-1],angles2[:,1:]),axis = -1)
    #angular_velocities_2 = tf.expand_dims(tf.transpose(tf.map_fn(fn=lambda x: x[:,1]-x[:,0] , elems= tf.transpose(angles_pack_for_mapfun_2, perm=[1, 0, 2]) ),perm=[1, 0]),axis = -1)
    
    
    #angular_velocities = tf.concat((angular_velocities_1,angular_velocities_2),axis = -1)
    #angular_velocities = tf.concat((tf.zeros_like(tf.expand_dims(angular_velocities[:,0],axis = 1)), angular_velocities), axis = 1)
    #start_angles = tf.expand_dims(tf.concat((angles1[:,0],angles2[:,0]),axis = -1), axis = 1)
   
    #hallo = start_angles + tf.math.cumsum(angular_velocities, axis = 1)
    
    #cartesian_x = tf.expand_dims((tf.cos(angles1[:,:,0]) + tf.cos(angles2[:,:,0]+angles1[:,:,0])) * self.length_arm, axis = -1)
    #cartesian_y = tf.expand_dims((tf.sin(angles1[:,:,0]) + tf.sin(angles2[:,:,0]+angles1[:,:,0])) * self.length_arm, axis = -1)
    #cartesian = tf.concat((cartesian_x,cartesian_y), axis = -1)
    
    
    #print(cartesian- filtered_tensor)
    
    #print(cartesian[0])
    
    #plt.scatter(cartesian[0,:,0],cartesian[0,:,1])
    #plt.scatter(cartesian[1,:,0],cartesian[1,:,1])
    #plt.scatter(cartesian[2,:,0],cartesian[2,:,1])
    #plt.scatter(cartesian[0,:,0]+.2,cartesian[0,:,1]+.2)
    #plt.scatter(filtered_tensor[0,:,0],filtered_tensor[0,:,1])
    #plt.scatter(filtered_tensor[1,:,0],filtered_tensor[1,:,1])
    #plt.scatter(filtered_tensor[2,:,0],filtered_tensor[2,:,1])
    #plt.show()
    #quit()
    
    #print(end_range/self.n_t)
    
    





class Trajectories_linear_old():
  def __init__(self, n_batch, n_t, length_arm=0.5 ,seed=3000):
    self.n_t = n_t
    self.n_batch = n_batch
    self.random_state = np.random.RandomState(seed=seed)
    self.length_arm = length_arm
    self.n_out = 2
    self.intervall = 0.2
    self.start = tf.constant([[[0.,0.5]]])
    
    
  def get_init(self):
    
    angles2 = tf.math.acos( (self.start[:,:,0]**2+self.start[:,:,1]**2-self.length_arm**2 - self.length_arm**2 ) / (2*self.length_arm*self.length_arm))
    angles1 = tf.expand_dims(tf.math.atan(self.start[:,:,1]/self.start[:,:,0]) - tf.math.atan((self.length_arm*tf.math.sin(angles2))/(self.length_arm+self.length_arm*tf.math.cos(angles2))),axis=-1)
    angles2 = tf.expand_dims(angles2, axis= -1)
    
    
    return tf.concat((angles1, angles2),axis = -1)  
    
  def getdata(self):
    x_dir = tf.random.uniform((self.n_batch,1))#-0.5
    y_dir = tf.random.uniform((self.n_batch,1), minval=0.5)
    coordinates = tf.concat((x_dir,y_dir),axis=1)
    
    magni = tf.expand_dims(tf.math.sqrt(tf.reduce_sum(coordinates**2, axis = 1)),axis = -1)
    unit_coordinates = coordinates/magni  
    
    total_len = self.length_arm+ self.length_arm
    
    stop = tf.random.uniform((self.n_batch,1), minval=0, maxval=self.intervall)

    end_range =   total_len - stop
    
    coordinates = tf.cast(tf.tile(self.start,[self.n_t,self.n_batch,1]),tf.float32)   #x is index 0 y is index 1
    
    step_len =  (end_range-tf.math.sqrt(tf.reduce_sum(self.start**2)))/self.n_t*unit_coordinates
    #initializer = tf.zeros_like(coordinates[0])
    coordinates = tf.scan(lambda a,x: a + step_len, coordinates)
    coordinates = tf.transpose(coordinates, perm=[1, 0, 2])
      
    angles2 = tf.math.acos( (coordinates[:,:,0]**2+coordinates[:,:,1]**2-self.length_arm**2 - self.length_arm**2 ) / (2*self.length_arm*self.length_arm))
    angles1 = tf.expand_dims(tf.math.atan(coordinates[:,:,1]/coordinates[:,:,0]) - tf.math.atan((self.length_arm*tf.math.sin(angles2))/(self.length_arm+self.length_arm*tf.math.cos(angles2))),axis=-1)
    angles2 = tf.expand_dims(angles2, axis= -1)
    
    angles_pack_for_mapfun_1 = tf.concat((angles1[:,:-1],angles1[:,1:]),axis = -1)
    angular_velocities_1 = tf.expand_dims(tf.transpose(tf.map_fn(fn=lambda x: x[:,1]-x[:,0] , elems= tf.transpose(angles_pack_for_mapfun_1, perm=[1, 0, 2]) ),perm=[1, 0]),axis = -1)
    
    angles_pack_for_mapfun_2 = tf.concat((angles2[:,:-1],angles2[:,1:]),axis = -1)
    angular_velocities_2 = tf.expand_dims(tf.transpose(tf.map_fn(fn=lambda x: x[:,1]-x[:,0] , elems= tf.transpose(angles_pack_for_mapfun_2, perm=[1, 0, 2]) ),perm=[1, 0]),axis = -1)
    
    
    angular_velocities = tf.concat((angular_velocities_1,angular_velocities_2),axis = -1)
    angular_velocities = tf.concat((tf.zeros_like(tf.expand_dims(angular_velocities[:,0],axis = 1)), angular_velocities), axis = 1)
    #start_angles = tf.expand_dims(tf.concat((angles1[:,0],angles2[:,0]),axis = -1), axis = 1)
    
    return coordinates, angular_velocities
    
    #hallo = start_angles + tf.math.cumsum(angular_velocities, axis = 1)
    
    #cartesian_x = tf.expand_dims((tf.cos(angles1[:,:,0]) + tf.cos(angles2[:,:,0]+angles1[:,:,0])) * self.length_arm, axis = -1)
    #cartesian_y = tf.expand_dims((tf.sin(angles1[:,:,0]) + tf.sin(angles2[:,:,0]+angles1[:,:,0])) * self.length_arm, axis = -1)
    #cartesian = tf.concat((cartesian_x,cartesian_y), axis = -1)
    
    
    #print(cartesian- filtered_tensor)
    
    #print(cartesian[0])
    
    #plt.scatter(cartesian[0,:,0],cartesian[0,:,1])
    #plt.scatter(cartesian[1,:,0],cartesian[1,:,1])
    #plt.scatter(cartesian[2,:,0],cartesian[2,:,1])
    #plt.scatter(cartesian[0,:,0]+.2,cartesian[0,:,1]+.2)
    #plt.scatter(filtered_tensor[0,:,0],filtered_tensor[0,:,1])
    #plt.scatter(filtered_tensor[1,:,0],filtered_tensor[1,:,1])
    #plt.scatter(filtered_tensor[2,:,0],filtered_tensor[2,:,1])
    #plt.show()
    #quit()
    
    #print(end_range/self.n_t)
    
    







class Trajectories():
  def __init__(self, n_batch, seq_length, n_periods, dt_step=.01, sine_seed=3000, static=False):
    self.n_sines = 5
    #self.periods = np.array([.5, 1., .5])[:self.n_sines]
    self.seq_length = seq_length
    self.n_periods = n_periods
    self.n_batch = n_batch
    self.random_state = np.random.RandomState(seed=sine_seed)
    #self.seed = sine_seed
    self.static = static
    self.dt_step = dt_step

  def getdata(self):
    t = np.linspace(0, 1 * 2 * np.pi, self.seq_length // self.n_periods)
    
    while True:
      if self.static:
          self.random_state = np.random.RandomState(seed=self.seed)
      periods = self.random_state.rand(self.n_batch, self.n_sines) * .7 + .3
      phase_motor0 = self.random_state.rand(self.n_batch, self.n_sines) * np.pi * 2
      amp0 = self.random_state.rand(self.n_batch, self.n_sines) * 1.5
      omega0 = np.sin(t[..., None, None] / periods[None, ...] + phase_motor0[None, ...]) * amp0[None, ...]
      # omega0 = (omega0 / (omega0.max(0) - omega0.min(0)) * 1.).sum(-1)
      omega0 = omega0.sum(-1)
      # phi0 = np.clip(dt_step * np.cumsum(omega0, 0), -np.pi / 2, 0)
      phi0 = self.dt_step * np.cumsum(omega0, 0)
      phi0_max = np.max(phi0, 0)
      phi0_min = np.min(phi0, 0)
      # assert np.allclose(phi0_max, -phi0_min)
      selector = np.logical_or(phi0_max > np.pi / 2, phi0_min < -np.pi / 2)
      sc = (np.pi / 2) / phi0_max[selector]
      sc2 = (-np.pi / 2) / phi0_min[selector]
      sc[sc < 0.] = 1.
      sc2[sc2 < 0.] = 1.
      sc = np.min((sc, sc2), 0)
      phi0[:, selector] = sc[None, :] * phi0[:, selector]
      omega0[:, selector] = sc[None, :] * omega0[:, selector]
      # fig, ax = plt.subplots(1, figsize=(6, 5))
      # ax.plot(phi0)
      # fig.savefig(os.path.expanduser('~/tempfig.png'), dpi=200)
      assert np.all(np.abs(phi0) - 1e-5 <= np.pi / 2)
      # phi0_max = np.max(phi0)
      # if phi0_max > 0:
      #     phi0 -= phi0_max

      phase_motor1 = self.random_state.rand(self.n_batch, self.n_sines) * np.pi * 2
      amp1 = self.random_state.rand(self.n_batch, self.n_sines) * 1.5
      periods = self.random_state.rand(self.n_batch, self.n_sines) * .7 + .3
      omega1 = np.sin(t[..., None, None] / periods[None, ...] + phase_motor1[None, ...]) * amp1[None, ...]
      omega1 = (omega1 / (omega1.max(0) - omega1.min(0)) * 1.).sum(-1)
      # phi1 = phi0 + np.clip(dt_step * np.cumsum(omega1, 0), 0, np.pi / 2) + np.pi / 2
      phi1_rel = self.dt_step * np.cumsum(omega1, 0)
      # phi1_rel_min = np.min(phi1_rel)
      # if phi1_rel_min < 0:
      #     phi1 = phi0 + phi1_rel - phi1_rel_min
      phi1_max = np.max(phi1_rel, 0)
      phi1_min = np.min(phi1_rel, 0)
      selector = np.logical_or(phi1_max > np.pi / 2, phi1_min < -np.pi / 2)
      sc = (np.pi / 2) / phi1_max[selector]
      sc2 = (-np.pi / 2) / phi1_min[selector]
      sc[sc < 0.] = 1.
      sc2[sc2 < 0.] = 1.
      sc = np.min((sc, sc2), 0)
      # sc = np.min(((np.pi / 4) / phi1_max[selector], (-np.pi / 4) / phi1_min[selector]), 0)
      phi1_rel[:, selector] = sc[None, :] * phi1_rel[:, selector]
      assert np.all(np.abs(phi1_rel) - 1e-5 <= np.pi / 2)
      omega1[:, selector] = sc[None, :] * omega1[:, selector]
      phi1 = phi0 + phi1_rel + np.pi / 2

      # x0, x1 = amp0[None, ...] * np.cos(phi0), amp1[None, ...] * np.cos(phi1)
      # y0, y1 = amp0[None, ...] * np.sin(phi0), amp1[None, ...] * np.sin(phi1)

      x = (np.cos(phi0) + np.cos(phi1)).T * .5
      y = (np.sin(phi0) + np.sin(phi1)).T * .5
      # fig, ax = plt.subplots(4, 4, figsize=(6, 5))
      # print(x[0, 0])
      # print(y[0, 0])
      # print(x[1, 0])
      # print(y[1, 0])
      # for i in range(4):
      #     for j in range(4):
      #         ax[i][j].scatter(x[i * 4 + j, :FLAGS.plasticity_every:10], y[i * 4 + j, :FLAGS.plasticity_every:10],
      #                          s=1, c=t[:FLAGS.plasticity_every:10], cmap='coolwarm')
      #         ax[i][j].plot([x[0, 0]], [y[0, 0]], '.', color='C0')
      #         ax[i][j].set_ylim([-1, 1])
      #         ax[i][j].set_xlim([-1, 1])
      #         ax[i][j].xaxis.set_ticks([])
      #         ax[i][j].yaxis.set_ticks([])
      # fig.savefig(os.path.expanduser('~/trajectory.png'), dpi=200)
      # fig.savefig(os.path.expanduser('~/trajectory.svg'),)
      # quit()
      # phase_x = np.random.rand(self.n_batch, self.n_sines) * 2 * np.pi
      # phase_y = np.random.rand(self.n_batch, self.n_sines) * 2 * np.pi
      # amp = np.random.rand(self.n_batch, self.n_sines) * .8
      # x = amp[None, ...] * np.cos(t[..., None, None] / self.periods[None, ...] + phase_x[None, ...])
      # y = amp[None, ...] * np.sin(t[..., None, None] / self.periods[None, ...] + phase_y[None, ...])
      # x = np.sum(x, -1).T
      # y = np.sum(y, -1).T
      # x = np.cumsum(x, 1) * dt_step
      # y = np.cumsum(y, 1) * dt_step
      x = np.tile(x[:, :], (1, 2))
      y = np.tile(y[:, :], (1, 2))
      omega0 = np.tile(omega0.T[:, :], (1, 2))
      omega1 = np.tile(omega1.T[:, :], (1, 2))
      yield x, y, omega0, omega1

            


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes
    
    
