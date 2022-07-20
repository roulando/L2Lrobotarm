import numpy as np
import numpy.random as rd
import tensorflow as tf

from collections import namedtuple

from tensorflow.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.framework import function
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell

import tensorflow_probability as tfp

import setup_arm
from setup_arm import Dataset

QuantizationInfoTuple = namedtuple('QuanInfo', (
    'W_b',
    'W_a',
    'O_b',
    'O_a'
))


LIFParameterTuple = namedtuple('LIFParameter', (
    'thr',
    'tau',
    'tau_o',
    'damp',
    'n_ref'
))

epropQuanTuple = namedtuple('epropQuanInfo', (
  'tr_in_b',
  'tr_in_a',
  'tr_rec_b',
  'tr_rec_a'
))


class ReadProgFull():
  def __init(self):
    pass
  
  def read(self, w):
    return w
    
  def prog(self, w):
    return w
 


class PcmNoiseReadProg():
  def __init__(self,w_max = 1., g_max = 25., t_start = 32):
    self.w_max = w_max
    self.g_max = g_max
    self.t_start = t_start
    self.nb = 5.
    self.a  = 1.
      
  def read_noise(self, w):
    g_t = w*self.g_max
    
    positive_part = tf.maximum(g_t, 0.)
    negative_part = tf.minimum(g_t, 0.)
    
    t_read = 250.0e-9
    
    q_s = tf.minimum(0.0088/tf.math.pow(g_t, 0.65), 0.2)
    q_s = tf.where(tf.math.is_nan(q_s), 0., q_s)
    sigma = g_t*q_s* tf.math.sqrt(tf.math.log((self.t_start+t_read)/t_read))
    
    noise = tf.random.normal(g_t.shape,mean=0.0,stddev=sigma)
    
    mask_pos = tf.cast(tf.cast(positive_part, tf.bool),tf.float32)
    mask_neg = tf.cast(tf.cast(negative_part, tf.bool),tf.float32)
    
    noise = tf.random.normal(g_t.shape,mean=0.0,stddev=sigma)
    
    positive_part_noise = tf.maximum(positive_part + noise*mask_pos, 0.)
    negative_part_noise = tf.minimum(negative_part + noise*mask_neg,0.)
    
    g_t_noise = positive_part_noise + negative_part_noise  
    w_noise = g_t_noise / self.g_max
    
    return w_clipped_noise-w
  
  def read(self, w):
    return w + tf.stop_gradient(self.read_noise(w))
    
  def prog(self, w):
    return Clip_with_grad(Quantize(w, self.nb, self.a) + tf.stop_gradient(self.prog_noise(w)), -self.w_max, self.w_max)
    
  def prog_noise(self,w):
    g_t = w * self.g_max
    
    positive_part = tf.maximum(g_t, 0.)
    negative_part = tf.minimum(g_t, 0.)

    mask_pos = tf.cast(tf.cast(positive_part, tf.bool),tf.float32)
    mask_neg = tf.cast(tf.cast(negative_part, tf.bool),tf.float32)
        
    sigma_prog = tf.maximum(-1.1731*tf.square(g_t/self.g_max) + 1.9650*tf.abs(g_t/self.g_max) +0.2635 , 0.)
    noise = tf.random.normal(g_t.shape,mean=0.0,stddev=sigma_prog)
    
    positive_part_noise = Clip_with_grad(positive_part + noise*mask_pos, 0.,self.g_max)
    negative_part_noise = Clip_with_grad(negative_part + noise*mask_neg, -self.g_max,0.)

    g_t_noise = positive_part_noise + negative_part_noise  
    w_noise = g_t_noise/self.g_max
    
    return w_noise-w


@tf.function
def exp_convolve(tensor, decay, damp):
  with tf.name_scope('ExpConvolve'):
    tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
    initializer = tf.zeros_like(tensor_time_major[0])

    filtered_tensor = tf.scan(lambda a, x: a * decay + damp * x, tensor_time_major, initializer=initializer)
    filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
  return filtered_tensor

@tf.custom_gradient
def Quantize(W, nb, a):
  non_sign_bits = nb-1
  m = pow(2,non_sign_bits)
  n = m/a
  W_quan  = tf.clip_by_value(tf.math.round(W*n),-m,m-1)/n
  
  def grad(dy):
    return (dy, None,None)
  
  return W_quan, grad



@tf.custom_gradient
def Quantize_min_grad(W, nb, a):
  non_sign_bits = nb-1
  m = pow(2,non_sign_bits)
  n = m/a
  W_quan  = tf.clip_by_value(tf.math.round(W*n),-m,m-1)/n
  
  def grad(dy):
    return (dy*tf.where(W_quan==0., 0., 1.), None,None)
  
  return W_quan, grad



@tf.custom_gradient
def Clip_with_grad(x,v_min,v_max):
  x_clipped  = tf.clip_by_value(x,v_min,v_max)
  
  def grad(dy):
    return (dy,None,None)
  
  return x_clipped, grad


@tf.function
def gradient_update(update, factor):
  return factor * update  



#@tf.function
def gradient_update_quan(update, factor):
  return Quantize_min_grad(factor * update, 8. , 1.  )




@tf.custom_gradient
def stochupdate_ignore(update, factor):
    update_pos_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value( update*factor,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_neg_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value(-update*factor,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_stoch = (update_pos_i - update_neg_i)*(1.0/pow(2,8-1))
    
    def grad(dy):
      return (dy*factor,None)
  
    return update_stoch, grad


@tf.custom_gradient
def stochupdate_linear(update, factor):
    update_pos_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value( update,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_neg_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value(-update,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_stoch = (update_pos_i - update_neg_i)*factor
    
    def grad(dy):
      return (dy*update*factor,None)
  
    return update_stoch, grad


@tf.custom_gradient
def stochupdate_linear_zero(update, factor):
    update_pos_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value( update*factor,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_neg_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value(-update*factor,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_stoch = (update_pos_i - update_neg_i)*(1.0/pow(2,8-1))
    
    def grad(dy):
      update_scaled = update * factor
      
      mask_1 = tf.where(update_scaled>1., 0., 1.)
      mask_2 = tf.where(update_scaled<-1., 0., 1.)
      update_1to1 = update_scaled*mask_1 *mask_2 
      return ( dy*(update_1to1)*(1.0/pow(2,8-1)), None)
      
    return update_stoch, grad

    
@tf.custom_gradient
def stochupdate_linear_fallof(update, factor):
    update_pos_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value( update*factor,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_neg_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value(-update*factor,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_stoch = (update_pos_i - update_neg_i)*(1.0/pow(2,8-1))
    
    def grad(dy):
      update_scaled = update * factor
      
      mask_1 = tf.where(update_scaled>1., 0., 1.)
      mask_2 = tf.where(update_scaled<-1., 0., 1.)
      update_1to1 = update_scaled*mask_1 *mask_2 

      update_1more = tf.clip_by_value(tf.where(update_scaled>1., -((update_scaled-1.)/100.), 0.), -1.,0.)
      update_1less = tf.clip_by_value(tf.where(update_scaled<-1., -((update_scaled+1.)/100.), 0.),0.,1.)
      return ( dy*(update_1to1+update_1more+update_1less)*(1.0/pow(2,8-1)), None)
  
    return update_stoch, grad   
    
@tf.custom_gradient
def stochupdate_godknows(update, factor):
    update_pos_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value( update*factor,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_neg_i = tf.cast(tfp.distributions.Bernoulli(probs=tf.clip_by_value(-update*factor,clip_value_min = 0., clip_value_max = 1.)).sample(), tf.float32)
    update_stoch = (update_pos_i - update_neg_i)*(1.0/pow(2,8-1))
    
    def grad(dy):
      update_scaled = update * factor
      
      mask_1 = tf.where(update_scaled>=-1., 1., 0.) * tf.where(update_scaled<0., 1., 0.)
      mask_2 = tf.where(update_scaled<=1., 1., 0.) * tf.where(update_scaled>0., 1., 0.)
      mask_3 = tf.where(update_scaled>1. , 1.,0.)
      mask_4 = tf.where(update_scaled<-1. , 1.,0.)

      stoch_update_grad = tf.exp(update_scaled-1)**8*mask_2 -(tf.exp(-update_scaled-1)**8)*mask_1 +mask_3*( tf.exp(1-update_scaled)**2)+ (-tf.exp(update_scaled+1)**2)*mask_4

      #update_1more = tf.clip_by_value(tf.where(update_scaled>1., -((update_scaled-1.)/100.), 0.), -1.,0.)
      #update_1less = tf.clip_by_value(tf.where(update_scaled<-1., -((update_scaled+1.)/100.), 0.),0.,1.)

      return ( dy* stoch_update_grad *(1.0/pow(2,8-1)), None)
  
    return update_stoch, grad    
    

@tf.custom_gradient
def UQuantize(W, nb,a):
  
  m = pow(2,nb)
  n = m/a
  W_uquan = tf.clip_by_value(tf.math.round(W*n),0,m-1)/n
  
  def grad(dy):
    return (dy, None,None)
  
  return W_uquan, grad


@tf.custom_gradient
def Quantize10(W, nb):
  
  W_uquan = tf.math.round(W/nb)*nb
  
  #W_uquan = tf.clip_by_value(tf.math.round(W*n),0,m-1)/n
  
  def grad(dy):
    return (dy, None)
  
  return W_uquan, grad


 
RSNNStateTuple = namedtuple('RNNNState', (
    'z',
    'v',
    'r'
))





@tf.custom_gradient
def SpikeFunction(v, thr ,dampening_factor):
  z_ = tf.greater(v, thr)
  z_ = tf.cast(z_, dtype=tf.float32)
  def grad(dy):
    return (dy * dampening_factor *tf.maximum(1. - tf.abs((v-thr)/thr), 0) / thr ,None ,None )
  return z_, grad



LIFcell_quan = namedtuple('QuantiLIFi', ('nb_w','a_w','nb_v','a_v','nb_out','a_out'))
eprop_quan = namedtuple('quantiepropi', ('nb_tr','a_tr'))

@tf.function
#def LIF_function(inputs,v,z,r,v_out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor, nb_v = 16., a_v = 1. , nb_out = 16., a_out = 256.):
def LIF_function(inputs,v,z,r,v_out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor, quan=None):  

  new_v = decay * v +  ( tf.matmul(inputs, w_in) + tf.matmul(z, (1.-tf.eye(w_rec.shape[0],w_rec.shape[0])) * w_rec) ) - z * thr
  
  new_z = tf.where(tf.greater(r, .1), tf.zeros_like(z), SpikeFunction(new_v,thr,dampening_factor))
  new_r = tf.clip_by_value(r + n_refractory * new_z - 1,
                           0., float(n_refractory))
  
  
  new_out = kappa * v_out + tf.matmul(new_z, w_out)
  
  return new_v, new_z, new_r, new_out                          


def read_noise(w):
  g_max = 25.
  t_start = 32.
  w_max = 1.
  g_t = w*g_max
  
  positive_part = tf.maximum(g_t, 0.)
  negative_part = tf.minimum(g_t, 0.)
  
  t_read = 250.0e-9
  
  q_s = tf.minimum(0.0088/tf.math.pow(g_t, 0.65), 0.2)
  q_s = tf.where(tf.math.is_nan(q_s), 0., q_s)
  sigma = g_t*q_s* tf.math.sqrt(tf.math.log((t_start+t_read)/t_read))
  
  noise = tf.random.normal(g_t.shape,mean=0.0,stddev=sigma)
  
  mask_pos = tf.cast(tf.cast(positive_part, tf.bool),tf.float32)
  mask_neg = tf.cast(tf.cast(negative_part, tf.bool),tf.float32)
  
  noise = tf.random.normal(g_t.shape,mean=0.0,stddev=sigma)
  
  positive_part_noise = tf.maximum(positive_part + noise*mask_pos, 0.)
  negative_part_noise = tf.minimum(negative_part + noise*mask_neg,0.)
  
  g_t_noise = positive_part_noise + negative_part_noise  
  w_noise = g_t_noise / g_max
  w_clipped_noise = Clip_with_grad(w_noise , -w_max, w_max)
  
  return w_clipped_noise-w






@tf.function
#def LIF_function_quant(inputs,v,z,r,v_out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor, nb_v = 16., a_v = 1. , nb_out = 16., a_out = 256.):
def LIF_function_quant(inputs,v,z,r,v_out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor, quan = LIFcell_quan(nb_w = 8. , a_w = 1. ,nb_v = 16., a_v = 1. , nb_out = 16., a_out = 256.)):
  w_in  = Quantize(w_in , quan.nb_w, quan.a_w)
  w_rec = Quantize(w_rec, quan.nb_w, quan.a_w)
  w_out = Quantize(w_out, quan.nb_w, quan.a_w)
  
  #new_v = decay * v +  ( tf.matmul(inputs, w_in) + tf.matmul(z, (1.-tf.eye(w_rec.shape[0],w_rec.shape[0])) * w_rec) ) - z * thr #Quantize( , quan.nb_v, quan.a_v) #decay * v +  ( tf.matmul(inputs, w_in) + tf.matmul(z, (1.-tf.eye(w_rec.shape[0],w_rec.shape[0])) * w_rec) ) - z * thr 
  asdf = tf_bi_ij_bj_noise(inputs, w_in)
  new_v = decay * v +  ( asdf + tf.matmul(z, (1.-tf.eye(w_rec.shape[0],w_rec.shape[0])) * w_rec) ) - z * thr #Quantize( , quan.nb_v, quan.a_v) #decay * v +  ( tf.matmul(inputs, w_in) + tf.matmul(z, (1.-tf.eye(w_rec.shape[0],w_rec.shape[0])) * w_rec) ) - z * thr 
  
  
  new_z = tf.where(tf.greater(r, .1), tf.zeros_like(z), SpikeFunction(new_v,thr,dampening_factor))
  new_r = tf.clip_by_value(r + n_refractory * new_z - 1,
                           0., float(n_refractory))
  
  new_out = kappa * v_out + tf.matmul(new_z, w_out) #Quantize(kappa * v_out + tf.matmul(new_z, w_out), quan.nb_out, quan.a_out)
  
  return new_v, new_z, new_r, new_out       



@tf.function
#def LIF_function_batch(inputs,v,z,r,out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor, nb_v = 16., a_v = 1. , nb_out = 16., a_out = 256.):
def LIF_function_batch(inputs,v,z,r,out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor, quan=None):  
  new_v = decay * v +  ( tf.einsum("bi,bir-> br",inputs,w_in) + tf.einsum("br, brj -> bj",z, (1.-tf.eye(w_rec.shape[1],w_rec.shape[1])) * w_rec) ) - z * thr
  
  new_z = tf.where(tf.greater(r, .1), tf.zeros_like(z), SpikeFunction(new_v,thr,dampening_factor))
  new_r = tf.clip_by_value(r + n_refractory * new_z - 1,
                           0., float(n_refractory))
  
  new_out = kappa * out + tf.einsum("bi,bir-> br",new_z, w_out)
                  
  return new_v, new_z, new_r, new_out


@tf.function
#def LIF_function_batch_quant(inputs,v,z,r,out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor, nb_w = 8. , a_w = 8. ,nb_v = 16., a_v = 1. , nb_out = 16., a_out = 256.):
def LIF_function_batch_quant(inputs,v,z,r,out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor, quan = LIFcell_quan(nb_w = 8. , a_w = 1. ,nb_v = 16., a_v = 1. , nb_out = 16., a_out = 256.)):  
  w_in  = Quantize(w_in , quan.nb_w, quan.a_w)
  w_rec = Quantize(w_rec, quan.nb_w, quan.a_w)
  w_out = Quantize(w_out, quan.nb_w, quan.a_w)
  
  new_v = decay * v +  ( tf.einsum("bi,bir-> br",inputs,w_in) + tf.einsum("br, brj -> bj",z, (1.-tf.eye(w_rec.shape[1],w_rec.shape[1])) * w_rec) ) - z * thr  #Quantize(decay * v +  ( tf.einsum("bi,bir-> br",inputs,w_in) + tf.einsum("br, brj -> bj",z, (1.-tf.eye(w_rec.shape[1],w_rec.shape[1])) * w_rec) ) - z * thr , quan.nb_v, quan.a_v) #decay * v +  ( tf.einsum("bi,bir-> br",inputs,w_in) + tf.einsum("br, brj -> bj",z, (1.-tf.eye(w_rec.shape[1],w_rec.shape[1])) * w_rec) ) - z * thr #Quantize(decay * v +  ( tf.einsum("bi,bir-> br",inputs,w_in) + tf.einsum("br, brj -> bj",z, (1.-tf.eye(w_rec.shape[1],w_rec.shape[1])) * w_rec) ) - z * thr , quan.nb_v, quan.a_v) 
  
  new_z = tf.where(tf.greater(r, .1), tf.zeros_like(z), SpikeFunction(new_v,thr,dampening_factor))
  new_r = tf.clip_by_value(r + n_refractory * new_z - 1,
                           0., float(n_refractory))
  
  new_out = kappa * out + tf.einsum("bi,bir-> br",new_z, w_out) #Quantize(kappa * out + tf.einsum("bi,bir-> br",new_z, w_out), quan.nb_out, quan.a_out)
                  
  return new_v, new_z, new_r, new_out
  

@tf.function
def eprop_update(L, trace_in, trace_rec, v ,z, x, r,damp, thr, alpha, quan = None):
  
  h = tf.maximum(damp*( 1.- tf.abs((v -thr)/thr)), 0.)/thr

  h  = tf.where(tf.greater(r, .1), tf.zeros_like(h), h)

  trace_in = trace_in  *alpha + x
  trace_rec = trace_rec*alpha + z
    
  return tf.einsum("br,bi->bir",h*L, trace_in), tf.einsum("br,bi->bir",h*L, trace_rec), trace_in, trace_rec



@tf.function
def outprop_update(L, trace_rec,z, alpha):
  trace_rec = trace_rec*alpha + z
    
  return tf.einsum("br,bo->bro",trace_rec,L ), trace_rec


@tf.function
def eprop_update_quant(L, trace_in, trace_rec,v ,z, x, damp, thr, alpha, quan = eprop_quan(nb_tr = 12., a_tr = 8.)):
  
  h = tf.maximum(damp*( 1.- tf.abs((v -thr)/thr)), 0.)/thr

  trace_in = UQuantize(trace_in  *alpha + x   ,  quan.nb_tr, quan.a_tr)
  trace_rec = UQuantize(trace_rec*alpha + z   , quan.nb_tr, quan.a_tr  )
  
  return tf.einsum("br,bi->bir",h*L, trace_in), tf.einsum("br,bi->bir",h*L, trace_rec), trace_in, trace_rec


@tf.custom_gradient
def tf_bi_bij_bj_noise(a, b):
  c = tf.einsum("bi,bij->bj",a, b+read_noise(b))
  
  def grad(dy):
    return (tf.einsum("bj,bij->bi",dy, b), tf.einsum("bj,bi->bij",dy, a))
  
  return c, grad




@tf.function
def LIF_function_batch_noise(inputs,v,z,r,out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor):  
  new_v = decay * v +  ( tf_bi_bij_bj_noise(inputs,w_in) + tf_bi_bij_bj_noise(z, (1.-tf.eye(w_rec.shape[1],w_rec.shape[1])) * w_rec) )- z * thr
  
  new_z = tf.where(tf.greater(r, .1), tf.zeros_like(z), SpikeFunction(new_v,thr,dampening_factor))
  new_r = tf.clip_by_value(r + n_refractory * new_z - 1,
                           0., float(n_refractory))
  
  new_out = kappa * out + tf_bi_bij_bj_noise(new_z, w_out)
                  
  return new_v, new_z, new_r, new_out




@tf.custom_gradient
def tf_bi_ij_bj_noise(a, b):
  c = tf.einsum("bi,ij->bj",a, b+read_noise(b))
  
  def grad(dy):
    return (tf.matmul(dy, tf.transpose(b)), tf.matmul(tf.transpose(a),dy))
  
  return c, grad



@tf.function
def LIF_function_noise(inputs,v,z,r,v_out,w_in, w_rec,w_out, thr,decay,kappa,n_refractory,dampening_factor):  

  new_v = decay * v +  ( tf_bi_ij_bj_noise(inputs, w_in) + tf_bi_ij_bj_noise(z, (1.-tf.eye(w_rec.shape[0],w_rec.shape[0])) * w_rec) ) - z * thr
  
  new_z = tf.where(tf.greater(r, .1), tf.zeros_like(z), SpikeFunction(new_v,thr,dampening_factor))
  new_r = tf.clip_by_value(r + n_refractory * new_z - 1,
                           0., float(n_refractory))
  
  
  new_out = kappa * v_out + tf_bi_ij_bj_noise(new_z, w_out)
  
  return new_v, new_z, new_r, new_out                          


LIFcell_metaeprop_trainee_LSG_Tuple = namedtuple('TraineeLSGState', ("out_trainee"
                                                                    ,"w_in_change"
                                                                    ,"w_rec_change"
                                                                    ,"v_trainee"
                                                                    ,"z_trainee"
                                                                    ,"r_trainee"
                                                                    ,"v_lsg"
                                                                    ,"z_lsg"
                                                                    ,"r_lsg"
                                                                    ,"L"
                                                                    ,"trace_in"
                                                                    ,"trace_rec"))



class LIFcell_metaeprop_trainee_LSG(tf.keras.layers.Layer):
  def __init__(self, n_in_trainee, units_trainee, n_out_trainee
                   , n_in_lsg , units_lsg, n_out_lsg
                   , LIF_par_trainee = LIFParameterTuple(thr=0.4, tau=20.,tau_o=20.,damp=0.3,n_ref=5)
                   , LIF_par_lsg = LIFParameterTuple(thr=0.4, tau=20.,tau_o=20.,damp=0.3,n_ref=5)
                   , lr = 0.01
                   , quantize = None
                   , old = False
                   , stop_w_in_update = False
                   , groups = None
                   , stochupdate = None
                   , noise = False
                   ,**kwargs):
    
    
    self.quantize = quantize is not None
    if quantize is not None:
      self.trainee_LIF_quan = quantize[0]
      self.lsg_LIF_quan = quantize[1]
      self.trainee_eprop_quan = quantize[2]
    else:
      self.trainee_LIF_quan = None
      self.lsg_LIF_quan = None
      self.trainee_eprop_quan = None
        
    self.dampening_factor_trainee = LIF_par_trainee.damp
    self.decay_trainee = tf.exp(-1./LIF_par_trainee.tau)
    self.kappa_trainee = tf.exp(-1./LIF_par_trainee.tau_o)
    
    self.units_trainee = units_trainee
    self.n_refractory_trainee = LIF_par_trainee.n_ref
    self.thr_trainee = LIF_par_trainee.thr
    self.n_out_trainee = n_out_trainee
    self.n_in_trainee = n_in_trainee
    
    self.dampening_factor_lsg = LIF_par_lsg.damp
    self.decay_lsg = tf.exp(-1./LIF_par_lsg.tau)
    self.kappa_lsg = tf.exp(-1./LIF_par_lsg.tau_o)
    
    self.units_lsg = units_lsg
    self.n_refractory_lsg = LIF_par_lsg.n_ref
    self.thr_lsg = LIF_par_lsg.thr
    self.n_out_lsg = n_out_lsg
    self.n_in_lsg = n_in_lsg
    
    if groups is not None:
      self.con_matrix =  self.get_connection_matrix_for_output_groups(groups)
    else:
      self.con_matrix = tf.eye(self.units_trainee)
    
    self.lr = lr
 
 
    #if noise:
    if noise:
      self.LIF_fun   = LIF_function_noise
      self.LIF_fun_b = LIF_function_batch_noise
      self.eprop_fun = eprop_update
    else:
      self.LIF_fun   = LIF_function
      self.LIF_fun_b = LIF_function_batch
      self.eprop_fun = eprop_update

    print("-"*50)
    
    print(self.LIF_fun.__name__)
    print(self.LIF_fun_b.__name__)
    print(self.eprop_fun.__name__)
    
    print("LIFcell_metaeprop_trainee_LSG parmaeter ")
    print("Quantization: "+str(quantize))
    print("Parameter Trainee: "+str(LIF_par_trainee))
    print("Parameter LSG: "+str(LIF_par_lsg))     
    print("Old setup "+str(old))
    print("eprop_ learning rate "+str(lr))
    print("Noise "+str(noise))
    print("groups "+ str(groups))
    print("inner learning rate "+str(lr))
    print("-"*50)
    
    if noise:
      self.w_module = PcmNoiseReadProg()
    else:
      self.w_module = ReadProgFull()
    
    self.state_size = [n_out_trainee, tf.constant((n_in_trainee, units_trainee)) , tf.constant((units_trainee,units_trainee)), units_trainee, units_trainee, units_trainee,  units_lsg,units_lsg,units_lsg, n_out_lsg , n_in_trainee, units_trainee]
    
    #[new_out_trainee      ,new_w_in_change,    new_w_rec_change,         new_v_trainee,  new_z_trainee,   new_r_trainee,  new_v_lsg ,   new_z_lsg  , new_r_lsg   , new_L    , new_trace_in   ,new_trace_rec ]
    
    self.output_size = [n_out_trainee, units_trainee, units_lsg]
    
    super(LIFcell_metaeprop_trainee_LSG, self).__init__(**kwargs)

  def get_connection_matrix_for_output_groups(self, groups):
    coc = np.zeros((self.units_trainee,groups))

    group_size =  self.units_trainee//groups
    
    for i in range(coc.shape[1]):
      begin = i*group_size
      end = begin + group_size
      if (coc.shape[0] - end) < group_size:
        end = coc.shape[0]  
      coc[begin:end,i]= 1

    return tf.constant(coc,dtype = tf.float32)

  def set_init_weights(self, w_in_trainee,w_rec_trainee,w_out_trainee,w_in_lsg, w_rec_lsg, w_out_lsg):
    self.w_in_lsg  = self.w_module.prog(w_in_lsg)
    self.w_rec_lsg = self.w_module.prog(w_rec_lsg)
    self.w_out_lsg = self.w_module.prog(w_out_lsg)
    
    self.w_in_trainee  = self.w_module.prog(w_in_trainee)
    self.w_rec_trainee = self.w_module.prog(w_rec_trainee)
    self.w_out_trainee = self.w_module.prog(w_out_trainee)
  
  def reprog_weights(self, delta_w_in, delta_w_rec):
    delta_w_in  = Quantize10(delta_w_in,1/16)
    delta_w_rec = Quantize10(delta_w_rec,1/16)
    w_in_trainee  = self.w_module.prog(self.w_in_trainee+delta_w_in)
    w_rec_trainee = self.w_module.prog(self.w_rec_trainee+delta_w_rec)
    return w_in_trainee, w_rec_trainee 
    
      
  #@tf.function
  def call(self, inputs, states):
    state = LIFcell_metaeprop_trainee_LSG_Tuple(out_trainee = states[0], w_in_change = states[1], w_rec_change = states[2], v_trainee = states[3], z_trainee=states[4], r_trainee=states[5], v_lsg = states[6], z_lsg = states[7], r_lsg = states[8], L=states[9], trace_in=states[10], trace_rec=states[11])
    input_trainee = inputs[0]

    w_in_tr_t   = self.w_in_trainee  #+ tf.stop_gradient(self.w_module.read_noise(self.w_in_trainee)) #self.w_module.read(self.w_in_trainee)  
    w_rec_tr_t  = self.w_rec_trainee #+ tf.stop_gradient(self.w_module.read_noise(self.w_rec_trainee)) #self.w_module.read(self.w_rec_trainee) 
    w_out_tr_t  = self.w_out_trainee #+ tf.stop_gradient(self.w_module.read_noise(self.w_out_trainee)) #self.w_module.read(self.w_out_trainee)
    
    w_in_lsg_t  = self.w_in_lsg  #+ tf.stop_gradient(self.w_module.read_noise(self.w_in_lsg)) #self.w_module.read(self.w_in_lsg)  
    w_rec_lsg_t = self.w_rec_lsg #+ tf.stop_gradient(self.w_module.read_noise(self.w_rec_lsg)) #self.w_module.read(self.w_rec_lsg) 
    w_out_lsg_t = self.w_out_lsg #+ tf.stop_gradient(self.w_module.read_noise(self.w_out_lsg))  #self.w_module.read(self.w_out_lsg)    
    
    #trainee
    new_v_trainee, new_z_trainee, new_r_trainee, new_out_trainee = self.LIF_fun_b(input_trainee,state.v_trainee,state.z_trainee,state.r_trainee , state.out_trainee , w_in_tr_t , w_rec_tr_t , w_out_tr_t ,self.thr_trainee,self.decay_trainee, self.kappa_trainee,self.n_refractory_trainee,self.dampening_factor_trainee)
    
    input_lsg = inputs[1]

    #lsg
    new_v_lsg, new_z_lsg, new_r_lsg, new_L = self.LIF_fun(input_lsg,state.v_lsg,state.z_lsg,state.r_lsg, state.L ,w_in_lsg_t, w_rec_lsg_t, w_out_lsg_t ,self.thr_lsg,self.decay_lsg,self.kappa_lsg,self.n_refractory_lsg,self.dampening_factor_lsg)
    
    new_L_scaled_up = tf.einsum("rl, bl-> br",self.con_matrix, new_L) 

    #eprop
    new_w_grad_in , new_w_grad_rec, new_trace_in, new_trace_rec = self.eprop_fun(new_L_scaled_up, state.trace_in, state.trace_rec, new_v_trainee ,new_z_trainee, input_trainee, self.dampening_factor_trainee, self.thr_trainee, self.decay_trainee)
    
    new_w_in_change  = state.w_in_change  - self.lr* new_w_grad_in  #- self.w_in_update * self.grad_to_update(new_w_grad_in, self.lr)
    new_w_rec_change = state.w_rec_change - self.lr* new_w_grad_rec #- self.grad_to_update(new_w_grad_rec, self.lr)
    
    return [new_out_trainee, new_z_trainee ,new_z_lsg,new_v_trainee,new_v_lsg] , [new_out_trainee,new_w_in_change,new_w_rec_change, new_v_trainee, new_z_trainee, new_r_trainee,  new_v_lsg, new_z_lsg, new_r_lsg, new_L, new_trace_in,new_trace_rec]

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))

LIFcell_metaoutweights_trainee_LSG_Tuple = namedtuple('TraineeLSGState', ("out_trainee"
                                                                    ,"w_out_change"
                                                                    ,"v_trainee"
                                                                    ,"z_trainee"
                                                                    ,"r_trainee"
                                                                    ,"v_lsg"
                                                                    ,"z_lsg"
                                                                    ,"r_lsg"
                                                                    ,"L"
                                                                    ,"trace_rec"))




class LIFcell_metaoutweights_trainee_LSG(tf.keras.layers.Layer):
  def __init__(self, n_in_trainee, units_trainee, n_out_trainee
                   , n_in_lsg , units_lsg, n_out_lsg
                   , LIF_par_trainee = LIFParameterTuple(thr=0.4, tau=20.,tau_o=20.,damp=0.3,n_ref=5)
                   , LIF_par_lsg = LIFParameterTuple(thr=0.4, tau=20.,tau_o=20.,damp=0.3,n_ref=5)
                   , lr = 0.01
                   , quantize = None
                   , old = False
                   , stop_w_in_update = False
                   , groups = None
                   , stochupdate = None
                   , noise = False
                   ,**kwargs):
    
    
    self.quantize = quantize is not None
    if quantize is not None:
      self.trainee_LIF_quan = quantize[0]
      self.lsg_LIF_quan = quantize[1]
      self.trainee_eprop_quan = quantize[2]
    else:
      self.trainee_LIF_quan = None
      self.lsg_LIF_quan = None
      self.trainee_eprop_quan = None
        
    self.dampening_factor_trainee = LIF_par_trainee.damp
    self.decay_trainee = tf.exp(-1./LIF_par_trainee.tau)
    self.kappa_trainee = tf.exp(-1./LIF_par_trainee.tau_o)
    
    self.units_trainee = units_trainee
    self.n_refractory_trainee = LIF_par_trainee.n_ref
    self.thr_trainee = LIF_par_trainee.thr
    self.n_out_trainee = n_out_trainee
    self.n_in_trainee = n_in_trainee
    
    self.dampening_factor_lsg = LIF_par_lsg.damp
    self.decay_lsg = tf.exp(-1./LIF_par_lsg.tau)
    self.kappa_lsg = tf.exp(-1./LIF_par_lsg.tau_o)
    
    if stop_w_in_update:
      self.w_in_update = 0.
    else:
      self.w_in_update = 1.
    
    self.units_lsg = units_lsg
    self.n_refractory_lsg = LIF_par_lsg.n_ref
    self.thr_lsg = LIF_par_lsg.thr
    self.n_out_lsg = n_out_lsg
    self.n_in_lsg = n_in_lsg
    
    self.lr = lr
 
    #if noise:
    if noise:
      self.LIF_fun   = LIF_function_noise
      self.LIF_fun_b = LIF_function_batch_noise
      self.eprop_fun = eprop_update
    else:
      self.LIF_fun   = LIF_function
      self.LIF_fun_b = LIF_function_batch
      self.eprop_fun = eprop_update

    print("-"*50)
    
    print(self.LIF_fun.__name__)
    print(self.LIF_fun_b.__name__)
    print(self.eprop_fun.__name__)
    
    print("LIFcell_metaeprop_trainee_LSG parmaeter ")
    print("Quantization: "+str(quantize))
    print("Parameter Trainee: "+str(LIF_par_trainee))
    print("Parameter LSG: "+str(LIF_par_lsg))     
    print("Old setup "+str(old))
    print("eprop_ learning rate "+str(lr))
    print("Noise "+str(noise))
    print("inner learning rate "+str(lr))
    print("-"*50)
    
    if noise:
      self.w_module = PcmNoiseReadProg()
    else:
      self.w_module = ReadProgFull()
    
    self.state_size = [n_out_trainee, tf.constant((n_in_trainee, units_trainee)) , tf.constant((units_trainee,units_trainee)), units_trainee, units_trainee, units_trainee,  units_lsg,units_lsg,units_lsg, n_out_lsg , n_in_trainee, units_trainee]
    
    #[new_out_trainee      ,new_w_in_change,    new_w_rec_change,         new_v_trainee,  new_z_trainee,   new_r_trainee,  new_v_lsg ,   new_z_lsg  , new_r_lsg   , new_L    , new_trace_in   ,new_trace_rec ]
    
    self.output_size = [n_out_trainee, units_trainee, units_lsg]
    
    super(LIFcell_metaoutweights_trainee_LSG, self).__init__(**kwargs)

  def get_connection_matrix_for_output_groups(self, groups):
    coc = np.zeros((self.units_trainee,groups))

    group_size =  self.units_trainee//groups
    
    for i in range(coc.shape[1]):
      begin = i*group_size
      end = begin + group_size
      if (coc.shape[0] - end) < group_size:
        end = coc.shape[0]  
      coc[begin:end,i]= 1

    return tf.constant(coc,dtype = tf.float32)

  def set_init_weights(self, w_in_trainee,w_rec_trainee,w_out_trainee,w_in_lsg, w_rec_lsg, w_out_lsg):
    self.w_in_lsg  = self.w_module.prog(w_in_lsg)
    self.w_rec_lsg = self.w_module.prog(w_rec_lsg)
    self.w_out_lsg = self.w_module.prog(w_out_lsg)
    
    self.w_in_trainee  = self.w_module.prog(w_in_trainee)
    self.w_rec_trainee = self.w_module.prog(w_rec_trainee)
    self.w_out_trainee = self.w_module.prog(w_out_trainee)
  
  def reprog_weights(self, delta_w_in, delta_w_rec):
    delta_w_in  = Quantize10(delta_w_in,1/16)
    delta_w_rec = Quantize10(delta_w_rec,1/16)
    w_in_trainee  = self.w_module.prog(self.w_in_trainee+delta_w_in)
    w_rec_trainee = self.w_module.prog(self.w_rec_trainee+delta_w_rec)
    return w_in_trainee, w_rec_trainee 
  
  def reprog_weights(self, delta_w_out):
    delta_w_out  = Quantize10(delta_w_out,1/16)
    return self.w_module.prog(self.w_out_trainee+delta_w_out)
    
  #@tf.function
  def call(self, inputs, states):
    state = LIFcell_metaeprop_trainee_LSG_Tuple(out_trainee = states[0], w_in_change = states[1], w_rec_change = states[2], v_trainee = states[3], z_trainee=states[4], r_trainee=states[5], v_lsg = states[6], z_lsg = states[7], r_lsg = states[8], L=states[9], trace_in=states[10], trace_rec=states[11])
    input_trainee = inputs[0]

    w_in_tr_t   = self.w_in_trainee  #+ tf.stop_gradient(self.w_module.read_noise(self.w_in_trainee)) #self.w_module.read(self.w_in_trainee)  
    w_rec_tr_t  = self.w_rec_trainee #+ tf.stop_gradient(self.w_module.read_noise(self.w_rec_trainee)) #self.w_module.read(self.w_rec_trainee) 
    w_out_tr_t  = self.w_out_trainee #+ tf.stop_gradient(self.w_module.read_noise(self.w_out_trainee)) #self.w_module.read(self.w_out_trainee)
    
    w_in_lsg_t  = self.w_in_lsg  #+ tf.stop_gradient(self.w_module.read_noise(self.w_in_lsg)) #self.w_module.read(self.w_in_lsg)  
    w_rec_lsg_t = self.w_rec_lsg #+ tf.stop_gradient(self.w_module.read_noise(self.w_rec_lsg)) #self.w_module.read(self.w_rec_lsg) 
    w_out_lsg_t = self.w_out_lsg #+ tf.stop_gradient(self.w_module.read_noise(self.w_out_lsg))  #self.w_module.read(self.w_out_lsg)    
    
    #trainee
    new_v_trainee, new_z_trainee, new_r_trainee, new_out_trainee = self.LIF_fun_b(input_trainee,state.v_trainee,state.z_trainee,state.r_trainee , state.out_trainee , w_in_tr_t , w_rec_tr_t , w_out_tr_t ,self.thr_trainee,self.decay_trainee, self.kappa_trainee,self.n_refractory_trainee,self.dampening_factor_trainee)
    
    input_lsg = inputs[1]

    #lsg
    new_v_lsg, new_z_lsg, new_r_lsg, new_L = self.LIF_fun(input_lsg,state.v_lsg,state.z_lsg,state.r_lsg, state.L ,w_in_lsg_t, w_rec_lsg_t, w_out_lsg_t ,self.thr_lsg,self.decay_lsg,self.kappa_lsg,self.n_refractory_lsg,self.dampening_factor_lsg)
    
    new_L_scaled_up = tf.einsum("rl, bl-> br",self.con_matrix, new_L) 

    #eprop
    #new_w_grad_in , new_w_grad_rec, new_trace_in, new_trace_rec = self.eprop_fun(new_L_scaled_up, state.trace_in, state.trace_rec, new_v_trainee ,new_z_trainee, input_trainee, self.dampening_factor_trainee, self.thr_trainee, self.decay_trainee)
    new_w_grad_out, trace_rec = outprop_update(L, state.trace_rec, new_z_trainee, self.decay_trainee)
    
    new_w_out_change  = state.w_out_change  - self.lr* new_w_grad_out
    
    #new_w_in_change  = state.w_in_change  - self.lr* new_w_grad_in  #- self.w_in_update * self.grad_to_update(new_w_grad_in, self.lr)
    #new_w_rec_change = state.w_rec_change - self.lr* new_w_grad_rec #- self.grad_to_update(new_w_grad_rec, self.lr)
    
    return [new_out_trainee, new_z_trainee ,new_z_lsg,new_v_trainee,new_v_lsg] , [new_out_trainee,new_w_out_change, new_v_trainee, new_z_trainee, new_r_trainee,  new_v_lsg, new_z_lsg, new_r_lsg, new_L, new_trace_in,new_trace_rec]

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))





#if self.stochupdate is None:
#      new_w_in_change  =    state.w_in_change  - self.lr * new_w_grad_in  
#      new_w_rec_change =    state.w_rec_change - self.lr * new_w_grad_rec
#    else:
#_w_in_change  =   state.w_in_change   - update_mask * self.lr * new_w_grad_in  
#new_w_rec_change =    state.w_rec_change - update_mask * self.lr * new_w_grad_rec
    
# h = tf.maximum(self.dampening_factor_trainee*( 1.- tf.abs((state.z_trainee-self.thr_trainee)/self.thr_trainee)), 0.)/self.thr_trainee
#    new_trace_in = state.trace_in*self.decay_trainee + input_trainee
#    new_trace_rec = state.trace_rec*self.decay_trainee + state.z_trainee
#    
#    new_w_in_change  = state.w_in_change -self.lr*tf.einsum("br,bi->bir",h*new_L, new_trace_in)
#    new_w_rec_change = state.w_rec_change -self.lr*tf.einsum("br,bi->bir",h*new_L, new_trace_rec)

RSNNOutputStateTuple = namedtuple('RNNNOutputState', (
    'z',
    'v',
    'r',
    'out'
))

class LIFcellWeightBatchOutput(tf.keras.layers.Layer):
  def __init__(self,units, n_out ,noise = False ,LIF_par = LIFParameterTuple(thr=0.4, tau=20.,tau_o=20.,damp=0.3,n_ref=5),quantize = None  ,**kwargs):
    
    self.dampening_factor = LIF_par.damp
    self.decay = tf.exp(-1./LIF_par.tau)
    self.kappa = tf.exp(-1./LIF_par.tau_o)
    self.units = units
    self.n_refractory = LIF_par.n_ref
    self.thr = LIF_par.thr
    self.n_out = n_out
    
    self.state_size = [units, units, units, n_out]
    self.output_size =[n_out,units]   #maybe shift somewhere else


    if quantize is not None:
      self.trainee_LIF_quan = quantize
    else:
      self.trainee_LIF_quan = None
    
    if noise:
      self.LIF_fun_b = LIF_function_batch_noise
    else:
      self.LIF_fun_b = LIF_function_batch
    
    super(LIFcellWeightBatchOutput, self).__init__(**kwargs)

  def set_weights(self, w_in, w_rec,w_out):
    self.w_in  = w_in
    self.w_rec = w_rec
    self.w_out = w_out
    
  #@tf.function
  def call(self, inputs, states):
    state = RSNNOutputStateTuple(v=states[0], z=states[1], r = states[2], out = states[3]) 
    new_v, new_z, new_r, new_out = self.LIF_fun_b(inputs,state.v,state.z,state.r,state.out,self.w_in, self.w_rec,self.w_out, self.thr,self.decay,self.kappa,self.n_refractory,self.dampening_factor)
    
    return [new_out, new_z], [new_v, new_z, new_r, new_out]

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))


class LIFcell(tf.keras.layers.Layer):
  def __init__(self,units,decay,n_refractory=0 ,thr=0.4, dtype=tf.float32, dampening_factor=0.3, **kwargs):
    self.dampening_factor = dampening_factor
    self.data_type = dtype  
    self._decay = decay
    self.units = units
    self.n_refractory = n_refractory
    self.thr = thr
    self.state_size = [units, units, units]
    self.output_size =[units, units]   #maybe shift somewhere else
    
    super(LIFcell, self).__init__(**kwargs)

  def set_weights(self, w_in, w_rec):
    self.w_in  = w_in
    self.w_rec = w_rec
    
  def get_weights(self):
    return [self.w_in, self.w_rec]
    
  @tf.function
  def call(self, inputs, states):
    state = RSNNStateTuple(v=states[0], z=states[1], r = states[2]) 
    new_v = self._decay * state.v +  ( tf.matmul(inputs, self.w_in) + tf.matmul(state.z, (1.-tf.eye(self.units,self.units)) * self.w_rec) ) - state.z * self.thr
    
    new_z = tf.where(tf.greater(state.r, .1), tf.zeros_like(state.z), SpikeFunction(new_v,self.thr,self.dampening_factor))
    new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                             0., float(self.n_refractory))
    return [new_z , new_v] , [new_v, new_z, new_r]

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))


class LIFcellWeightperBatch(tf.keras.layers.Layer):
  def __init__(self,units,decay, n_refractory=0 ,thr=0.4, dampening_factor=0.3, **kwargs):
    self.dampening_factor = dampening_factor  
    self._decay = decay
    self.units = units
    self.n_refractory = n_refractory
    self.thr = thr
    self.state_size = [units, units, units]
    self.output_size =[units, units]   #maybe shift somewhere else
    
    super(LIFcellWeightperBatch, self).__init__(**kwargs)

  def set_weights(self, w_in, w_rec):
    self.w_in  = w_in
    self.w_rec = w_rec
    
  def get_weights(self):
    return [self.w_in, self.w_rec]
    
  @tf.function
  def call(self, inputs, states):
    state = RSNNStateTuple(v=states[0], z=states[1], r = states[2]) 
    new_v = self._decay * state.v +  ( tf.einsum("bi,bir-> br",inputs,self.w_in) + tf.einsum("br, brj -> bj",state.z, (1.-tf.eye(self.units,self.units)) * self.w_rec) ) - state.z * self.thr
    
    new_z = tf.where(tf.greater(state.r, .1), tf.zeros_like(state.z), SpikeFunction(new_v,self.thr,self.dampening_factor))
    new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                             0., float(self.n_refractory))
    return [new_z , new_v] , [new_v, new_z, new_r]

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))



