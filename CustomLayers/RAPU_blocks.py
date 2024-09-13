import tensorflow as tf 
from tensorflow.keras.layers import BatchNormalization, add, Conv2D, UpSampling2D, Resizing

kernel_initializer = 'he_uniform'

def RAPU(x, filters): 
    x  = BatchNormalization(axis=-1)(x)
    
    x1 = atrous_block(x, filters)
    x2 = resnet_block(x, filters)
    
    x  = add([x1, x2])
    x  = BatchNormalization(axis=-1)(x)

    return x
    
def resnet_block(x, filters,dilation_rate=1):
    x1 = Conv2D(filters, (1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same',
                dilation_rate=dilation_rate)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x_final = add([x, x1])

    x_final = BatchNormalization(axis=-1)(x_final)

    return x_final    
    
def atrous_block(x, filters):
    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=1)(x)

    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=2)(x)

    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same',
               dilation_rate=3)(x)

    x = BatchNormalization(axis=-1)(x)

    return x
    
def convf_bn_act(inputs, filters, kernel_size, strides=(1, 1), activation='relu', padding='same'):
    
    x = Conv2D(filters, kernel_size=kernel_size, strides = strides, padding=padding, use_bias = False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    #x = bn_act(x, activation=activation)
    
    return x    
    
  
    
def SBA(L_input, H_input):
    dim = 16
    
    L_input = Conv2D(dim, 1, padding='same', use_bias=False)(L_input) 
    H_input = Conv2D(dim, 1, padding='same', use_bias=False)(H_input)    
      
    g_L = tf.keras.layers.Activation('sigmoid')(L_input)
    g_H = tf.keras.layers.Activation('sigmoid')(H_input)
    
    L_input = convf_bn_act(L_input,dim,1) 
    H_input = convf_bn_act(H_input,dim,1)   
    
    
    L_feature = L_input + L_input * g_L + (1 - g_L) * UpSampling2D((2,2))(g_H * H_input)
    H_feature = H_input + H_input * g_H + (1 - g_H) * Resizing(H_input.shape[1], H_input.shape[2])(g_L * L_input)
    
    H_feature = UpSampling2D((2,2))(H_feature)
    out = tf.keras.layers.Concatenate(axis=-1)([L_feature, H_feature])
    
    out = convf_bn_act(out, dim*2, 3)
    out = Conv2D(1, 1, use_bias=False)(out)
    
    return out 
