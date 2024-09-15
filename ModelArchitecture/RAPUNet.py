import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
from keras.layers import add
from keras.models import Model
from keras_cv_attention_models import caformer
from CustomLayers.RAPU_blocks import resnet_block, RAPU, convf_bn_act, SBA 

kernel_initializer = 'he_uniform'
interpolation = "nearest"


def create_model(img_height, img_width, input_chanels, out_classes, starting_filters):
    backbone = caformer.CAFormerS18(input_shape=(352, 352, 3), pretrained="imagenet", num_classes = 0)
    layer_names = ['stack4_block3_mlp_Dense_1', 'stack3_block9_mlp_Dense_1', 'stack2_block3_mlp_Dense_1', 'stack1_block3_mlp_Dense_1']
    layers =[backbone.get_layer(x).output for x in layer_names]
    
    input_layer = backbone.input
    print('Starting RAPUNet')

    p1 = Conv2D(starting_filters * 2, 3, strides=2, padding='same')(input_layer)  
  
    #from metaformer
    p2 = Conv2D(starting_filters*4,1,padding='same')(layers[3]) #88,88
    p3 = Conv2D(starting_filters*8,1,padding='same')(layers[2]) #44,44
    p4 = Conv2D(starting_filters*16,1,padding='same')(layers[1]) #22,22     
    p5 = Conv2D(starting_filters*32,1,padding='same')(layers[0]) #11,11
    
    
    t0 = RAPU(input_layer, starting_filters)  #352, 352

    l1i = Conv2D(starting_filters * 2, 2, strides=2, padding='same')(t0)    
    s1 = add([l1i, p1])     
    t1 = RAPU(s1, starting_filters * 2)  #176,176      
   
    
    l2i = Conv2D(starting_filters * 4, 2, strides=2, padding='same')(t1)
    s2 = add([l2i, p2])
    t2 = RAPU(s2, starting_filters * 4) #88,88
    

    l3i = Conv2D(starting_filters * 8, 2, strides=2, padding='same')(t2)
    s3 = add([l3i, p3])
    t3 = RAPU(s3, starting_filters * 8) #44,44
   
    l4i = Conv2D(starting_filters * 16, 2, strides=2, padding='same')(t3)
    s4 = add([l4i, p4])
    t4 = RAPU(s4, starting_filters * 16) #22,22
    
    l5i = Conv2D(starting_filters * 32, 2, strides=2, padding='same')(t4)
    s5 = add([l5i, p5]) 
    
    t51 = resnet_block(s5, starting_filters*32)
    t51 = resnet_block(t51, starting_filters*32) 
    t53 = resnet_block(t51, starting_filters*16) #11,11
    t53 = resnet_block(t53, starting_filters*16) #11,11
        
    #Aggregation

    dim =32
    
    outd = tf.keras.layers.Concatenate(axis=-1)([UpSampling2D((4,4))(t53), UpSampling2D((2,2))(t4)])
    outd = tf.keras.layers.Concatenate(axis=-1)([outd, t3]) #44,44 
    outd = convf_bn_act(outd, dim, 1)
    outd = Conv2D(1, kernel_size=1, use_bias=False)(outd)  #output1 44,44,1
    
    L_input = convf_bn_act(t2,dim, 3) #88,88,32   
    
    H_input = tf.keras.layers.Concatenate(axis=-1)([UpSampling2D((2,2))(t53), t4])
    H_input = convf_bn_act(H_input, dim,1)
    H_input = UpSampling2D((2,2))(H_input)  #44,44,32
    
    out2 = SBA(L_input, H_input) #output 88,88,1   
         
    out1 = UpSampling2D(size=(8,8), interpolation='bilinear')(outd)
    out2 = UpSampling2D(size=(4,4), interpolation='bilinear')(out2)
        
    out_duat = out1+out2 
    output = Conv2D(out_classes, (1, 1), activation='sigmoid')(out_duat)
    
    model = Model(inputs=input_layer, outputs=output)

    return model
