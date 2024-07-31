
import tensorflow as tf
import numpy as np
import time
import datetime
import pickle
import json
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

#print(device_lib.list_local_devices("GPU"))



# x1=np.arange(0,10,1000)
# xt1 = tf.constant(x1)
# xt2 = tf.Variable(x1)
# with tf.device('/GPU:0'):
# 	print("GPU working!")
# 	for i in range(100000):
# 		xt2 = xt1*2
# 		xt2 = xt1**2
# 		xt2 = xt1 + xt1	
# 	pass

