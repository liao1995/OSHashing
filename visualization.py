import scipy.io as sio
import numpy as np
from PIL import Image

ep = sio.loadmat('end_points.mat')
pool1 = ep['osvos/pool1']
pool1 = np.squeeze(pool1)
pool1_res = list()

for i in range(8):
  pool1_res.append(pool1[:,:,i*8])

for i in range(8):
  for j in range(1,8):
    pool1_res[i] = np.column_stack((pool1_res[i], pool1[:,:,i*8+j]))

for i in range(1,8):
  pool1_res[0] = np.row_stack((pool1_res[0], pool1_res[i]))

im = Image.fromarray(pool1_res[0])
im.show()
