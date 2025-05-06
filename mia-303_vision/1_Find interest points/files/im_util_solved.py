# Copyright 2017 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import PIL.Image as pil
import scipy.signal as sps
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

def convolve_1d(x, k):
  """
  Convolve vector x with kernel k

  Inputs: x=input vector (Nx)
          k=input kernel (Nk)

  Outputs: y=output vector (Nx)
  """
  y=np.zeros_like(x)

  """
  *******************************************
  *** TODO: write code to perform convolution
  *******************************************

  The output should be the same size as the input
  You can assume zero padding, and an odd-sized kernel
  """

  Nx = len(x)
  Nk = len(k)
  y = np.zeros_like(x)
  pad_width = Nk // 2
  
  for i in range(Nx):
    for j in range(Nk):
      if i + j - pad_width >= 0 and i + j - pad_width < Nx:
        y[i] += x[i+j-pad_width] * k[j]

  """
  *******************************************
  """

  return y

def convolve_rows(im, k):
  """
  Convolve image im with kernel k

  Inputs: im=input image (H, W, B)
          k=1D convolution kernel (N)

  Outputs: im_out=output image (H, W, B)
  """
  im_out = np.zeros_like(im)

  """
  *****************************************
  *** TODO: write code to convolve an image
  *****************************************

  Convolve the rows of image im with kernel k
  The output should be the same size as the input
  You can assume zero padding, and an odd-sized kernel
  """

  H, W, B = im.shape
  N = len(k)
  for i in range(H):
    for j in range(B):
      im_out[i, :, j] = convolve_1d(im[i, :, j], k)


  """
  *****************************************
  """

  return im_out

def gauss_kernel(sigma):
  """
  1D Gauss kernel of standard deviation sigma
  """
  l = int(np.ceil(2 * sigma))
  x = np.linspace(-l, l, 2*l+1)


  """
  *******************************************
  *** TODO: compute gaussian kernel at each x
  *******************************************
  """

  gx = np.exp(-x**2/(2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)

  """
  *******************************************
  """

  gx = np.expand_dims(gx,0)
  return gx

def convolve_gaussian(im, sigma):
  """
  2D gaussian convolution
  """
  imc=np.zeros_like(im)

  """
  ***************************************
  *** TODO separable gaussian convolution
  ***************************************
  """

  gx = gauss_kernel(sigma)
  imc = convolve(im, gx)

  """
  ***************************************
  """
  return imc

def compute_gradients(img):

  Ix=np.zeros_like(img)
  Iy=np.zeros_like(img)

  """
  ***********************************************
  *** TODO: write code to compute image gradients
  ***********************************************
  """

  kernel_gx = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]]
  )
  kernel_gy = np.array(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]]
  )
  Ix = convolve(img, kernel_gx)
  Iy = convolve(img, kernel_gy)

  """
  ***********************************************
  """
  return Ix, Iy

def image_open(filename):
  """
  Returns a numpy float image with values in the range (0,1)
  """
  pil_im = pil.open(filename)
  im_np = np.array(pil_im).astype(np.float32)
  im_np /= 255.0
  return im_np

def image_save(im_np, filename):
  """
  Saves a numpy float image to file
  """
  if (len(im_np.shape)==2):
    im_np = np.expand_dims(im_np, 2)
  if (im_np.shape[2]==1):
    im_np= np.repeat(im_np, 3, axis=2)
  im_np = np.maximum(0.0, np.minimum(im_np, 1.0))
  pil_im = pil.fromarray((im_np*255).astype(np.uint8))
  pil_im.save(filename)

def image_figure(im, dpi=100):
  """
  Creates a matplotlib figure around an image,
  useful for writing to file with savefig()
  """
  H,W,_=im.shape
  fig=plt.figure()
  fig.set_size_inches(W/dpi, H/dpi)
  ax=fig.add_axes([0,0,1,1])
  ax.imshow(im)
  return fig, ax

def plot_two_images(im1, im2):
  """
  Plot two images and return axis handles
  """
  ax1=plt.subplot(1,2,1)
  plt.imshow(im1)
  plt.axis('off')
  ax2=plt.subplot(1,2,2)
  plt.imshow(im2)
  plt.axis('off')
  return ax1, ax2

def normalise_01(im):
  """
  Normalise image to the range (0,1)
  """
  mx = im.max()
  mn = im.min()
  den = mx-mn
  small_val = 1e-9
  if (den < small_val):
    print('image normalise_01 -- divisor is very small')
    den = small_val
  return (im-mn)/den

def grey_to_rgb(img):
  """
  Convert greyscale to rgb image
  """
  if (len(img.shape)==2):
    img = np.expand_dims(img, 2)

  img3 = np.repeat(img, 3, 2)
  return img3

def disc_mask(l):
  """
  Create a binary cirular mask of radius l
  """
  sz = 2 * l + 1
  m = np.zeros((sz,sz))
  x = np.linspace(-l,l,2*l+1)/l
  x = np.expand_dims(x, 1)
  m = x**2
  m = m + m.T
  m = m<1
  m = np.expand_dims(m, 2)
  return m

def convolve(im, kernel):
  """
  Wrapper for scipy convolution function
  This implements a general 2D convolution of image im with kernel
  Note that strictly speaking this is correlation not convolution

  Inputs: im=input image (H, W, B) or (H, W)
          kernel=kernel (kH, kW)

  Outputs: imc=output image (H, W, B)
  """
  if (len(im.shape)==2):
    im = np.expand_dims(im, 2)
  H, W, B = im.shape
  imc = np.zeros((H, W, B))
  for band in range(B):
    imc[:, :, band] = sps.correlate2d(im[:, :, band], kernel, mode='same')
  return imc

def coordinate_image(num_rows,num_cols,r0,r1,c0,c1):
  """
  Creates an image size num_rows, num_cols
  with coordinates linearly spaced in from r0->r1 and c0->c1
  """
  rval=np.linspace(r0,r1,num_rows)
  cval=np.linspace(c0,c1,num_cols)
  c,r=np.meshgrid(cval,rval)
  M = np.stack([r,c,np.ones(r.shape)],-1)
  return M

def transform_coordinates(coord_image, M):
  """
  Transform an image containing row,col,1 coordinates by matrix M
  """
  M=np.expand_dims(M,2)
  uh=np.dot(coord_image,M.T)
  uh=uh[:, :, 0, :]
  uh=uh/np.expand_dims(uh[:, :, 2],2)
  return uh

def warp_image(im, coords):
  """
  Warp image im using row,col,1 image coords
  """
  im_rows,im_cols,im_bands=im.shape
  warp_rows,warp_cols,_=coords.shape
  map_coords=np.zeros((3,warp_rows,warp_cols,im_bands))
  for b in range(im_bands):
    map_coords[0,:,:,b]=coords[:,:,0]
    map_coords[1,:,:,b]=coords[:,:,1]
    map_coords[2,:,:,b]=b
  warp_im = map_coordinates(im, map_coords, order=1)
  return warp_im