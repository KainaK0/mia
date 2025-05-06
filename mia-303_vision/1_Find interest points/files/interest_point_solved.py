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
import scipy.ndimage.filters as filters
from scipy.ndimage import map_coordinates
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from . import im_util

class InterestPointExtractor:
  """
  Class to extract interest points from an image
  """
  def __init__(self):
    self.params={}
    self.params['border_pixels']=10
    self.params['strength_threshold_percentile']=95
    self.params['supression_radius_frac']=0.01

  def find_interest_points(self, img):
    """
    Find interest points in greyscale image img

    Inputs: img=greyscale input image (H, W, 1)

    Outputs: ip=interest points of shape (2, N)
    """
    ip_fun = self.corner_function(img)
    row, col = self.find_local_maxima(ip_fun)

    ip = np.stack((row,col))
    return ip

  def corner_function(self, img):
    """
    Compute corner strength function in image im

    Inputs: img=grayscale input image (H, W, 1)

    Outputs: ip_fun=interest point strength function (H, W, 1)
    """

    H, W, _ = img.shape

    # FORNOW: random interest point function
    ip_fun = np.random.randn(H, W, 1)

    """
    **********************************************************
    *** TODO: write code to compute a corner strength function
    **********************************************************
    """

    Ix, Iy = im_util.compute_gradients(img)
    window_size = 5
    sum_kernel = np.ones((window_size, window_size))
    Ix2 = im_util.convolve(Ix**2, sum_kernel)
    Iy2 = im_util.convolve(Iy**2, sum_kernel)
    IxIy = im_util.convolve(Ix*Iy, sum_kernel)
    det = Ix2 * Iy2 - IxIy**2
    trace = Ix2 + Iy2
    ip_fun = det - 0.04 * trace**2

    """
    **********************************************************
    """

    return ip_fun

  def find_local_maxima(self, ip_fun):
    """
    Find local maxima in interest point strength function

    Inputs: ip_fun=corner strength function (H, W, 1)

    Outputs: row,col=coordinates of interest points
    """

    H, W, _ = ip_fun.shape

    # radius for non-maximal suppression
    suppression_radius_pixels = int(self.params['supression_radius_frac']*max(H, W))

    # minimum of strength function for corners
    strength_threshold=np.percentile(ip_fun, self.params['strength_threshold_percentile'])

    # don't return interest points within border_pixels of edge
    border_pixels = self.params['border_pixels']

    # row and column coordinates of interest points
    row = []
    col = []

    # FORNOW: random row and column coordinates
    row = np.random.randint(0,H,100)
    col = np.random.randint(0,W,100)

    """
    ***************************************************
    *** TODO: write code to find local maxima in ip_fun
    ***************************************************

    Hint: try scipy filters.maximum_filter with im_util.disc_mask
    """

    max_ip_fun = filters.maximum_filter(ip_fun, footprint=im_util.disc_mask(suppression_radius_pixels))
    maxima = (ip_fun == max_ip_fun) & (ip_fun > strength_threshold)
    border_mask = np.zeros_like(maxima)
    border_mask[border_pixels:H-border_pixels, border_pixels:W-border_pixels, :] = True
    row, col, _ = np.where(maxima & border_mask)

    """
    ***************************************************
    """

    return row, col

class DescriptorExtractor:
  """
  Extract descriptors around interest points
  """
  def __init__(self):
    self.params={}
    self.params['patch_size']=16
    self.params['ratio_threshold']=0.75

  def get_descriptors(self, img, ip):
    """
    Extact descriptors from grayscale image img at interest points ip

    Inputs: img=grayscale input image (H, W, 1)
            ip=interest point coordinates (2, N)

    Returns: descriptors=vectorized descriptors (N, num_dims)
    """
    patch_size=self.params['patch_size']
    patch_size_div2=int(patch_size/2)
    num_dims=patch_size**2

    H,W,_=img.shape
    num_ip=ip.shape[1]
    descriptors=np.zeros((num_ip,num_dims))


    for i in range(num_ip):
      row=ip[0,i]
      col=ip[1,i]

      # FORNOW: random image patch
      patch=np.random.randn(patch_size,patch_size)

      """
      ******************************************************
      *** TODO: write code to extract descriptor at row, col
      ******************************************************
      """

      for j in range(patch_size):
        for k in range(patch_size):
          patch[j,k] = img[row+j-patch_size_div2, col+k-patch_size_div2]

      """
      ******************************************************
      """

      descriptors[i, :]=np.reshape(patch,num_dims)

    # normalise descriptors to 0 mean, unit length
    mn=np.mean(descriptors,1,keepdims=True)
    sd=np.std(descriptors,1,keepdims=True)
    small_val = 1e-6
    descriptors = (descriptors-mn)/(sd+small_val)

    return descriptors

  def compute_distances(self, desc1, desc2):
    """
    Compute distances between descriptors

    Inputs: desc1=descriptor array (N1, num_dims)
            desc2=descriptor array (N2, num_dims)

    Returns: dists=array of distances (N1,N2)
    """
    N1,num_dims=desc1.shape
    N2,num_dims=desc2.shape

    ATB=np.dot(desc1,desc2.T)
    AA=np.sum(desc1*desc1,1)
    BB=np.sum(desc2*desc2,1)

    dists=-2*ATB+np.expand_dims(AA,1)+BB

    return dists

  def match_descriptors(self, desc1, desc2):
    """
    Find nearest neighbour matches between descriptors

    Inputs: desc1=descriptor array (N1, num_dims)
            desc2=descriptor array (N2, num_dims)

    Returns: match_idx=nearest neighbour index for each desc1 (N1)
    """
    dists=self.compute_distances(desc1, desc2)

    match_idx=np.argmin(dists,1)

    return match_idx

  def match_ratio_test(self, desc1, desc2):
    """
    Find nearest neighbour matches between descriptors
    and perform ratio test

    Inputs: desc1=descriptor array (N1, num_dims)
            desc2=descriptor array (N2, num_dims)

    Returns: match_idx=nearest neighbour inded for each desc1 (N1)
             ratio_pass=whether each match passes ratio test (N1)
    """
    N1,num_dims=desc1.shape

    dists=self.compute_distances(desc1, desc2)

    sort_idx=np.argsort(dists,1)

    #match_idx=np.argmin(dists,1)
    match_idx=sort_idx[:,0]

    d1NN=dists[np.arange(0,N1),sort_idx[:,0]]
    d2NN=dists[np.arange(0,N1),sort_idx[:,1]]

    ratio_threshold=self.params['ratio_threshold']
    ratio_pass=(d1NN<ratio_threshold*d2NN)

    return match_idx,ratio_pass

def draw_interest_points_ax(ip, ax):
  """
  Draw interest points ip on axis ax
  """
  for row,col in zip(ip[0,:],ip[1,:]):
    circ1 = Circle((col,row), 5)
    circ1.set_color('black')
    circ2 = Circle((col,row), 3)
    circ2.set_color('white')
    ax.add_patch(circ1)
    ax.add_patch(circ2)

def draw_interest_points_file(im, ip, filename):
  """
  Draw interest points ip on image im and save to filename
  """
  fig,ax = im_util.image_figure(im)
  draw_interest_points_ax(ip, ax)
  fig.savefig(filename)
  plt.close(fig)

def draw_matches_ax(ip1, ipm, ax1, ax2):
  """
  Draw matches ip1, ipm on axes ax1, ax2
  """
  for r1,c1,r2,c2 in zip(ip1[0,:], ip1[1,:], ipm[0,:], ipm[1,:]):
    rand_colour=np.random.rand(3,)

    circ1 = Circle((c1,r1), 5)
    circ1.set_color('black')
    circ2 = Circle((c1,r1), 3)
    circ2.set_color(rand_colour)
    ax1.add_patch(circ1)
    ax1.add_patch(circ2)

    circ3 = Circle((c2,r2), 5)
    circ3.set_color('black')
    circ4 = Circle((c2,r2), 3)
    circ4.set_color(rand_colour)
    ax2.add_patch(circ3)
    ax2.add_patch(circ4)

def draw_matches_file(im1, im2, ip1, ipm, filename):
  """
  Draw matches ip1, ipm on images im1, im2 and save to filename
  """
  H1,W1,B1=im1.shape
  H2,W2,B2=im2.shape

  im3 = np.zeros((max(H1,H2),W1+W2,3))
  im3[0:H1,0:W1,:]=im1
  im3[0:H2,W1:(W1+W2),:]=im2

  fig,ax = im_util.image_figure(im3)
  col_offset=W1

  for r1,c1,r2,c2 in zip(ip1[0,:], ip1[1,:], ipm[0,:], ipm[1,:]):
    rand_colour=np.random.rand(3,)

    circ1 = Circle((c1,r1), 5)
    circ1.set_color('black')
    circ2 = Circle((c1,r1), 3)
    circ2.set_color(rand_colour)
    ax.add_patch(circ1)
    ax.add_patch(circ2)

    circ3 = Circle((c2+col_offset,r2), 5)
    circ3.set_color('black')
    circ4 = Circle((c2+col_offset,r2), 3)
    circ4.set_color(rand_colour)
    ax.add_patch(circ3)
    ax.add_patch(circ4)

  fig.savefig(filename)
  plt.close(fig)

def plot_descriptors(desc,plt):
  """
  Plot a random set of descriptor patches
  """
  num_ip,num_dims = desc.shape
  patch_size = int(np.sqrt(num_dims))

  N1,N2=2,8
  figsize0=plt.rcParams['figure.figsize']
  plt.rcParams['figure.figsize'] = (16.0, 4.0)
  for i in range(N1):
    for j in range(N2):
      ax=plt.subplot(N1,N2,i*N2+j+1)
      rnd=np.random.randint(0,num_ip)
      desc_im=np.reshape(desc[rnd,:],(patch_size,patch_size))
      plt.imshow(im_util.grey_to_rgb(im_util.normalise_01(desc_im)))
      plt.axis('off')

  plt.rcParams['figure.figsize']=figsize0

def plot_matching_descriptors(desc1,desc2,desc1_id,desc2_id,plt):
  """
  Plot a random set of matching descriptor patches
  """
  num_inliers=desc1_id.size
  num_ip,num_dims = desc1.shape
  patch_size=int(np.sqrt(num_dims))

  figsize0=plt.rcParams['figure.figsize']

  N1,N2=1,8
  plt.rcParams['figure.figsize'] = (16.0, N1*4.0)

  for i in range(N1):
    for j in range(N2):
      rnd=np.random.randint(0,num_inliers)

      desc1_rnd=desc1_id[rnd]
      desc2_rnd=desc2_id[rnd]

      desc1_im=np.reshape(desc1[desc1_rnd,:],(patch_size,patch_size))
      desc2_im=np.reshape(desc2[desc2_rnd,:],(patch_size,patch_size))

      ax=plt.subplot(2*N1,N2,2*i*N2+j+1)
      plt.imshow(im_util.grey_to_rgb(im_util.normalise_01(desc1_im)))
      plt.axis('off')
      ax=plt.subplot(2*N1,N2,2*i*N2+N2+j+1)
      plt.imshow(im_util.grey_to_rgb(im_util.normalise_01(desc2_im)))
      plt.axis('off')

  plt.rcParams['figure.figsize'] = figsize0