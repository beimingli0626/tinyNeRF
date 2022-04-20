import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction
import numpy as np
import matplotlib.pyplot as plt


def get_rays(H: int, W: int, F: float, cam2world: torch.Tensor):
    r"""Compute rays passing through all pixels of an image

    Args:
      H: Height of an image
      W: Width of an image
      G: Focal length of camera
      cam2world: A 6-DoF rigid-body transform (4x4 matrix) that transforms a 3D point 
      from the camera frame to the world frame (in homogeneous coordinate)
    
    Returns:
      ray_origins: A tensor of shape (H, W, 3) denoting the centers of each ray. 
        ray_origins[i][j] denotes the origin of the ray passing through pixel at
        row index `i` and column index `j`.
      ray_directions: A tensor of shape (H, W, 3) denoting the direction of each 
        ray. ray_directions[i][j] denotes the direction (x, y, z) of the ray 
        passing through the pixel at row index `i` and column index `j`.
    
    Note:
      We use homogeneous coordinate for transformation. Each point in the 3D space
      could be represent as [x, y, z, 1]. With cam2world @ [x, y, z, 1].T, we
      are transfomring that point from camera fram to the world frame.
      Besides, the origin point of the camera frame is [0, 0, 0, 1]. we could 
      get world-frame coordinate of that origin with cam2world @ [0, 0, 0, 1].T,
      which turns out to be cam2world[:3, -1].
    """
    ray_origins, ray_directions = None, None

    # 'i' is the x axis of points, all columns are identical
    # 'j' is the y axis of points, all rows are identical 
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')
    i = i.to(cam2world) # (H, W)
    j = j.to(cam2world) # (H, W)

    # Calculate the direction w.r.t the camera pinhole, whose x, y coordinates are
    # the same as the center of image (W/2, H/2), and z coordinate is negative 
    # focal length. You can image constructing a 3D coordinate, whose origin is
    # at the center of the image.
    # We simply normalize the direction with focal length here, so that coordinates 
    # value lie in [-1, 1], which is beneficial for numerical stability.
    directions = torch.stack([(i - W * .5) / F,
                              -(j - H * .5) / F,
                              -torch.ones_like(i)] , dim=-1) # (H, W, 3)

    # Apply transformation to the direction, f(d) = Ad = dA^(T)
    ray_directions = directions @ cam2world[:3, :3].t() # (H, W, 3)
    
    # All the rays share the same origin
    ray_origins = cam2world[:3, -1].expand(ray_directions.shape) # (H, W, 3)
    return ray_origins, ray_directions


def sample_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_point: float,
    far_point: float,
    num_samples: int,
    random: Optional[bool] = True
) -> Tuple[torch.Tensor]:
    r"""Compute a set of 3D points given the bundle of rays. The near_thresh and far_thresh
    variables indicate the bounds within which 3D points are to be sampled.

    Reference: https://github.com/krrish94/nerf-pytorch

    Args:
      ray_origins: origin of each ray in the bundle as returned by the 
        get_ray_bundle() function, shape (H, W, 3)
      ray_directions: direction of each ray in the bundle as returned by the
        get_ray_bundle() funciton, shape (H, W, 3)
      near_point: depth of nearest point that we are interested in
      far_point: depth of farthest point that we are interested in
      num_samples: number of samples to be sampled along each ray. Samples are drawn
        randomly, while trying to ensure some form of uniform spacing among them
      random: whether or not to randomly the sampling of query points.
        If random=False, 3D points are sampled uniformly along each ray
    
    Returns:
      sampled_points: axis of the sampled points along each ray, shape (H, W, num_samples, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)
    """
    H, W, _ = ray_origins.shape

    # Uniformly sample depth values, (H, W, num_samples)
    depth_values = torch.linspace(near_point, far_point, num_samples).to(ray_origins)
    depth_values = depth_values.expand(H, W, num_samples)

    # If we want to randomly sampled along each ray, we add some noise to the uniformly
    # sampled depths, in order to make sure certain uniformity among them, notice that
    # we randomize for each ray (the noises are not the same for all the rays)
    if random:
      # We randomly sample noice in [0, 1), shape (H, W, num_samples)
      noise = torch.rand(H, W, num_samples).to(ray_origins)

      # Normalize the noise by uniform distance among samples
      noise = noise * (far_point - near_point) / num_samples

      # Add noise to depth value, get shape (H, W, num_samples)
      depth_values = depth_values + noise

    # Note: ray_directions all have different lengths, but are all close to 1,
    #       we don't transfer them to unit vector for simplicity (?)
    # (H, W, num_samples, 3) = (H, W, 1, 3) + (H, W, 1, 3) * (H, W, num_samples, 1)
    sampled_points = ray_origins[..., None, :] + ray_directions[..., None, :] \
                      * depth_values[..., :, None]
    return sampled_points, depth_values


def positional_encoding(
    pos_in, freq=32, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input. (Section 5.1 of original paper)
    We use positional encoding to map continuous input coordinates into a 
    higher dimensional space to enable our MLP to more easily approximate a 
    higher frequency function.

    Args:
      pos_in: input tensor to be positionally encoded, (H*W*num_samples, 3) for sampled point
      freq: mapping from R into a higher dimensional space R^(2L), in which
                 L is called the frequency
      include_input: whether or not to include the input in positional encoding
      log_sampling: sample logarithmically in frequency space, otherwise linearly
    
    Returns:
      pos_out: positional encoding of the input tensor. 
               (H*W*num_samples, (include_input + 2*freq) * 3)
    """
    # Whether or not include the input in positional encoding
    pos_out = [pos_in] if include_input else []

    # Shape of freq_bands: (freq)
    if log_sampling:
        freq_bands = 2.0 ** torch.linspace(0.0, freq - 1, freq).to(pos_in)
    else:
        freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (freq - 1), freq).to(pos_in)

    # TODO: why reduce \pi when calculating sin and cos
    for freq in freq_bands:
        for func in [torch.sin, torch.cos]:
            pos_out.append(func(pos_in * freq))

    pos_out = torch.cat(pos_out, dim=-1)
    return pos_out


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True)

    Args:
      tensor: tensor whose cumulative product along dim=-1 is to be computed.
    
    Returns:
      cumprod: cumprod of Tensor along dim=-1
    """
    # cumprod = (tensor[0], tensor[0]*tensor[1], tensor[0]*tensor[1]*tensor[2], ...)
    cumprod = torch.cumprod(tensor, dim=-1)

    # Roll down the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)

    # cumprod = (1, tensor[0], tensor[0]*tensor[1], ...)
    cumprod[..., 0] = 1.
   
    return cumprod


def volume_rendering(
    radiance_field: torch.Tensor,
    ray_origins: torch.Tensor,
    depth_values: torch.Tensor
) -> Tuple[torch.Tensor]:
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    bundle, and the sampled depth values along them.
    Referring to the notebook for more details on volume rendering equations.

    Args:
      radiance_field: at each query location (X, Y, Z), our model predict 
        RGB color and a volume density (sigma), shape (H, W, num_samples, 4)
      ray_origins: origin of each ray, shape (H, W, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)
    
    Returns:
      rgb_map: rendered RGB image, shape (H, W, 3)
      depth_map: rendered depth image, shape (H, W)
      acc_map: accumulated transmittance map, shape (H, W)
    """
    rgb_map, depth_map, acc_map = None, None, None

    # Concatenate the H and W dimension, so that the first dimension represents
    # the number of rays
    H, W, num_samples, _ = radiance_field.shape
    radianceField = radiance_field.clone().contiguous().view(H*W, num_samples, 4) # (num_rays, num_samples, 4)
    rayOrigins = ray_origins.clone().contiguous().view(H*W, -1) # (num_rays, 3)
    depthValues = depth_values.clone().contiguous().view(H*W, -1) # (num_rays, num_samples)

    # Apply relu to the predicted volume density to make sure that all the values
    # are larger or equal than zero
    sigma = F.relu(radianceField[..., 3]) # (num_rays, num_samples)

    # Apply sigmoid to predicted RGB color (which is a logit), so that all the values
    # lie between -1 and 1
    rgb = torch.sigmoid(radianceField[..., :3]) # (num_rays, num_samples, 3)

    # Redundant vector
    one_e_10 = torch.tensor([1e10]).to(rayOrigins)
    one_e_10 = one_e_10.expand(depthValues[..., :1].shape) # (num_rays, 1)

    # We get the distance between sample points, but notice that the last sample
    # points of each ray would not have corresponding distance, we set it to be
    # a large value (1e10), so that it's approximately zero when `exp(-1e10 * sigma)`
    delta = depthValues[..., 1:] - depthValues[..., :-1] # (num_rays, num_samples-1)
    delta = torch.cat((delta, one_e_10), dim=-1) # (num_rays, num_samples)

    # Calculating `alpha = 1−exp(−𝜎𝛿)`
    alpha = 1. - torch.exp(-sigma * delta)  # (num_rays, num_samples)

    # Calculate transmittance value, notice that T_1 = 1
    # It's possible that we get alpha=1 (sigma=0) for point A, which could make
    # transmittance of all the points after point A to be 0, we also want to take
    # their information into consideration, therefore we add a small value (1e-10)
    # to avoid vanishing transmittance
    trans = cumprod_exclusive(1. - alpha + 1e-10) # (num_rays, num_samples)
    weights = alpha * trans # (num_rays, num_samples)

    # (num_rays, num_samples, 1) * (num_rays, num_samples, 3) -> (num_rays, num_samples, 3)
    rgb_map = (weights[..., None] * rgb).sum(dim=-2) # (num_rays, 3)
    rgb_map = rgb_map.contiguous().view(H, W, 3) # (H, W, 3)
    
    # (num_rays, num_samples) * (num_rays, num_samples) -> (num_rays, num_samples)
    depth_map = (weights * depthValues).sum(dim=-1) # (num_rays)
    depth_map = depth_map.contiguous().view(H, W) # (H, W)

    # accumulated transmittance map
    acc_map = weights.sum(-1) # (num_rays)
    acc_map = acc_map.contiguous().view(H, W) # (H, W)

    return rgb_map, depth_map, acc_map


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


class TinyNeRF(torch.nn.Module):
    def __init__(self, pos_dim, fc_dim=128):
      r"""Initialize a tiny nerf network, which composed of linear layers and
      ReLU activation. More specifically: linear - relu - linear - relu - linear
      - relu -linear.
      The module is intentionally made small so that we could achieve reasonable
      training time on Google Colab.

      Args:
        pos_dim: dimension of the positional encoding output
        fc_dim: dimension of the fully connected layer
      """
      # python3 don't need to specify the arguments of super
      super().__init__()
      # super(TinyNeRF, self).__init__()

      self.nerf = nn.Sequential(
                    nn.Linear(pos_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, 4)
                  )
    
    def forward(self, x):
      r"""Output volume density and RGB color (4 dimensions), given a set of 
      positional encoded points sampled from the rays
      """
      x = self.nerf(x)
      return x


def tinynerf_step_forward(height, width, focal_length, trans_matrix,
                             near_point, far_point, num_depth_samples_per_ray,
                             encoder, get_minibatches_function, model):
    r"""Perform one iteration of training, which take information of one of the
    training images, and try to predict its rgb values

    Args:
      height: height of the image
      width: width of the image
      focal_length: focal length of the camera
      trans_matrix: transformation matrix, which is also the camera pose
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      encoding_function: encode the query points, which is positional_encoding for us
      get_minibatches_function: function to cut the ray bundles into several chunks
        to avoid out-of-memory issue
      model: predefined tiny NeRF model

    Returns:
      rgb_predicted: predicted rgb values of the training image
    """
    # Get the "bundle" of rays through all image pixels.
    # (H, W, 3) & (H, W, 3)
    ray_origins, ray_directions = get_rays(height, width, focal_length, trans_matrix)
    
    # Sample points along each ray
    # (H, W, num_samples, 3) & (H, W, num_samples)
    sampled_points, depth_values = sample_points_from_rays(
        ray_origins, ray_directions, near_point, far_point, num_depth_samples_per_ray
    )

    # "Flatten" the sampled points, (H * W * num_samples, 3)
    flattened_sampled_points = sampled_points.reshape((-1, 3))

    # Encode the sampled points (default: positional encoding). (H * W * num_samples, encode_dim)
    encoded_points = encoder(flattened_sampled_points)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=16384)
    predictions = []
    for batch in batches:
      predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0) # (H*W*num_samples, 4)

    # "Unflatten" the radiance field.
    unflattened_shape = list(sampled_points.shape[:-1]) + [4] # (H, W, num_samples, 4)
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape) # (H, W, num_samples, 4)

    # Perform differentiable volume rendering to re-synthesize the RGB image. # (H, W, 3)
    rgb_predicted, _, _ = volume_rendering(radiance_field, ray_origins, depth_values)

    return rgb_predicted


def train(images, poses, hwf, i_split, near_point, 
          far_point, num_depth_samples_per_ray, encoder,
          num_iters, model, DEVICE="cuda"):
    r"""Training a tiny nerf model

    Args:
      images: all the images extracted from dataset (including train, val, test)
      poses: poses of the camera, which are used as transformation matrix
      hwf: [height, width, focal_length]
      i_split: [train set index, val set index, test set index]
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      encoder: encode the sampled points, which is positional_encoding for us
      num_iters: number of training iterations
      model: predefined tiny NeRF model
    """
    # Image information
    H, W, focal_length = hwf
    H = int(H)
    W = int(W)
    i_train, i_val, i_test = i_split

    # Optimizer parameters
    lr = 5e-3

    # Misc parameters
    display_every = 100  # Number of iters after which stats are displayed

    # Define Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    # Use the first test images for visualization
    test_idx = len(i_train)
    test_img_rgb = images[test_idx, ..., :3]
    test_pose = poses[test_idx]

    for i in range(num_iters):
      # Randomly pick a training image as the target, get rgb value and camera pose
      train_idx = np.random.randint(len(i_train))
      train_img_rgb = images[train_idx, ..., :3]
      train_pose = poses[train_idx]

      # Run one iteration of TinyNeRF and get the rendered RGB image.
      rgb_predicted = tinynerf_step_forward(H, W, focal_length,
                                              train_pose, near_point,
                                              far_point, num_depth_samples_per_ray,
                                              encoder, get_minibatches, model)

      # Compute mean-squared error between the predicted and target images
      loss = torch.nn.functional.mse_loss(rgb_predicted, train_img_rgb)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      # Display rendered test images and corresponding loss
      if i % display_every == 0:
        # Render test image
        rgb_predicted = tinynerf_step_forward(H, W, focal_length,
                                              test_pose, near_point,
                                              far_point, num_depth_samples_per_ray,
                                                encoder, get_minibatches, model)
        
        # Calculate loss and Peak Signal-to-Noise Ratio (PSNR)
        loss = F.mse_loss(rgb_predicted, test_img_rgb)
        print("Loss:", loss.item())
        psnr = -10. * torch.log10(loss)
        
        psnrs.append(psnr.item())
        iternums.append(i)

        # Visualizing PSNR
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(rgb_predicted.detach().cpu().numpy())
        plt.title(f"Iteration {i}")
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title("PSNR")
        plt.show()

    print('Finish training')
