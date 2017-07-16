import numpy as np

# Mean squared error
def MSE (original, approximation):
    if original.shape != approximation.shape:
        return -1
    sum = 0
    for i in range(original.shape[0]*original.shape[1]):
        sum += (original.item(i) - approximation.item(i)) * (original.item(i) - approximation.item(i))
    return sum / (original.shape[0]*original.shape[1])

# Peak signal-to-noise ratio for colorized images
def PSNR_color (original, approximation):
    if original.shape != approximation.shape:
        return -1
    mse_r = MSE(original[:,:,0], approximation[:,:,0])
    mse_g = MSE(original[:,:,1], approximation[:,:,1])
    mse_b = MSE(original[:,:,2], approximation[:,:,2])
    mse = (mse_r + mse_g + mse_b) / 3
    psnr = 10 * np.log10(1 / mse)
    return psnr
    
from skimage.io import imread
image_path = '.\\parrots.jpg'
image = imread(image_path)
from pylab import imshow
import matplotlib.pyplot as plt
%matplotlib inline

# Show initial image
plt.figure()
imshow(image)

from skimage import img_as_float
image = img_as_float(image)
# Make pixel triplets
pixels = [[image.item(i),image.item(i+1),image.item(i+2)] 
          for i in range(image.shape[0]*image.shape[1]*image.shape[2]) if i % 3 == 0]

from sklearn.cluster import k_means
psnrs_mean = []
psnrs_median = []
cluster_nums = range(2,16)

for cluster_num in cluster_nums:
    
    # Get algorithm results
    centroid, label, inertia = k_means(pixels, cluster_num, init='k-means++', random_state=241)

    # Initialize service variables and output images
    clusterized_image = np.empty([image.shape[0], image.shape[1], image.shape[2]])
    clusterized_image_midcolor = np.empty([image.shape[0], image.shape[1], image.shape[2]])
    clusterized_image_median = np.empty([image.shape[0], image.shape[1], image.shape[2]])
    pixels_per_cluster = [[] for cluster in range(cluster_num)]
    stride = image.shape[1]
    
    # Group pixels by cluster
    for i in range(image.shape[0]*image.shape[1]):
        cluster = label[i]
        pixel = centroid[cluster]
        clusterized_image[i // stride, i % stride] = pixel
        pixels_per_cluster[cluster].append(pixels[i])
    
    # Calculate median and mean colors per cluster
    median_colors = []
    mean_colors = []
    for pixel_set in pixels_per_cluster:
        median_colors.append(np.median(pixel_set, axis=0))
        mean_colors.append(np.mean(pixel_set, axis=0))
    
    # Colorize image using median color and mean color
    for i in range(image.shape[0]*image.shape[1]):
        cluster = label[i]
        clusterized_image_median[i // stride, i % stride] = median_colors[cluster]
        clusterized_image_midcolor[i // stride, i % stride] = mean_colors[cluster]
    
    # Show images colorized
    # using median color (to the left)
    # using mean color (to the right)
    fig = plt.figure(figsize=(14, 14))
    fig.add_subplot(1,2,1)
    imshow(clusterized_image_median)
    fig.add_subplot(1,2,2)
    imshow(clusterized_image_midcolor)
    
    psnrs_mean.append(PSNR_color(image, clusterized_image_midcolor))
    psnrs_median.append(PSNR_color(image, clusterized_image_median))

# Show PSNR to cluster number graphs for
# median color image (to the left)
# mean color image (to the right)
psnr_threshlod = 20.0
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.plot(psnrs_median, 'r', linewidth=2)
plt.legend(['PSNR for median color'])
plt.axhline(y=psnr_threshlod)
fig.add_subplot(1,2,2)
plt.plot(psnrs_mean, 'g', linewidth=2)
plt.legend(['PSNR for mean color'])
plt.axhline(y=psnr_threshlod)

print('Minimal PSNR for mean colorized image more than %.3f and cluster number for it' % psnr_threshlod)
i = 0
for psnr in psnrs_mean:
    if (psnr > psnr_threshlod):
        print(psnr, cluster_nums[i])
        break
    i += 1
i = 0
print('Minimal PSNR for median colorized image more than %.3f and cluster number for it' % psnr_threshlod)
for psnr in psnrs_median:
    if (psnr > psnr_threshlod):
        print(psnr, cluster_nums[i])
        break
    i += 1
