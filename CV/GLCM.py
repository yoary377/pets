import cv2
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

PATCH_SIZE = 25

# open the camera image
image = cv2.imread("earth.jpg", cv2.IMREAD_GRAYSCALE)

# select some patches from grassy areas of the image
clouds_locations = [(309, 480), (338, 650), (112, 273), (402, 422)]
clouds_patches = []
for loc in clouds_locations:
    clouds_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                          loc[1]:loc[1] + PATCH_SIZE])

# select some patches from sky areas of the image
water_locations = [(123, 141), (329, 139), (135, 500), (47, 567)]
water_patches = []
for loc in water_locations:
    water_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                         loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (clouds_patches + water_patches):
    glcm = graycomatrix(patch, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(graycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in clouds_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in water_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(clouds_patches)], ys[:len(clouds_patches)], 'go',
        label='Clouds')
ax.plot(xs[len(clouds_patches):], ys[len(clouds_patches):], 'bo',
        label='Water')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(clouds_patches):
    ax = fig.add_subplot(3, len(clouds_patches), len(clouds_patches) * 1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel(f"Clouds {i + 1}")

for i, patch in enumerate(water_patches):
    ax = fig.add_subplot(3, len(water_patches), len(water_patches) * 2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel(f"Water {i + 1}")

# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()