import imgaug.augmenters as iaa
import cv2
import glob

# img = cv2.imread("images\\ng 2.jpg")
# cv2.imshow("Image", img)
# cv2.waitKey(0)
#1. Load Dataset
images = []
images_path = glob.glob("images/*.jpg")
for img_path in images_path:
    img = cv2.imread(img_path)
    images.append(img)

#2. Image Augmentation
augmentation = iaa.Sequential([
    #iaa.Fliplr(0.8), # horizontal flips
    #iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
         rotate=(-25, 25),
         shear=(-8, 8)
     )
])

augmented_images = augmentation(images = images)
i = 251
ng = 'ng '
dir = 'C:\\Users\\NAMPC\\Desktop\\image_yolo\\images\\'
#3. Show images
for img in augmented_images:
    name = dir + ng + str(i) +'.jpg'
    cv2.imwrite(name, img)
    i+=1
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)