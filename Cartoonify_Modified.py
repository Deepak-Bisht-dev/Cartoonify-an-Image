# importing libraries openCV and numpy for the project...

from calendar import c
import cv2
import numpy as np
import matplotlib.pyplot as plt

#defining a function for reading file
def read_file(filename):
  img = cv2.imread(filename)

  #converting color to rgb from bgr as usually the color format is bgr
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  
  # plt.imshow(img)
  plt.show()
  return img

filename = "Avengers.jpg"
img = read_file(filename)

#using adaptive threshold for edge mask 
def edge_mask(img,line_size,blur_value):
  #output edges of images
  gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) #input image 

  gray_blur = cv2.medianBlur(gray,blur_value)

  edges = cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_size,blur_value)
  
  #output edges of images
  return edges

gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

line_size = 11
blur_value = 7  #we can play with line size and blur value also 

edges = edge_mask(img, line_size, blur_value)

# plt.imshow(edges,cmap ="gray")              #gray or binary 
# plt.show()

#reduce the colour palette 
def color_quantization(img , k):

  #transform the image
  data = np.float32(img).reshape((-1,3))

  #determine criteria 
  criteria = (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER,20,0.001)

  # implementing k-means #k clusters are formed randomly
  ret , label , center = cv2.kmeans( data, k, None , criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)

  result = center[label.flatten()]
  result = result.reshape(img.shape)

  return result

result = color_quantization(img , k=9) #k no of colors 

# plt.imshow(img)              
# plt.show()


#reduce the noise the image will get a little bit blurred up
blurred = cv2.bilateralFilter(result,d=3,sigmaColor = 200,sigmaSpace = 200) #d is diameter of each pixel

# plt.imshow(img)              
# plt.show()


#combine edge mask with the quantiz img 
def cartoon():
  c = cv2.bitwise_and(blurred,blurred,mask = edges)
  
  # plt.imshow(img)
  # plt.title("ORIGINAL IMAGE")
  # plt.show()

  # plt.imshow(c)
  # plt.title("CARTOONIFIED IMG")
  # plt.show()

  # plt.figure(figsize=(17,10))
  # plt.subplot(1,2, 1)
  # plt.title("ORIGINAL IMAGE", size=20)
  # plt.imshow(img)
  # plt.axis('off')
  # plt.subplot(1,2,2)
  # plt.title("CARTOONIFIED IMG", size=20)
  # plt.imshow(c)
  # plt.axis('off')
  # plt.show()
  

  plt.figure(figsize=(100,100))
  plt.subplot(1,5,1)
  plt.title('Original image', size = 15)
  plt.imshow(img)
  plt.axis('off')
  plt.subplot(1,5,2)
  plt.title('Grey image', size = 15)
  plt.imshow(gray, cmap = "gray")
  plt.axis('off')
  plt.subplot(1,5,3)
  plt.title('median filter', size = 15)
  plt.imshow(result)
  plt.axis('off')
  plt.subplot(1,5,4)
  plt.title('Edge detection', size = 15)
  plt.imshow(blurred)
  plt.axis('off')
  plt.subplot(1,5,5)
  plt.title('Cartoonified Image', size=15)
  plt.imshow(c)
  plt.axis('off')
  plt.show()

  cv2.waitKey(0)
  cv2.destroyAllWindows()

cartoon()