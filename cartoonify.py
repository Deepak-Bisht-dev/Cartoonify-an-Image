def Cartoonify(photo):

	# importing libraries openCV and numpy for the project...
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt


	# reading image using imread func.
	img = cv2.imread(photo)

	# Convert BGR image to RGB format
	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Convert to Grey Image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# smoothening a grayscale image
	blurred = cv2.medianBlur(gray, 5)

	# retrieving the edges of an image
	edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
										cv2.THRESH_BINARY, 7, 5)

	# preparing a mask image
	color = cv2.bilateralFilter(img, 5, 200, 200)

	# putting all things togethere using bitwise_and 
	cartoon = cv2.bitwise_and(color, color, mask=edges)


	# cv2.imshow("Image", img)
	# # cv2.imshow("Gray_Image", gray)
	# # cv2.imshow("edges", edges)
	# cv2.imshow("Cartoon", cartoon)

	plt.figure(figsize=(17,10))
	plt.subplot(1,5,1)
	plt.title('Original image', size = 20)
	plt.imshow(rgb_img)
	plt.axis('off')
	plt.subplot(1,5,2)
	plt.title('Grey image', size = 20)
	plt.imshow(gray, cmap = "gray")
	plt.axis('off')
	plt.subplot(1,5,3)
	plt.title('Smoothening', size = 20)
	plt.imshow(blurred)
	plt.axis('off')
	plt.subplot(1,5,4)
	plt.title('Edge detection', size = 20)
	plt.imshow(edges)
	plt.axis('off')
	plt.subplot(1,5,5)
	plt.title('Cartoonified Image', size=20)
	plt.imshow(cartoon)
	plt.axis('off')
	plt.show()


	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Function call
Cartoonify(photo = "Avengers.jpg")


