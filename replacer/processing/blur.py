import cv2

# Step 1: Read the image
image = cv2.imread("copper.png")

# Step 2: Apply Gaussian Blur
# Parameters: (source image, (kernel width, kernel height), standard deviation)
blurred_image = cv2.GaussianBlur(image, (15, 15), 10)

# Step 3: Show the original and blurred images
# cv2.imshow('Original Image', image)
# cv2.imshow('Blurred Image', blurred_image)

# Step 4: Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save the blurred image to a file
cv2.imwrite("blurred_image.png", blurred_image)
