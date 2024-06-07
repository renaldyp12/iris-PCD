import cv2
import numpy as np

def grayscale(image):
  height, width, channels = image.shape

  # Create an empty array for the grayscale image
  gray_image = np.zeros((height, width), dtype=np.uint8)

  # Loop through each pixel in the image
  for i in range(height):
    for j in range(width):
      # Get the B, G, R values from the image
      B = image[i, j, 0]
      G = image[i, j, 1]
      R = image[i, j, 2]

      # Calculate the grayscale value using the formula
      gray_value = int(0.299 * R + 0.587 * G + 0.114 * B)

      # Assign the grayscale value to the new image
      gray_image[i, j] = gray_value

  return gray_image

def bitwiseNot(image):
  not_image = np.zeros_like(image)

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      not_image[i, j] = 255 - image[i, j]

  return not_image

def threshold(image, threshold_value, max_value=255):
  # Iterate over each pixel value in the grayscale image
  binary = np.zeros_like(image)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if image[i, j] > threshold_value:
        binary[i, j] = 0  # Below threshold, set to 0
      else:
        binary[i, j] = max_value  # Above threshold, set to max_value

  return binary

def detect_pupil(image):
  gray = grayscale(image)
  cv2.imshow('Grayscale Image', gray)

  binary = threshold(gray, 40, 255)
  cv2.imshow('threshold Image', binary)

  binary = bitwiseNot(binary)

  smoothed = cv2.GaussianBlur(binary, (5, 5), 0)
  # cv2.imshow('Smoothed Image', smoothed)
  return smoothed


def find_center(smoothed):
  height, width = smoothed.shape

  left = width
  right = 0
  top = height
  bottom = 0

  for y in range(height):
    for x in range(width):
      if smoothed[y, x] == 0:
        if x < left:
          left = x
        if x > right:
          right = x
        if y < top:
          top = y
        if y > bottom:
          bottom = y

  # Calculate the center and radius
  center_x = (left + right) // 2
  center_y = (top + bottom) // 2
  radius_x = (right - left) // 2
  radius_y = (bottom - top) // 2
  radius = (radius_x + radius_y) // 2

  return (center_x, center_y), radius


def detect_iris(image, center, radius):
  gray = grayscale(image)
  smoothed = cv2.GaussianBlur(gray, (5, 5), 0)

  circles = cv2.HoughCircles(smoothed, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                             param1=50, param2=30, minRadius=radius + 10, maxRadius=radius + 50)

  if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
      # Check if the detected circle's center is close to the pupil's center
      if np.linalg.norm(np.array((x, y)) - np.array(center)) < radius:
        return (x, y), r

  return None, None


def main():
  image = cv2.imread('images/aevar2.bmp')
  gray = grayscale(image)
  binary = threshold(gray, 95, 255)
  blurred = cv2.GaussianBlur(binary, (9,9), 2)
  binary = bitwiseNot(blurred)
  cv2.imshow('Edge Image 1', binary)

  eye = threshold(gray, 170, 255)
  blurredEye = cv2.GaussianBlur(eye, (9, 9), 2)

  binaryEye = bitwiseNot(blurredEye)
  binaryEye = cv2.Canny(binaryEye, 20, 100)
  cv2.imshow('Edge Image 2', binaryEye)
  #
  centerRetina, radiusRetina = find_center(binary)
  centerEye, radiusEye = find_center(binaryEye)

  # Deteksi pupil
  smoothed = detect_pupil(image)
  center, radius = find_center(smoothed)

  axes_lengths = (radiusEye, 75)  # Radius for both axes
  angle = 0
  start_angle = 0
  end_angle = 360
  color = (0, 0, 255)  # Green color in BGR
  thickness = 2

  if center and radius:
    cv2.circle(image, center, radius, (255, 0, 0), 2)
    cv2.circle(image, centerRetina, radiusRetina, (0, 255, 0), 2)
    cv2.ellipse(image, centerEye, axes_lengths, angle, start_angle, end_angle, color, thickness)
  #if iris_center and iris_radius:
    # cv2.circle(image, iris_center, iris_radius, (0, 255, 0), 2)


  # Tampilkan hasil
  cv2.imshow('Detected Pupil and Iris', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()