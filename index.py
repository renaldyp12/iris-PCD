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


def detect_pupil(image):
  gray = grayscale(image)

  _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

  binary = bitwiseNot(binary)

  smoothed = cv2.GaussianBlur(binary, (5, 5), 0)

  return smoothed


def find_pupil_center(smoothed):
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



def main():
  # Baca gambar
  image = cv2.imread('images/zulaikahl1.bmp')

  # Deteksi pupil
  smoothed = detect_pupil(image)
  center, radius = find_pupil_center(smoothed)
  print(center, radius)
  # Deteksi iris

  cv2.circle(image, center, radius, (255, 0, 0), 2)

  # Tampilkan hasil
  cv2.imshow('IRIS - PCD', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()