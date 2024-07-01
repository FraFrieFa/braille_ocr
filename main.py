

import cv2
import numpy as np
import sys

print(sys.argv)

image_path = sys.argv[1]

braille_data = { ((0,0),) : "a", ((0,0), (0,1)) : "b", ((0,0), (1,0)) : "c", ((0,0), (1,0), (1,1)) : "d", ((0,0), (1,1)) : "e", ((0,0), (1,0), (0,1)) : "f", ((0,0), (1,0), (0,1), (1,1)) : "g", ((0,0), (0,1), (1,1)) : "h", ((1,0), (0,1)) : "i", ((1,0), (0,1), (1,1)) : "j", ((0,0), (0,2)) : "k", ((0,0), (0,1), (0,2)) : "l", ((0,0), (1,0), (0,2)) : "m", ((0,0), (1,0), (1,1), (0,2)) : "n", ((0,0), (1,1), (0,2)) : "o", ((0,0), (1,0), (0,1), (0,2)) : "p", ((0,0), (1,0), (0,1), (1,1), (0,2)) : "q", ((0,0), (0,1), (1,1), (0,2)) : "r", ((1,0), (0,1), (0,2)) : "s", ((1,0), (0,1), (1,1), (0,2)) : "t", ((0,0), (0,2), (1,2)) : "u", ((0,0), (0,1), (0,2), (1,2)) : "v", ((1,0), (0,1), (1,1), (1,2)) : "w", ((0,0), (1,0), (0,2), (1,2)) : "x", ((0,0), (1,0), (1,1), (0,2), (1,2)) : "y", ((0,0), (1,1), (0,2), (1,2)) : "z", ((1,0), (1,1), (0,2), (1,2)): "!", ((0,0), (1,0), (0,1), (0,2), (1,2)) : "&", ((0,0), (1,0), (0,1),(1,1), (0,2), (1,2)) : "%", ((1,0), (1,1), (0,1), (0,2), (1,2)) : "st", ((1,0), (0,2), (1,2)) : "ie", ((0,0), (0,1), (1,1), (1,2)) : "ue", ((1,0), (0,1), (1,2)) : "oe", ((0,0), (1,2)) : "au", ((0,0), (1,0), (0,1), (1,1), (1,2)) : "^", ((1,0), (0,1), (0,2), (1,2)) : "sz" }

image = None

newdict = {}
for x in braille_data:
	if x != tuple(sorted(x)):
		newdict[tuple(sorted(x))] = braille_data[x]
		print(x, tuple(sorted(x)))

for x in newdict:
	braille_data[x] = newdict[x]


def on_trackbar(val):
	global image
	threshold = cv2.getTrackbarPos(tb1_name, window_name)
	iterations = cv2.getTrackbarPos(tb2_name, window_name)
	saturation = cv2.getTrackbarPos(tb3_name, window_name)
	th = cv2.equalizeHist(src_gray)
	_, th = cv2.threshold(th, threshold, 255, cv2.THRESH_BINARY)
	th = cv2.multiply(th, saturation/100)
	kernel = np.ones((2,2), np.uint8)
	dilation = cv2.dilate(th, kernel, iterations=iterations)
	cv2.imshow(window_name, dilation)
	image = dilation

def detect_dots(binary_image):
	params = cv2.SimpleBlobDetector_Params()
	params.minThreshold = 5
	params.maxThreshold = 255

	params.filterByArea = True
	params.minArea = 1
	params.maxArea = 200

	params.filterByColor = True
	params.blobColor = 255

	params.filterByCircularity = True
	params.minCircularity = 0.1

	params.filterByConvexity = True
	params.minConvexity = 0.5

	#params.filterByArea = True
	#params.minArea = 10
	#params.maxArea = 100
	#params.filterByCircularity = True
	#params.minCircularity = 0.7
	#params.filterByConvexity = True
	#params.minConvexity = 0.87

	detector = cv2.SimpleBlobDetector_create(params)
	keypoints = detector.detect(binary_image)

	return keypoints

def visualize_keypoints(image, keypoints):
	img_copy = image.copy()

	if len(img_copy.shape) == 2:
		img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

	keypoints = [(int(pt.pt[0]), int(pt.pt[1])) for pt in keypoints]

	keypoints.sort(key=lambda x: (x[1], x[0]))

	delta = 17

	from math import sqrt

	while True:
		current_dot = keypoints[0]
		found_points = []
		hits = []
		for x in range(-1, 2):
			for y in range(-2,3):
				new_dot = current_dot[0] + x * delta, current_dot[1] + y * delta
				for d in keypoints:
					distance = sqrt((d[0]-new_dot[0])**2 + (d[1] - new_dot[1])**2)
					if distance < 5:
						found_points.append(d)
						hits.append((x,y))
		minx = min(_e[0] for _e in hits)
		miny = min(_e[1] for _e in hits)

		minx_real = min(_e[0] for _e in found_points)
		miny_real = min(_e[1] for _e in found_points)

		leftmost_point = (minx_real, miny_real)

		for _i in range(len(hits)):
			hits[_i] = (hits[_i][0] - minx,hits[_i][1]-miny)

		for d in found_points:
			cv2.circle(img_copy, d, 10, (255,0,0), thickness =1)
			keypoints.remove(d)
		if len(keypoints) == 0:
			break
		cv2.circle(img_copy, current_dot, 10, (0,0,255), thickness=3)

		hits = tuple(sorted(hits))

		if (0,0) in hits and (1,0) in hits:
			print("Calculat delta:", found_points)

		if hits in braille_data:
			cv2.putText(img_copy, braille_data[hits], (leftmost_point[0]-5, leftmost_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
		else:
			_index = 0
			for h in hits:
				cv2.putText(img_copy, str(h), (leftmost_point[0]-5, leftmost_point[1] + _index*10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
				_index += 1

	for x,y in keypoints:
		cv2.circle(img_copy, (x,y), 10, (0,255,0), thickness=2)


	#cv2.namedWindow("keypoints", cv2.WINDOW_GUI_EXPANDED)
	cv2.imshow("keypoints", img_copy)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


#image_path = "IMG-20240624-WA0000.jpg"

src = cv2.imread(image_path)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

window_name = "demo"
tb1_name = "Threshold"
tb2_name = "Iterations"
tb3_name = "Saturation"

cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)

cv2.createTrackbar(tb1_name, window_name, 0, 255, on_trackbar)
cv2.createTrackbar(tb2_name, window_name, 0, 10, on_trackbar)
cv2.createTrackbar(tb3_name, window_name, 0, 200, on_trackbar)

on_trackbar(0)

cv2.waitKey()
cv2.destroyAllWindows()

dots = detect_dots(image)

for d in dots:
	d.size = 10

visualize_keypoints(src, dots)
