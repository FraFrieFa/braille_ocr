from collections import OrderedDict
from bisect import bisect_right, bisect_left
import cv2
import numpy as np
import sys

center = (288, 726)
radius = 6

selected_circle = (0,0)

def select_circle(event, x, y, flags, param):
	global center, img, selected_circle

	if event == cv2.EVENT_LBUTTONDOWN or event== cv2.EVENT_LBUTTONUP:
		center = (x,y)
		img_copy = img.copy()
		cv2.rectangle(img_copy, (center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius), (0, 255, 0), 1)
		cv2.imshow("image", img_copy)

def change_radius(new_radius):
	global radius
	radius = new_radius
	img_copy = img.copy()
	#cv2.circle(img_copy, center, radius, (0, 255, 0), 1)
	cv2.rectangle(img_copy, (center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius), (0, 255, 0), 1)
	cv2.imshow("image", img_copy)

if len(sys.argv) != 2:
	print("2 arguments required")
	exit()

file_path = sys.argv[1]

start_img = cv2.imread(file_path)

img = start_img.copy()

cv2.namedWindow("image", cv2.WINDOW_GUI_EXPANDED)
cv2.setMouseCallback("image", select_circle)

cv2.imshow("image", img)

cv2.createTrackbar("radius", "image",  0, max(img.shape), change_radius)
cv2.setTrackbarPos("radius", "image", radius)

cv2.waitKey(0)

print("Center", center, "radius", radius)


pattern = []

for y in range(-radius, radius):
	line = []
	for x in range(-radius, radius):
		k = img[center[1]+y][center[0]+x]
		if x*x+y*y > radius*radius:
			k[0:3] = 0
		line.append(k)
	pattern.append(line)
pattern = np.array(pattern)

cv2.imshow("image", pattern)

print(pattern.shape, radius)

cv2.waitKey(0)

mask = np.zeros_like(img, dtype=np.uint8)
cv2.circle(mask, center, radius, (255,255,255), -1)
masked_img = cv2.bitwise_and(img,mask)
template = start_img[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(start_img.copy(), cv2.COLOR_BGR2GRAY)


res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.85

loc = np.where(res >= threshold)

w,h = [template.shape[0]]*2

dots = list(zip(*loc[::-1]))

sparse_clusters = OrderedDict()

from math import sqrt

print("Filtering", len(dots), "dots")

for dot in dots:

	xmin = dot[0] - 5
	xmax = dot[0] + 5

	ymin = dot[1] - 5
	ymax = dot[1] + 5

	y_range_start = bisect_left(list(sparse_clusters.keys()), ymin)
	y_range_end = bisect_right(list(sparse_clusters.keys()), ymax)

	if y_range_start == y_range_end:
		sparse_clusters[dot[1]] = [dot[0]]
		#sparse_clusters = OrderedDict(sorted(sparse_clusters.items()))
	else:
		found_cluster = False
		for y_index in range(y_range_start, y_range_end):
			y_value = list(sparse_clusters.keys())[y_index]
			row = sparse_clusters[y_value]
			x_range_start = bisect_left(row, xmin)
			x_range_end = bisect_left(row, xmax)
			if x_range_start == x_range_end:
				continue
			for x_index in range(x_range_start, x_range_end):
				x_value = row[x_index]
				distance = sqrt((x_value-dot[0])**2 + (y_value-dot[1])**2)
				if distance < 10:
					found_cluster = True
					cluster_position = (y_value, x_value)
					break
			if found_cluster:
				break
		if not found_cluster:
			if dot[1] in sparse_clusters:
				sparse_clusters[dot[1]].append(dot[0])
			else:
				sparse_clusters[dot[1]] = [dot[0]]
				#sparse_clusters = OrderedDict(sorted(sparse_clusters.items()))


dots = []
for y, xs in sparse_clusters.items():
	for x in xs:
		dots.append((x,y))

print("Detected", len(dots), "dots")

img_copy = start_img.copy()
for pt in dots:
	x,y = pt
	cv2.circle(img_copy, (x + w//2, y + h//2), w//2, (0,0,255), 1)

cv2.imshow("image", img_copy)

cv2.waitKey(0)

#out_path = "detected_braille.png"
#cv2.imwrite(out_path, img_copy)

