
import cv2
import numpy as np
import sys

braille_data = { ((0,0),) : "a", ((0,0), (0,1)) : "b", ((0,0), (1,0)) : "c", ((0,0), (1,0), (1,1)) : "d", ((0,0), (1,1)) : "e", ((0,0), (1,0), (0,1)) : "f", ((0,0), (1,0), (0,1), (1,1)) : "g", ((0,0), (0,1), (1,1)) : "h", ((1,0), (0,1)) : "i", ((1,0), (0,1), (1,1)) : "j", ((0,0), (0,2)) : "k", ((0,0), (0,1), (0,2)) : "l", ((0,0), (1,0), (0,2)) : "m", ((0,0), (1,0), (1,1), (0,2)) : "n", ((0,0), (1,1), (0,2)) : "o", ((0,0), (1,0), (0,1), (0,2)) : "p", ((0,0), (1,0), (0,1), (1,1), (0,2)) : "q", ((0,0), (0,1), (1,1), (0,2)) : "r", ((1,0), (0,1), (0,2)) : "s", ((1,0), (0,1), (1,1), (0,2)) : "t", ((0,0), (0,2), (1,2)) : "u", ((0,0), (0,1), (0,2), (1,2)) : "v", ((1,0), (0,1), (1,1), (1,2)) : "w", ((0,0), (1,0), (0,2), (1,2)) : "x", ((0,0), (1,0), (1,1), (0,2), (1,2)) : "y", ((0,0), (1,1), (0,2), (1,2)) : "z", ((1,0), (1,1), (0,2), (1,2)): "!", ((0,0), (1,0), (0,1), (0,2), (1,2)) : "&", ((0,0), (1,0), (0,1),(1,1), (0,2), (1,2)) : "%", ((1,0), (1,1), (0,1), (0,2), (1,2)) : "st", ((1,0), (0,2), (1,2)) : "ie", ((0,0), (0,1), (1,1), (1,2)) : "ue", ((1,0), (0,1), (1,2)) : "oe", ((0,0), (1,2)) : "au", ((0,0), (1,0), (0,1), (1,1), (1,2)) : "^", ((1,0), (0,1), (0,2), (1,2)) : "sz" }

center = (288, 726)
radius = 6

selected_circle = (0,0)

def visualize_keypoints(image, keypoints):
	img_copy = image.copy()

	if len(img_copy.shape) == 2:
		img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

	keypoints.sort(key=lambda x: (x[1], x[0]))

	delta = 15

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

	cv2.imshow("image", img_copy)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

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

dots = zip(*loc[::-1])
filtered_dots = []

from math import sqrt
for dot in dots:
	can_add = True
	for d2 in filtered_dots:
		distance = sqrt((d2[1]-dot[1])**2 + (d2[0]-dot[0])**2)
		if distance < 10:
			can_add = False
			break
	if can_add:
		filtered_dots.append(dot)
		print(len(filtered_dots))

dots = filtered_dots


img_copy = start_img.copy()
for pt in dots:
	x,y = pt
	cv2.circle(img_copy, (x + w//2, y + h//2), w//2, (0,0,255), 1)

cv2.imshow("image", img_copy)

print(loc)


cv2.waitKey(0)

visualize_keypoints(img_copy, dots)

#out_path = "detected_braille.png"
#cv2.imwrite(out_path, img_copy)

