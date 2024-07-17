import numpy as np
from collections import OrderedDict
from bisect import bisect_right, bisect_left
import cv2
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem
from PyQt6.QtGui import QPixmap, QPainter, QImage, QTransform, QPen
from PyQt6.QtCore import Qt, QPoint
from scipy.spatial import cKDTree

from math import sqrt

import translate_braille

import time

center = 288, 726
radius = 6
selection_start = QPoint(center[0] - radius, center[1] - radius)
selection_end = QPoint(center[0] + radius, center[1] + radius)

def on_key_pressed(event):
	global img_copy
	#print(event.text())
	if event.text() == "d":
		start_time = time.time()
		print("Selection start", selection_start, "selection end", selection_end)

		selection_width = selection_end.x() - selection_start.x()
		selection_height = selection_end.y() - selection_start.y()

		radius = round(max(selection_width/2, selection_height/2))
		center = round((selection_start.x() + selection_end.x()) / 2), round((selection_start.y() + selection_end.y())/2)

		mask = np.zeros_like(img, dtype=np.uint8)
		cv2.circle(mask, center, radius, (255,255,255), -1)
		masked_img = cv2.bitwise_and(img,mask)

		template = start_img[selection_start.y():selection_end.y(), selection_start.x():selection_end.x()]
		template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.cvtColor(start_img.copy(), cv2.COLOR_BGR2GRAY)
		res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
		threshold = 0.85
		loc = np.where(res >= threshold)
		w,h = [template.shape[0]]*2
		dots = [tuple(_e) for _e in list(zip(*loc[::-1]))]

		print("Filtering", len(dots), "dots")
		img_height, img_width, img_depth = img.shape

		kdtree = cKDTree(dots)
		seen_dots = set()
		filtered_dots = []
		for dot in dots:
			if dot in seen_dots:
				continue

			dot_collection = []
			dots_in_range = [dots[_i] for _i in kdtree.query_ball_point(dot, 5)]
			for d2 in dots_in_range:
				seen_dots.add(d2)
				dot_collection.append(d2)

			x_average = sum([_i[0] for _i in dot_collection]) / len(dot_collection)
			y_average = sum([_i[1] for _i in dot_collection]) / len(dot_collection)

			filtered_dots.append((int(x_average), int(y_average)))

		dots = filtered_dots
		print("Resulting in", len(dots), "dots")

		kdtree = cKDTree(dots)

		img_copy = start_img.copy()

		img_center = (img_width/2, img_height/2)
		center_dot = dots[kdtree.query(img_center, 1)[1]]

		dot_amount_x = 2
		dot_amount_y = 3

		dot_distance = 16

		dots = [(_e[0] + w//2, _e[1] + w//2) for _e in dots]
		kdtree = cKDTree(dots)

		seen_dots = set()

		import random

		# size of a letter cell
		cell_max_x = 2
		cell_max_y = 3

		annotated_dots = {}

		for dot in dots:
			if dot in seen_dots:
				continue

			color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))

			to_check = [(0,0)]
			grid_positions = { (0,0) : dot }
			checked = set()

			max_x = 0
			min_x = 0
			max_y = 0
			min_y = 0

			while len(to_check):
				to_check_next_try = []
				next_element = to_check.pop()
				checked.add(next_element)
				dot_position = grid_positions[next_element]
				check_x = [next_element[0]-1, next_element[0], next_element[0]+1]
				check_y = [next_element[1]-1, next_element[1], next_element[1]+1]
				for x in check_x:
					for y in check_y:
						if (x,y) in checked:
							continue
						theory_max_x = max(x, max_x)
						theory_min_x = min(x, min_x)
						theory_max_y = max(y, max_y)
						theory_min_y = min(y, min_y)
						if theory_max_x - theory_min_x + 1 > cell_max_x:
							continue
						if theory_max_y - theory_min_y + 1 > cell_max_y:
							continue
						sub_dx = x-next_element[0]
						sub_dy = y-next_element[1]
						dots_in_range = kdtree.query_ball_point((dot_position[0] + sub_dx * dot_distance, dot_position[1] + sub_dy * dot_distance), 3)
						if len(dots_in_range):
							closest_dot = dots[kdtree.query((dot[0] + x * dot_distance, dot[1] + y * dot_distance), 1)[1]]
							seen_dots.add(closest_dot)
							cv2.circle(img_copy, closest_dot, w//2, color,-1)
							to_check_next_try.append((x,y))
							grid_positions[(x,y)] = closest_dot

							max_x = theory_max_x
							min_x = theory_min_x
							max_y = theory_max_y
							min_y = theory_min_y

				to_check += to_check_next_try

			cv2.circle(img_copy, dot, w//2, color,-1)

			# move dots
			new_grid_positions = {}
			for x,y in grid_positions:
				new_grid_positions[(x - min_x, y - min_y)] = grid_positions[(x,y)]
			grid_positions = new_grid_positions

			max_x -= min_x
			min_x = 0
			max_y -= min_y
			min_y = 0

			interpolations = {}

			for x in range(max_x+1):
				for y in range(max_y+1):
					if (x,y) in grid_positions:
						dot_position = grid_positions[(x,y)]
						directions = [(-1,0), (1,0), (0,-1), (0,1)]
						for dx, dy in directions:
							if x + dx < min_x:
								continue
							if x + dx > max_x:
								continue
							if y + dy < min_y:
								continue
							if y + dy > max_y:
								continue
							if (x+dx,y+dy) in grid_positions:
								continue
							if (x+dx,y+dy) not in interpolations:
								interpolations[(x+dx,y+dy)] = []
							interpolations[(x+dx,y+dy)].append((dot_position[0] + dx*dot_distance, dot_position[1] + dy*dot_distance))

			for x,y in interpolations:
				sumx = 0
				sumy = 0
				for sub_x, sub_y in interpolations[(x,y)]:
					sumx += sub_x
					sumy += sub_y
				sumx, sumy = round(sumx/len(interpolations[(x,y)])), round(sumy/len(interpolations[(x,y)]))
				grid_positions[(x,y)] = (sumx,sumy)

			possible_drift_x = cell_max_x - max_x
			possible_drift_y = cell_max_y - max_y

			for x,y in grid_positions:
				dot_position = grid_positions[(x,y)]
				#cv2.circle(img_copy, dot_position, w//2, (70,70,70),1)

				annotated_dots[dot_position] = (range(x, x+possible_drift_x), range(y, y+possible_drift_y))

			#cv2.circle(img_copy, grid_positions[(min_x, min_y)], w//2, (0,0,0), 1)

		print("Done grouping dots")

		influence_radius = dot_distance * 10

		line_distance = 35
		letter_distance = 25

		cell_size_x = dot_distance + letter_distance
		cell_size_y = dot_distance*2 + line_distance

		dot_probability = {_e : [] for _e in dots}

		for dot in annotated_dots:
			#print(dot)
			close_dots = [dots[_i] for _i in kdtree.query_ball_point(dot, influence_radius)]
			# DEBUG
			#for close_dot in close_dots:
			#	cv2.circle(img_copy, close_dot, w//2, (0,255,0), 1)
			#print(annotated_dots[dot])
			x_range, y_range = annotated_dots[dot]
			dot_origins_to_check = [] #[(_x,_y) for _x in x_range for _y in y_range]
			for x in x_range:
				for y in y_range:
					dot_origins_to_check.append((x,y))
			permutation_count = len(dot_origins_to_check)
			#print(dot_origins_to_check)
			for x,y in dot_origins_to_check:
				#print(x,y)
				delta_x = [dot_distance, letter_distance]
				delta_y = [dot_distance, dot_distance, line_distance]
				checkspots_x = [0]
				_d = 0
				for _i in range(cell_max_x):
					_d += delta_x[(_i + x) % cell_max_x]
					checkspots_x.append(_d)
				checkspots_y = [0]
				_d = 0
				for _i in range(cell_max_y):
					_d += delta_y[(_i + y) % cell_max_y]
					checkspots_y.append(_d)
				#print("modpoints for x", checkspots_x)
				#print("modpoints for y", checkspots_y)
				distance_sum = 0
				for close_dot in close_dots:
					#print("checking", close_dot)
					dif_x = (close_dot[0] - dot[0]) % cell_size_x
					dif_y = (close_dot[1] - dot[1]) % cell_size_y
					closest_x_index = 0
					distance_to_x_value = dif_x
					for _i, _e in enumerate(checkspots_x):
						if abs(_e - dif_x) < distance_to_x_value:
							distance_to_x_value = abs(_e - dif_x)
							closest_x_index = _i
					closest_y_index = 0
					distance_to_y_value = dif_y
					for _i, _e in enumerate(checkspots_y):
						if abs(_e - dif_y) < distance_to_y_value:
							distance_to_y_value = abs(_e - dif_y)
							closest_y_index = _i
					total_distance = sqrt(distance_to_x_value**2 + distance_to_y_value**2)
					distance_to_origin = sqrt((close_dot[0] - dot[0])**2 + (close_dot[1] - dot[1])**2)
					distance_sum += total_distance
					probable_x_index = (x+closest_x_index) % cell_max_x
					probable_y_index = (y+closest_y_index) % cell_max_y
					#print(dif_x % cell_size_x, dif_y % cell_size_y, "probably", probable_x_index, probable_y_index, total_distance, distance_to_origin)
					dot_probability[close_dot].append((probable_x_index, probable_y_index, total_distance, distance_to_origin, permutation_count))

					"""
					if probable_x_index == 0:
						cv2.circle(img_copy, close_dot, w//2, (0,0,255), -1)
					elif probable_x_index == 1:
						cv2.circle(img_copy, close_dot, w//2, (0,255,255), -1)
					if probable_y_index == 0:
						cv2.circle(img_copy, close_dot, w//2, (0,0,255), 1)
					elif probable_y_index == 1:
						cv2.circle(img_copy, close_dot, w//2, (0,255,255), 1)
					elif probable_y_index == 2:
						cv2.circle(img_copy, close_dot, w//2, (255,255,255), 1)
					"""
				#print("check done, total distance:", distance_sum)
			# DEBUG
			#cv2.circle(img_copy, dot, w//2, (0,0,0), 2)

		print("Aligned dots to rows and columns")

		dot_positions = {}

		filtered_dots = []

		for dot in dot_probability:
			weights = dot_probability[dot]

			distances = {}
			probability = {}

			for prob_x, prob_y, distance, origin_distance, perm_weight in weights:
				if (prob_x,prob_y) not in probability:
					distances[(prob_x, prob_y)] = []
					probability[(prob_x, prob_y)] = 0
				# linear weight for origin_distance, 1 at 0, 0 at influence_radius
				if origin_distance != 0: # dot does not influence its own position
					weight = 1 - origin_distance / influence_radius
					probability[(prob_x,prob_y)] += weight * 1 / perm_weight
					distances[(prob_x,prob_y)].append((distance, origin_distance))

			pixel_drift_for_origin_distance = (radius/5) / ((cell_max_x-1) * dot_distance + letter_distance)

			max_key = max(probability, key=probability.get)
			max_probability = probability[max_key]

			average_distance = []
			for distance, origin_distance in distances[max_key]:
				if origin_distance > influence_radius / 2:
					continue
				average_distance.append(distance)

			import statistics
			if len(average_distance) == 0:
				average_distance = radius*10
			else:
				average_distance = statistics.fmean(average_distance)

			if average_distance <= radius:
				dot_x, dot_y = max_key
				#print(dot_x, dot_y)
				red = 255*dot_x
				green = 120*dot_y
				#cv2.circle(img_copy, dot, w//2, (red,green,0), -1)
				dot_positions[dot] = (dot_x, dot_y)
				filtered_dots.append(dot)

		#dots = filtered_dots
		kdtree = cKDTree(dots)

		for dot in filtered_dots:
			cv2.circle(img_copy, dot, w//2, (70,70,70), -1)

		letters = []

		seen_dots = set()
		for dot in dot_positions:
			if dot in seen_dots:
				continue
			hits = []
			dot_x, dot_y = dot_positions[dot]
			for x in range(cell_max_x):
				for y in range(cell_max_y):
					other_dot_position = (dot[0] + (x-dot_x)*dot_distance, dot[1] + (y-dot_y)*dot_distance)
					close_dots = [dots[_i] for _i in kdtree.query_ball_point(other_dot_position, radius)]
					for close_dot in close_dots:
						seen_dots.add(close_dot)
					if len(close_dots)== 0:
						cv2.circle(img_copy, other_dot_position, w//2, (70,70,70), 1)
					else:
						closest_dot = dots[kdtree.query(other_dot_position, 1)[1]]
						cv2.circle(img_copy, closest_dot, w//2, (255,70,70), -1)
						hits.append((x,y))
			start_point = (dot[0] - dot_x * dot_distance, dot[1] - dot_y * dot_distance)
			hits = tuple(sorted(hits))

			# todo use average of found points to calculate empty slots

			#cv2.putText(img_copy, text, (start_point[0]-5, start_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			letters.append((start_point, hits))

		#todo check for overlapping letters
		letter_centers = []
		letter_center_hits = {}
		for (x,y), hits in letters:
			xmin = x - radius // 2
			ymin = y - radius // 2
			xmax = x + dot_distance*(cell_max_x-1) + radius // 2
			ymax = y + dot_distance*(cell_max_y-1) + radius // 2
			letter_centers.append(((xmin+xmax)//2, (ymin+ymax)//2))
			letter_center_hits[((xmin+xmax)//2, (ymin+ymax)//2)] = hits

		letterkdtree = cKDTree(letter_centers)

		seen_letters = set()

		strings = []

		for letter in letters:
			(x,y), hits = letter

			xmin = x - radius // 2
			ymin = y - radius // 2
			xmax = x + dot_distance*(cell_max_x-1) + radius // 2
			ymax = y + dot_distance*(cell_max_y-1) + radius // 2
			letter_center = ((xmin+xmax)//2, (ymin+ymax)//2)

			has_next = len(letterkdtree.query_ball_point((letter_center[0] + dot_distance * (cell_max_x-1) + letter_distance, letter_center[1]), dot_distance)) > 0
			has_prev = len(letterkdtree.query_ball_point((letter_center[0] - dot_distance * (cell_max_x-1) - letter_distance, letter_center[1]), dot_distance)) > 0


			c1 = 255 if has_next else 150
			c2 = 255 if has_prev else 150

			color = (c1,255,c2)

			#cv2.rectangle(img_copy, (x - radius//2,y - radius//2), (x + dot_distance*(cell_max_x-1)+radius//2,y+dot_distance*(cell_max_y-1)+radius//2), color, -1)

			rectX = x-radius
			rectY = y-radius
			rectWidth = 2*radius + dot_distance * (cell_max_x-1)
			rectHeight = 2*radius + dot_distance * (cell_max_y-1)

			sub_img = img_copy[rectY:rectY+rectHeight,rectX:rectX+rectWidth]
			if 0 not in sub_img.shape:
				overlay = np.full(sub_img.shape, color, dtype=np.uint8)
				transparent_square = cv2.addWeighted(sub_img, 0.4, overlay, 0.6, 1.0)
				img_copy[rectY:rectY+rectHeight,rectX:rectX+rectWidth] = transparent_square

			isKurzschrift = True #False

			if isKurzschrift:

				kurzschrift_data = {
					((1,1), (1,2)) : (0b011, "ach"),
					((0,1), (1,1)) : (0b110, "al"),
					((0,1), (0,2), (1,1)) : (0b110, "an"),
					((0,2), (1,1), (1,2)) : (0b110, "ar"),
					((0,2), (1,0)) : (0b100, "aus"),
					((0,1), (0,2)) : (0b110, "be"),
					((1,0), (1,2)) : (0b011, "ck"),
					((0,1), (0,2), (1,1), (1,2)) : (0b010, "eh"),
					((0,0), (0,1), (1,0), (1,2)) : (0b111, "ein"),
					((0,0), (0,2), (1,0), (1,1), (1,2)) : (0b111, "el"),
					((0,0), (0,1), (0,2), (1,1), (1,2)) : (0b111, "em"),
					((0,0), (1,0)) : (0b111, "en"),
					((0,1), (0,2), (1,0), (1,2)) : (0b100, "ent"),
					((0,0), (0,1), (1,0), (1,1), (1,2)) : (0b111, "er"),
					((0,0), (0,1), (0,2), (1,0), (1,1), (1,2)) : (0b111, "es"),
					((0,0), (0,2), (1,0), (1,2)) : (0b100, "ex"),
					((0,0), (0,1), (0,2), (1,0), (1,2)) : (0b111, "ge"),
					((0,2), (1,0), (1,1), (1,2)) : (0b011, "ich"),
					((1,0), (1,1)) : (0b011, "ig"),
					((0,2), (1,1)) : (0b111, "in"),
					((1,0), (1,1), (1,2)) : (0b011, "lich"),
					((0,0), (0,1), (0,2), (1,0), (1,1)) : (0b011, "ll"),
					((0,0), (0,2), (1,0), (1,2)) : (0b011, "mm"),
					((0,1), (1,2)) : (0b110, "or"),
					#((0,0), (0,1), (0,2), (1,0), (1,1)) : (0b100, "pro"),
					((0,1), (0,2), (1,0), (1,2)) : (0b011, "ss"),
					((0,1), (0,2), (1,2)) : (0b011, "te"),
					((0,1), (1,0), (1,2)) : (0b111, "un"),
				}
				for m in kurzschrift_data:
					if m != tuple(sorted(m)):
						raise ValueError("please sort dict")

				isAnlaut = not has_prev
				isInlaut = has_prev and has_next
				isAuslaut = not has_next

				binary_info = (4 if isAnlaut else 0) | (2 if isInlaut else 0) | (1 if isAuslaut else 0)

				text = ""
				if hits in kurzschrift_data:
					other_bin_data, text_data = kurzschrift_data[hits]
					if (other_bin_data & binary_info) != 0:
						text= text_data


				einformige_k = {
					((0,0),) : "aber",
					((0,0), (0,1), (1,2)) : "wie",
					((0,2), (1,0), (1,2)) : "die",
					((0,0), (1,1)) : "den",
					((0,1), (1,0)) : "ihr",
					((0,0), (1,2)) : "auf",
					((0,1), (0,2), (1,0)) : "sie",
					((0,2), (1,2)) : "im",
					((0,0), (0,1), (0,2), (1,0)) : "so",
					((0,0), (0,1), (0,2), (1,1)) : "der",
					((0,0), (0,2), (1,2)) : "und",
					((0,1), (0,2), (1,0), (1,1)) : "mit",
					((0,0), (1,0), (1,2)) : "als",
					((0,0), (1,0)) : "sich",
					((0,1), (0,2), (1,0), (1,2)) : "dass",
					((0,0), (1,0), (1,1)) : "das",
					((0,0), (0,2), (1,0), (1,2)) : "immer",
					((0,0), (1,1), (1,2)) : "schon",
					((0,0), (0,2), (1,1), (1,2)) : "zu",
				}
				for m in einformige_k:
					if m != tuple(sorted(m)):
						raise ValueError("please sort dict")


				if not has_prev and not has_next:
					if hits in einformige_k:
						text = einformige_k[hits]

				if text == "":
					text = translate_braille.translate_letter(hits)

			else:
				if hits in translate_braille.braille_data:
					text = translate_braille.translate_letter(hits)
				else:
					text = "??"
					print("unknown character", hits)
			cv2.putText(img_copy, text, (x, y + dot_distance*(cell_max_y-1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)

		update_image(img_copy)
		print("Time spent:", time.time()-start_time)

def update_image(img):
	height,width,channels = img.shape
	bytes_per_line = 3*width
	q_image = QImage(img.data,width,height,bytes_per_line,QImage.Format.Format_BGR888)
	pixmap_item.setPixmap(QPixmap.fromImage(q_image))

def on_scroll(event):
	if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
		if event.angleDelta().y() > 0:
			factor = 1.5
		else:
			factor = 2/3

		graphicsView.scale(factor, factor)

		#nx0 = posx - factor * (posx - cur_xlim[0])
		#nx1 = xrange * factor + nx0
		#ny0 = posy - factor * (posy - cur_ylim[0])
		#ny1 = yrange * factor + ny0

	else:
		old_scroll_event(event)

def mousePressed(event):
	global selection_start
	if event.button() == Qt.MouseButton.RightButton:
		selection_start = graphicsView.mapToScene(event.position().toPoint()).toPoint()
	old_mouse_event(event)

def mouseReleased(event):
	global selection_end
	global selection_start
	if event.button() == Qt.MouseButton.RightButton:
		selection_end = graphicsView.mapToScene(event.position().toPoint()).toPoint()
		minx = min(selection_start.x(), selection_end.x())
		miny = min(selection_start.y(), selection_end.y())
		maxx = max(selection_start.x(), selection_end.x())
		maxy = max(selection_start.y(), selection_end.y())
		selection_start = QPoint(minx,miny)
		selection_end = QPoint(maxx,maxy)
		rect_item.setRect(selection_start.x(), selection_start.y(), selection_end.x() - selection_start.x(), selection_end.y() - selection_start.y())
	old_mouse_release_event(event)

def mouseMoved(event):
	global selection_end
	if event.buttons() == Qt.MouseButton.RightButton:
		selection_end = graphicsView.mapToScene(event.position().toPoint()).toPoint()
		minx = min(selection_start.x(), selection_end.x())
		miny = min(selection_start.y(), selection_end.y())
		maxx = max(selection_start.x(), selection_end.x())
		maxy = max(selection_start.y(), selection_end.y())
		local_selection_start = QPoint(minx,miny)
		local_selection_end = QPoint(maxx,maxy)
		rect_item.setRect(
			local_selection_start.x(),
			local_selection_start.y(),
			local_selection_end.x() - local_selection_start.x(),
			local_selection_end.y() - local_selection_start.y())

	old_mouse_move_event(event)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("2 arguments required")
		exit()
	file_path = sys.argv[1]
	start_img = cv2.imread(file_path)
	img = start_img.copy()

	app = QApplication([])
	window = QMainWindow()

	scene = QGraphicsScene()

	rect_item = QGraphicsRectItem(selection_start.x(),selection_start.y(),selection_end.x() - selection_start.x(),selection_end.y() - selection_start.y())
	rect_item.setPen(QPen(Qt.GlobalColor.red, 0))
	rect_item.setZValue(1)
	scene.addItem(rect_item)

	pixmap_item = QGraphicsPixmapItem(QPixmap(file_path))
	scene.addItem(pixmap_item)

	graphicsView = QGraphicsView(scene)
	graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)

	windowWidth, windowHeight = graphicsView.width(), graphicsView.height()
	imageHeight, imageWidth, _ = img.shape

	scaleX = windowWidth / imageWidth
	scaleY = windowHeight / imageHeight

	scale = max(scaleX,scaleY)

	graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

	old_scroll_event = graphicsView.wheelEvent
	graphicsView.wheelEvent = on_scroll

	old_mouse_event = graphicsView.mousePressEvent
	old_mouse_release_event = graphicsView.mouseReleaseEvent
	old_mouse_move_event = graphicsView.mouseMoveEvent
	graphicsView.mousePressEvent = mousePressed
	graphicsView.mouseReleaseEvent = mouseReleased
	graphicsView.mouseMoveEvent = mouseMoved

	window.setCentralWidget(graphicsView)
	window.show()

	app.installEventFilter(window)
	window.keyPressEvent = on_key_pressed

	app.exec()

	#cv2.imwrite("modified_img.jpg", img_copy)
