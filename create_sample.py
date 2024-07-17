
import numpy as np
from collections import OrderedDict
from bisect import bisect_right, bisect_left
import cv2
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QPainter, QImage, QTransform
from PyQt6.QtCore import Qt

import translate_braille

width = 500
height = 500

image = np.zeros((height,width,3), np.uint8)

dot_distance = 15
radius = 5
letter_distance = 25
line_distance = 35

cell_max_x = 2
cell_max_y = 3

def update_image(img):
	height,width,channels = img.shape
	bytes_per_line = 3*width
	q_image = QImage(img.data,width,height,bytes_per_line,QImage.Format.Format_BGR888)
	pixmap_item.setPixmap(QPixmap.fromImage(q_image))

def draw_letter(x,y,c):
	if c == " ":
		return
	dots = translate_braille.reverse_braille_data[c]
	for dx,dy in dots:
		cv2.circle(image, (x + dx*dot_distance ,y + dy*dot_distance), radius, (255,255,255), -1)

if __name__ == "__main__":

	app = QApplication([])
	window = QMainWindow()
	scene = QGraphicsScene()
	pixmap_item = QGraphicsPixmapItem(QPixmap())
	scene.addItem(pixmap_item)

	graphicsView = QGraphicsView(scene)
	graphicsView.setRenderHint(QPainter.RenderHint.Antialiasing)

	window.setCentralWidget(graphicsView)
	window.show()

	text_position = 10,10
	import random
	text = ""
	for c in range(91):
		new_char = chr(random.randint(48,127))
		while new_char not in translate_braille.reverse_braille_data:
			new_char = chr(random.randint(48,127))
		text += new_char

	text = text.lower()

	print(text)

	for char in text:
		draw_letter(*text_position,char)
		text_position = (text_position[0] + dot_distance * (cell_max_x-1) + letter_distance, text_position[1])
		if text_position[0] > width:
			text_position = (10, text_position[1] + dot_distance * (cell_max_y-1) + line_distance)

	update_image(image)

	app.exec()

	cv2.imwrite("./samples/sample_image.jpg", image)
