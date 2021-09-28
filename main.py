from tkinter import *
import cv2
import numpy as np
from math import cos, sin, pi, floor, ceil
import time


class MasterGUI:
    def __init__(self, root):
        self.width = 800
        self.height = 800
        image = self.get_image()
        max_dimension = max(*image.shape[:-1])

        self.border = int(round((2 ** .5 * max_dimension - max_dimension + 1) / 2 + .5, 0))
        self.size = max_dimension + 2 * self.border
        self.step = int(round(self.width / self.size, 0))

        self.image = np.zeros((self.size, self.size, 3))
        self.image[:, :] = image[0, 0]
        other_border = (self.size - min(*image.shape[:-1])) / 2

        other_border_approx = int(other_border)
        other_border_extra = int((other_border - other_border_approx) * 2)

        self.image[self.border:-self.border, other_border_approx + other_border_extra: -other_border_approx] = image

        self.root = root

        self.root.title("Rotating Raster")

        self.canvas = Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()

        while True:
            self.run_rotation()

    def get_image(self):
        scale = 1
        width = 13 * scale
        height = 10 * scale

        img = cv2.imread(f'space-invader.jpg')

        x_step = img.shape[1] / width
        y_step = img.shape[0] / height

        reduced_img = np.zeros((width, height, 3))
        for x in range(width):
            for y in range(height):
                reduced_img[x, y, :] = img[int(round(y_step / 2 + y_step * y, 0)), int(round(x_step / 2 + x_step * x, 0)), :]

        return reduced_img

    def run_rotation(self):
        start = time.time()
        total_time = 10
        while time.time() - start < total_time:
            self.draw(self.rotate_by_point(self.image, (time.time() - start) / total_time * 2 * pi))

    def draw(self, image):
        self.canvas.delete('all')
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                color = '#' + ''.join(list(map(lambda px: hex(int(px))[2:], image[x, y, :])))
                sx, sy = x * self.step, y * self.step
                self.canvas.create_rectangle(sx, sy, sx + self.step, sy + self.step, fill=color, outline='')
        self.root.update_idletasks()
        self.root.update()

    def rotate_yolo(self, image, radians):
        rotation_mat = np.array([[cos(radians), -sin(radians)],
                                 [sin(radians), cos(radians)]])
        rotated_image = np.zeros((self.size, self.size, 3))
        rotated_image[:, :] = image[0, 0]

        center_shift = self.size / 2
        for i in range(self.border, self.size - self.border):
            for j in range(self.border, self.size - self.border):
                old_location = np.array([[i], [j]])
                old_location = old_location - center_shift
                new_location = np.matmul(rotation_mat, old_location)
                new_location = np.round(new_location + center_shift)
                rotated_image[int(new_location[0, 0]), int(new_location[1, 0])] = image[i, j]
        return rotated_image

    def rotate_by_point(self, image, radians):
        rotation_mat = np.array([[cos(radians), -sin(radians)],
                                 [sin(radians), cos(radians)]])
        rotated_image = np.zeros((self.size, self.size, 3))
        rotated_image[:, :] = image[0, 0]

        center_shift = self.size / 2
        for i in range(self.border, self.size - self.border):
            for j in range(self.border, self.size - self.border):
                old_location = np.array([[i], [j]])
                old_location = old_location - center_shift + 0.5
                new_location = np.matmul(rotation_mat, old_location)
                new_location = new_location + center_shift - 0.5
                x, y = new_location[0, 0], new_location[1, 0]
                nearby_points = [
                    [floor(x), floor(y)],
                    [floor(x), ceil(y)],
                    [ceil(x), floor(y)],
                    [ceil(x), ceil(y)]
                ]
                for point in nearby_points:
                    distance_sq = (x - point[0]) ** 2 + (y - point[1]) ** 2
                    if distance_sq <= .5:
                        rotated_image[point[0], point[1]] = image[i, j]
        return rotated_image


app = Tk()

ui = MasterGUI(app)

app.mainloop()
