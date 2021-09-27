from tkinter import *
import cv2
import numpy as np


class MasterGUI:
    def __init__(self, root):
        self.width = 600
        self.height = 600
        self.root = root

        self.root.title("Rotating Raster")

        self.canvas = Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()

        self.image = self.get_image()

        self.draw()

    def get_image(self):
        width = 13
        height = 10

        img = cv2.imread(f'space-invader.jpg')
        print(img.shape)
        x_step = img.shape[1] / width
        y_step = img.shape[0] / height

        reduced_img = np.zeros((width, height, 3))
        for x in range(width):
            for y in range(height):
                reduced_img[x, y, :] = img[int(round(y_step / 2 + y_step * y, 0)), int(round(x_step / 2 + x_step * x, 0)), :]

        return reduced_img

    def draw(self):
        step = 30
        for x in range(self.image.shape[0]):
            for y in range(self.image.shape[1]):
                color = '#' + ''.join(list(map(lambda px: hex(int(px))[2:], self.image[x, y, :])))
                self.canvas.create_rectangle(x * step, y * step, x * step + step, y * step + step, fill=color, outline='')


app = Tk()

ui = MasterGUI(app)

app.mainloop()
