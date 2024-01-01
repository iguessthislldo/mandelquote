#!/usr/bin/env python3

from time import sleep
from argparse import ArgumentParser
import random
import math
import time
import traceback
import concurrent.futures as futures

from PIL import Image, ImageOps, ImageDraw, ImageShow

file_output = True

try:
    from inky.auto import auto as Inky
except ImportError:
    file_output = True


class FileOutput:
    def __init__(self):
        self.n = 1
        self.img = None
        self.resolution = (1000, 600)

    def set_image(self, img):
        self.img = img

    def show(self):
        self.img.save(f'{self.n:04}.png')
        self.n += 1


class Mandelbrot:
    def __init__(self, out):
        self.out = out
        self.window = (-2, 1, -1, 1)
        self.max_iter = 100
        self.zoom = 2
        self.sat_avg = 3
        self.count = 15
        self.min_interval = 0
        self.slice_height = 1

    def mandelbrot(self, x, y):
        c = complex(x, y)
        z = 0
        n = 0
        while abs(z) <= 2 and n < self.max_iter:
            z = z * z + c
            n += 1
        return n

    def generate_slice(self, row_start, h, get_img=False, get_edges=False):
        edges = [] if get_edges else None
        w = self.out.resolution[0]
        draw = None
        if get_img:
            img = Image.new('HSV', (w, h), (0, 0, 0))
            draw = ImageDraw.Draw(img)
        re_start, re_end, im_start, im_end = self.window
        for col in range(0, w):
            for row in range(row_start, row_start + h):
                x = re_start + (col / w) * (re_end - re_start)
                y = im_start + (row / self.out.resolution[1]) * (im_end - im_start)
                m = self.mandelbrot(x, y)
                if draw is not None:
                    if m == self.max_iter:
                        # White looks better on eInk display
                        hue = 0
                        sat = 0
                    else:
                        # Make color more saturated the closer it is to an edge
                        hue = int(m * 255 / self.max_iter)
                        sat = int((hue + 255 * self.sat_avg) / (1 + self.sat_avg))
                    val = 255
                    draw.point((col, row - row_start), (hue, sat, val))
                if get_edges and m == (self.max_iter - 1):
                    edges.append((x, y))
        return (row_start, img.convert('RGB') if get_img else None, edges)

    def generate(self, get_img=True, edges=None):
        if get_img:
            img = Image.new('RGB', self.out.resolution, (0, 0, 0))
            w, h = self.out.resolution
        get_edges = edges is not None
        with futures.ProcessPoolExecutor() as ex:
            # Break image up into slices that can be computed in parallel
            jobs = []
            for i in range(0, h // self.slice_height):
                row_start = i * self.slice_height
                jobs.append(ex.submit(self.generate_slice,
                    row_start, self.slice_height, get_img, get_edges))
            # If slices don't fit neatly, make an extra one for the remainder
            rem = h % self.slice_height
            if rem != 0:
                jobs.append(ex.submit(self.generate_slice,
                    row_start + self.slice_height, rem, get_img, get_edges))

            # As jobs finish, copy slices into the result image
            for job in futures.as_completed(jobs):
                row_start, slice_img, slice_edges = job.result()
                if get_edges:
                    edges.extend(slice_edges)
                if get_img:
                    img.paste(slice_img, (0, row_start))
        return img

    def generate_seq(self):
        for i in range(0, self.count):
            n = i + 1
            print(f'{n}/{self.count}', self.window)

            start_gen = time.monotonic()
            last = i == self.count - 1
            edges = None if last else []
            img = self.generate(True, edges)
            if edges is not None:
                print(len(edges), 'edges that can be used')
            end_gen = time.monotonic()
            print(end_gen - start_gen, 's to generate')

            self.out.set_image(img)
            self.out.show()
            end_show = time.monotonic()
            print(end_show - end_gen, 's to show')

            total = end_show - start_gen
            if total < self.min_interval:
                sleep(self.min_interval - total)

            if not last:
                center = random.choice(edges)
                w = ((self.window[1] - self.window[0]) / 2) / self.zoom
                h = ((self.window[3] - self.window[2]) / 2) / self.zoom
                self.window = (center[0] - w, center[0] + w, center[1] - h, center[1] + h)
                self.max_iter += 10


if file_output:
    loop = False
    out = FileOutput()
    m = Mandelbrot(out)
else:
    loop = True
    out = Inky(ask_user=True, verbose=True)
    m = Mandelbrot(out)
    m.min_interval = 30

print(f'{out.resolution[0]}x{out.resolution[1]}')
while True:
    try:
        m.generate_seq()
    except Exception:
        if loop:
            traceback.print_exc()
        else:
            raise
    if not loop:
        break
