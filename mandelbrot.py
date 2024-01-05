#!/usr/bin/env python3

import sys
from time import sleep
from argparse import ArgumentParser
import random
import math
import time
import traceback
import concurrent.futures as futures
import colorsys

from PIL import Image, ImageOps, ImageDraw, ImageShow

try:
    from inky.auto import auto as Inky
    file_output = False
except ImportError:
    traceback.print_exc()
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
        self.max_iter = 150
        self.zoom = 2
        self.sat_weight = 0.0005
        self.count = 15
        self.min_interval = 0
        self.slice_height = 1
        self.smooth = True
        self.p = 2
        self.escape_radius = 100
        self.init_color = (0, 0, 0)
        self.max_iter_color = (255, 255, 255)
        self.seed = random.randrange(sys.maxsize)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.rng = random.Random(self._seed)

    def mandelbrot(self, x, y):
        c = complex(x, y)
        z = 0
        n = 0
        while abs(z) <= self.escape_radius and n < self.max_iter:
            z = z ** self.p + c
            n += 1
        n_float = n
        if self.smooth and n < self.max_iter:
            n_float =- math.log(math.log(abs(z), self.escape_radius), self.p)
        return n, n_float

    def _color(self, n_float):
        # Make color more saturated the closer it is to an edge
        ratio = n_float / self.max_iter
        hue = ratio
        if self.sat_weight is None:
            sat = 1
        else:
            sat = (self.sat_weight ** ratio - 1) / (self.sat_weight - 1)
        val = 1
        return tuple([int(c * 255) for c in colorsys.hsv_to_rgb(hue, sat, val)])

    @staticmethod
    def interpolate_color(c1, c2, frac):
        return tuple([int((c2[i] - c1[i]) * frac + c1[i]) for i in range(0, 3)])

    def color(self, n, n_float):
        if n == self.max_iter:
            c = self.max_iter_color
        elif self.smooth and n < self.max_iter:
            n_int = math.floor(n)
            c1 = self._color(n_int)
            c2 = self._color(n_int + 1)
            c = self.interpolate_color(c1, c2, n_float % 1)
        else:
            c = self._color(n)
        return c

    def generate_slice(self, row_start, h, get_img=False, get_edges=False):
        edges = [] if get_edges else None
        w = self.out.resolution[0]
        draw = None
        if get_img:
            img = Image.new('RGB', (w, h), self.init_color)
            draw = ImageDraw.Draw(img)
        re_start, re_end, im_start, im_end = self.window
        for col in range(0, w):
            for row in range(row_start, row_start + h):
                x = re_start + (col / w) * (re_end - re_start)
                y = im_start + (row / self.out.resolution[1]) * (im_end - im_start)
                n, n_float = self.mandelbrot(x, y)
                if draw is not None:
                    draw.point((col, row - row_start), self.color(n, n_float))
                if get_edges and n == (self.max_iter - 1):
                    edges.append((x, y))
        return (row_start, img if get_img else None, edges)

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
                center = self.rng.choice(edges)
                w = ((self.window[1] - self.window[0]) / 2) / self.zoom
                h = ((self.window[3] - self.window[2]) / 2) / self.zoom
                self.window = (center[0] - w, center[0] + w, center[1] - h, center[1] + h)
                self.max_iter = int(self.max_iter * 1.1)


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
print(f'seed: {m.seed}')
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
