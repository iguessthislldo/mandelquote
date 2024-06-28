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
import textwrap
import array

from PIL import Image, ImageOps, ImageDraw, ImageShow, ImageFont

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
    def __init__(self):
        self.window = (-2, 1, -1, 1)
        self.max_iter = 75
        self.zoom = 2
        # Must be <1 or None, smaller means more saturated, None disables desaturation
        self.sat_weight = 0.0005
        self.slice_height = 100
        self.smooth = True
        # Color range is evaluated separately for each zoom
        self.dynamic_color = True
        self.p = 2
        self.escape_radius = 10
        self.init_color = (0, 0, 0)
        self.max_iter_color = (255, 255, 255)

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

    def _color(self, min_n, n_float):
        # Make color more saturated the closer it is to an edge
        ratio = n_float / (self.max_iter - min_n)
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

    def color(self, min_n, n, n_float):
        if n == self.max_iter:
            c = self.max_iter_color
        elif self.smooth and n < self.max_iter:
            n_int = math.floor(n - min_n)
            c1 = self._color(min_n, n_int)
            c2 = self._color(min_n, n_int + 1)
            c = self.interpolate_color(c1, c2, n_float % 1)
        else:
            c = self._color(min_n, n)
        return c

    def generate_slice(self, resolution, row_start, h, get_data=False, get_edges=False):
        edges = [] if get_edges else None
        w = resolution[0]
        int_data = None
        float_data = None
        if get_data:
            init = [0] * (w * h)
            int_data = array.array('I', init)
            float_data = array.array('f', init)
        re_start, re_end, im_start, im_end = self.window
        for col in range(0, w):
            for row in range(row_start, row_start + h):
                x = re_start + (col / w) * (re_end - re_start)
                y = im_start + (row / resolution[1]) * (im_end - im_start)
                n, n_float = self.mandelbrot(x, y)
                if get_data:
                    index = (row - row_start) * w + col
                    int_data[index] = n
                    float_data[index] = n_float
                if get_edges and n == (self.max_iter - 1):
                    edges.append((x, y))
        return (row_start, int_data if get_data else None, float_data if get_data else None, edges)

    def generate(self, resolution, get_img=None, edges=None):
        w, h = resolution
        img = None
        if get_img:
            img = Image.new('RGBA', resolution, (0, 0, 0))
        get_edges = edges is not None

        with futures.ProcessPoolExecutor() as ex:
            # Break image up into slices that can be computed in parallel
            jobs = []
            for i in range(0, h // self.slice_height):
                row_start = i * self.slice_height
                jobs.append(ex.submit(self.generate_slice,
                    resolution, row_start, self.slice_height, get_img, get_edges))
            # If slices don't fit neatly, make an extra one for the remainder
            rem = h % self.slice_height
            if rem != 0:
                jobs.append(ex.submit(self.generate_slice,
                    row_start + self.slice_height, rem, get_img, get_edges))

            # Gather slices as jobs finish
            min_n = sys.maxsize if self.dynamic_color else 0
            slices = []
            for job in futures.as_completed(jobs):
                row_start, int_data, float_data, slice_edges = job.result()
                if get_edges:
                    edges.extend(slice_edges)
                if get_img:
                    if self.dynamic_color:
                        for n in int_data:
                            min_n = min(n, min_n)
                    slices.append((row_start, int_data, float_data))

            # Color and copy slices into the result image
            for row_start, int_data, float_data in slices:
                slice_img = Image.new('RGB', (w, self.slice_height), self.init_color)
                draw = ImageDraw.Draw(slice_img)
                for col in range(0, w):
                    for row in range(row_start, row_start + self.slice_height):
                        r = row - row_start
                        index = r * w + col
                        n = int_data[index]
                        n_float = float_data[index]
                        draw.point((col, r), self.color(min_n, n, n_float))
                img.paste(slice_img, (0, row_start))

        return img

    def zoom_in(self, center):
        w = ((self.window[1] - self.window[0]) / 2) / self.zoom
        h = ((self.window[3] - self.window[2]) / 2) / self.zoom
        self.window = (center[0] - w, center[0] + w, center[1] - h, center[1] + h)
        self.max_iter = int(self.max_iter * 1.1)


class QuoteWriter:
    def __init__(self, width):
        self.quotes_path = 'quotes.txt'
        self.margin = 20
        self.box_margin = 10
        self.box_color = '#bbbbbbbb'
        self.font_path = 'monofur.ttf'
        self.font_size = 24
        self.fill = '#000000'
        self.stroke_width = 2
        self.stroke_fill = '#ffffff'

        with open(self.quotes_path) as quotes_file:
            self.quotes = [q.strip().split(' -- ') for q in quotes_file.readlines()]

        self.font = ImageFont.truetype(self.font_path, size=self.font_size)

        self.col_width = 1
        line_size = self.font.getbbox('X' * (self.col_width + 1))
        self.line_height = line_size[3]
        while line_size[2] < (width - self.margin * 2):
            line_size = self.font.getbbox('X' * (self.col_width + 1))
            self.col_width += 1

    def draw_text(self, draw, offset, text):
        draw.text((self.margin, offset), text, font=self.font,
            fill=self.fill, stroke_width=self.stroke_width, stroke_fill=self.stroke_fill)
        return offset + self.line_height

    def write(self, img, rng):
        quote_text, quote_attr = rng.choice(self.quotes)
        lines = textwrap.wrap(quote_text, width=self.col_width)

        text_box = Image.new('RGBA', img.size, 0)
        draw = ImageDraw.Draw(text_box)
        draw.rectangle(
            (
                self.box_margin, self.box_margin,
                img.width - self.box_margin, self.box_margin + self.line_height * (len(lines) + 2)
            ),
            fill=self.box_color,
        )

        offset = self.margin
        for line in lines:
            offset = self.draw_text(draw, offset, line)
        self.draw_text(draw, offset, '-- ' + quote_attr)

        return Image.alpha_composite(img, text_box)


class Sequence:
    def __init__(self):
        self.count = 15
        self.skip = 0
        self.min_interval = 0
        self.seed = random.randrange(sys.maxsize)

        self.mandelbrot = Mandelbrot()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.rng = random.Random(self._seed)

    def generate(self, out):
        quotes = QuoteWriter(out.resolution[0])

        for i in range(0, self.count):
            n = i + 1
            print(f'{n}/{self.count}', self.mandelbrot.window)

            start_gen = time.monotonic()
            last = i == self.count - 1
            edges = None if last else []
            img = self.mandelbrot.generate(
                out.resolution, get_img=n > self.skip, edges=edges)
            if edges is not None:
                print(len(edges), 'edges that can be used')
            end_gen = time.monotonic()
            print(end_gen - start_gen, 's to generate')

            if img:
                img = quotes.write(img, self.rng)
                out.set_image(img)
                out.show()
            end_show = time.monotonic()
            print(end_show - end_gen, 's to show')

            total = end_show - start_gen
            if total < self.min_interval:
                sleep(self.min_interval - total)

            if not last:
                self.mandelbrot.zoom_in(self.rng.choice(edges))


seq = Sequence()
if file_output:
    loop = False
    out = FileOutput()
else:
    loop = True
    out = Inky(ask_user=True, verbose=True)
    seq.min_interval = 120
    seq.skip = 2

print(f'{out.resolution[0]}x{out.resolution[1]}')
print(f'seed: {seq.seed}')
while True:
    try:
        seq.generate(out)
    except Exception:
        if loop:
            traceback.print_exc()
        else:
            raise
    if not loop:
        break
