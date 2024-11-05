#!/usr/bin/env python3

import sys
import os
from time import sleep
from argparse import ArgumentParser
import random
import math
from time import time
import traceback
import concurrent.futures as futures
import colorsys
import textwrap
import array
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw, ImageShow, ImageFont

use_mandelbrot_native_impl = False
try:
    from mandelbrot_native import mandelbrot_native
    use_mandelbrot_native_impl = True
except ImportError:
    pass


def setup_pi_buttons():
    import RPi.GPIO as GPIO

    buttons = [5, 6, 16, 24]
    labels = ['A', 'B', 'C', 'D']
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(buttons, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def handle_button(pin):
        label = labels[buttons.index(pin)]
        print("Button press detected on pin: {} label: {}".format(pin, label))
        if label == 'A':
            os.system('shutdown now')

    for pin in buttons:
        GPIO.add_event_detect(pin, GPIO.FALLING, handle_button, bouncetime=250)


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


class Log:
    def __init__(self):
        self.level = 0
        self.timers = []

    def print(self, *args):
        print('    ' * self.level, '-', *args)

    def push_level(self):
        self.level += 1

    def pop_level(self):
        self.level -= 1

    def push_time(self, *args):
        self.timers += [(time(), args)]

    def pop_time(self):
        start, args = self.timers.pop()
        interval = time() - start
        self.print(interval, 's to', *args)
        return interval


class EdgeError(Exception):
    pass


image_mode = 'RGBA'


class Mandelbrot:
    def __init__(self, workers, p=2, smooth=True, sat=True, dynamic_color=True):
        self.workers = workers
        self.min_worker_pixels = 100000
        self.window = (-2, 1, -1, 1)
        self.max_iter = 75
        self.zoom = 2
        # Must be <1 or None, smaller means more saturated, None disables desaturation
        self.sat_weight = 0.0005 if sat else None
        self.smooth = smooth
        # Color range is evaluated separately for each zoom
        self.dynamic_color = dynamic_color
        self.p = p
        self.escape_radius = 10
        self.init_color = (0, 0, 0)
        self.max_iter_color = (255, 255, 255)

    def mandelbrot(self, x, y, get_float):
        c = complex(x, y)
        z = 0
        n = 0
        while abs(z) <= self.escape_radius and n < self.max_iter:
            z = z ** self.p + c
            n += 1
        n_float = None
        if get_float:
            n_float = n
            if n < self.max_iter:
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

    def draw_slice(self, draw, w, row_start, slice_height, int_data, float_data, min_n):
        for col in range(0, w):
            for row in range(row_start, row_start + slice_height):
                index = (row - row_start) * w + col
                n = int_data[index]
                n_float = float_data[index]
                draw.point((col, row), self.color(min_n, n, n_float))

    def mandelbrot_python_impl(self, resolution, row_start, slice_height,
            get_int_data, get_float_data, get_edges):
        edges = [] if get_edges else None
        w, h = resolution
        int_data = None
        float_data = None
        get_data = get_int_data or get_float_data
        if get_data:
            init = [0] * (w * slice_height)
            if get_int_data:
                int_data = array.array('I', init)
            if get_float_data:
                float_data = array.array('d', init)
        re_start, re_end, im_start, im_end = self.window
        for col in range(0, w):
            for row in range(row_start, row_start + slice_height):
                x = re_start + (col / w) * (re_end - re_start)
                y = im_start + (row / h) * (im_end - im_start)
                n, n_float = self.mandelbrot(x, y, get_float_data)
                if get_data:
                    index = (row - row_start) * w + col
                    if get_int_data:
                        int_data[index] = n
                    if get_float_data:
                        float_data[index] = n_float
                if get_edges and n == (self.max_iter - 1):
                    edges.append((x, y))
        return (int_data, float_data, edges)

    def mandelbrot_native_impl(self, resolution, row_start, slice_height,
            get_int_data, get_float_data, get_edges):
        w, h = resolution
        re_start, re_end, im_start, im_end = self.window
        return mandelbrot_native(w, h, row_start, slice_height,
            re_start, re_end, im_start, im_end,
            self.escape_radius, self.max_iter, self.p,
            get_int_data, get_float_data, get_edges)

    def generate_slice(self, resolution, row_start, slice_height,
            get_int_data, get_float_data, get_edges):
        impl = self.mandelbrot_native_impl if use_mandelbrot_native_impl \
            else self.mandelbrot_python_impl
        int_data, float_data, edges = impl(
            resolution, row_start, slice_height,
            get_int_data, get_float_data, get_edges)
        if get_int_data and not get_float_data:
            float_data = int_data
        return (row_start, slice_height, int_data, float_data, edges)

    def generate(self, log, resolution, get_img=False, edges=None):
        w, h = resolution
        img = None
        draw = None
        if get_img:
            img = Image.new(image_mode, resolution, (0, 0, 0))
            draw = ImageDraw.Draw(img)
        get_edges = edges is not None

        log.push_time('generate and join slices')
        with futures.ThreadPoolExecutor(max_workers = self.workers) as ex:
            if w * h > self.min_worker_pixels:
                max_slice_height = max(h // ex._max_workers, 1)
            else:
                max_slice_height = h

            # Break image up into slices that can be computed in parallel
            get_int_data = get_img
            get_float_data = get_img and self.smooth
            jobs = []
            log.print(resolution, 'image is made of', h / max_slice_height, 'slices')
            for i in range(0, h // max_slice_height):
                row_start = i * max_slice_height
                jobs.append(ex.submit(self.generate_slice,
                    resolution, row_start, max_slice_height,
                    get_int_data, get_float_data, get_edges))
            # If slices don't fit neatly, make an extra one for the remainder
            rem = h % max_slice_height
            if rem != 0:
                jobs.append(ex.submit(self.generate_slice,
                    resolution, row_start + max_slice_height, rem,
                    get_int_data, get_float_data, get_edges))

            # Gather slices as jobs finish
            min_n = sys.maxsize if self.dynamic_color else 0
            slices = []
            for job in futures.as_completed(jobs):
                row_start, slice_height, int_data, float_data, slice_edges = job.result()
                if get_edges:
                    edges.extend(slice_edges)
                if get_img:
                    if self.dynamic_color:
                        for n in int_data:
                            min_n = min(n, min_n)
                        slices.append((row_start, slice_height, int_data, float_data))
                    else:
                        # Without dynamic color we can just add the slice to
                        # the image right now because we don't need to know the
                        # min value of int_data.
                        self.draw_slice(
                            draw, w, row_start, slice_height, int_data, float_data, min_n)
        log.pop_time()

        if get_edges:
            msg = str(len(edges)) + ' edges that can be used'
            log.print(msg)
            if not edges:
                raise EdgeError(msg)

        # Color and copy slices into the result image
        if get_img and self.dynamic_color:
            log.push_time('dynamic color')
            for row_start, slice_height, int_data, float_data in slices:
                self.draw_slice(draw, w, row_start, slice_height, int_data, float_data, min_n)
            log.pop_time()

        return img

    def zoom_in(self, center):
        w = ((self.window[1] - self.window[0]) / 2) / self.zoom
        h = ((self.window[3] - self.window[2]) / 2) / self.zoom
        self.window = (center[0] - w, center[0] + w, center[1] - h, center[1] + h)
        self.max_iter = int(self.max_iter * 1.1)


class TextWriter:
    def __init__(self, data_path, width, font_size):
        self.box_margin = 10
        self.text_margin = 10
        self.abs_text_margin = self.box_margin + self.text_margin
        self.box_color = '#bbbbbbbb'
        self.font_path = data_path / 'monofur.ttf'
        self.font_size = 24
        self.fill = '#000000'
        self.stroke_width = 2
        self.stroke_fill = '#ffffff'

        self.font = ImageFont.truetype(self.font_path, size=self.font_size)

        test_char = '|'
        line_size = self.font.getbbox(test_char)
        self.char_height = line_size[3]
        self.char_width = line_size[2]
        self.col_width = 1
        while line_size[2] <= (width - self.abs_text_margin * 2):
            line_size = self.font.getbbox(test_char * (self.col_width + 1))
            self.col_width += 1

    def wrap_text(self, *args):
        return textwrap.wrap(' '.join([str(x) for x in args]), width=self.col_width)

    def draw_text(self, draw, offset, text):
        draw.text((self.abs_text_margin, offset), text, font=self.font,
            fill=self.fill, stroke_width=self.stroke_width, stroke_fill=self.stroke_fill)
        return offset + self.char_height

    def start_text_box(self, img, line_count, bottom=False, use_text_width=None):
        text_box = Image.new('RGBA', img.size, 0)
        draw = ImageDraw.Draw(text_box)
        text_height = self.char_height * line_count
        box_height = text_height + self.text_margin * 2
        if bottom:
            box_y = img.height - self.box_margin - box_height
        else:
            box_y = self.box_margin
        if use_text_width is not None:
            box_w = self.char_width * use_text_width + self.box_margin * 2
        else:
            box_w = img.width - self.box_margin * 2
        box_x = self.box_margin
        draw.rectangle(
            (box_x, box_y, box_x + box_w, box_y + box_height),
            fill=self.box_color,
        )
        return text_box, draw, box_y + self.text_margin

    def end_text_box(self, img, text_box):
        return Image.alpha_composite(img, text_box)


class QuoteWriter(TextWriter):
    def __init__(self, data_path, width):
        super().__init__(data_path, width, font_size=24)
        self.quotes_path = data_path / 'quotes.txt'
        with self.quotes_path.open() as quotes_file:
            self.quotes = [q.strip().split(' -- ') for q in quotes_file.readlines()]

    def write(self, img, rng):
        quote_text, quote_attr = rng.choice(self.quotes)
        lines = self.wrap_text(quote_text)
        text_box, draw, offset = self.start_text_box(img, len(lines) + 1)
        for line in lines:
            offset = self.draw_text(draw, offset, line)
        self.draw_text(draw, offset, '-- ' + quote_attr)
        return self.end_text_box(img, text_box)


class InfoWriter(TextWriter):
    def __init__(self, data_path, width):
        super().__init__(data_path, width, font_size=12)

    def write(self, img, *args):
        lines = self.wrap_text(*args)
        text_box, draw, offset = self.start_text_box(img,
            len(lines), bottom=True, use_text_width=max([len(line) for line in lines]))
        for line in lines:
            offset = self.draw_text(draw, offset, line)
        return self.end_text_box(img, text_box)


class Sequence:
    def __init__(self, data_path, resolution, skip_resolution, out, count, skip, min_interval,
            seed, text, workers, p, smooth, sat, dynamic_color):
        self.data_path = data_path
        self.resolution = resolution
        self.skip_resolution = skip_resolution
        self.count = skip + count
        self.skip = skip
        self.min_interval = min_interval
        self.seed = random.randrange(sys.maxsize) if seed is None else seed
        self.text = text
        self.workers = workers
        self.p = p
        self.smooth = smooth
        self.sat = sat
        self.dynamic_color = dynamic_color
        self.img_dir = data_path / 'img'

        self.out = out
        width = resolution[0]
        if text:
            self.quotes = QuoteWriter(data_path, width)
            self.info = InfoWriter(data_path, width)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.rng = random.Random(self._seed)

    def sleep_min(self, log, interval):
        if interval < self.min_interval:
            sleep_for = self.min_interval - interval
            log.print('Sleeping for', sleep_for, 's')
            sleep(sleep_for)

    def show_img(self, img):
        self.out.set_image(img)
        self.out.show()

    def _generate_and_show(self, log):
        mandelbrot = Mandelbrot(self.workers, self.p, self.smooth, self.sat, self.dynamic_color)
        skip_res = list(self.skip_resolution)
        for i in range(0, self.count):
            n = i + 1
            get_img = n > self.skip
            log.print(f'{n}/{self.count} get_img: {get_img}', mandelbrot.window)
            log.push_time('generate and show')
            log.push_level()

            log.push_time('generate')
            last = i == self.count - 1
            edges = None if last else []
            while True:
                res = self.resolution if get_img else skip_res
                log.print(res)
                log.push_level()
                try:
                    img = mandelbrot.generate(log, res, get_img=get_img, edges=edges)
                except EdgeError:
                    if get_img:
                        raise
                    skip_res = [n * 2 for n in skip_res]
                    res = skip_res
                    continue
                finally:
                    log.pop_level()
                break
            log.pop_time()

            if img and get_img:
                log.push_time('show')
                if self.text:
                    img = self.quotes.write(img, self.rng)
                    img = self.info.write(img, f'p={self.p}')
                self.show_img(img)
                log.pop_time()

            interval = log.pop_time()
            log.pop_level()

            if get_img:
                self.sleep_min(log, interval)

            if not last:
                mandelbrot.zoom_in(self.rng.choice(edges))

    def show_image_file(self, log):
        if self.img_dir.is_dir():
            log.push_time('show image file')
            img_paths = list(self.img_dir.iterdir())
            success = False
            while not success and img_paths:
                img_path = self.rng.choice(img_paths)
                log.print(img_path)

                try:
                    parts = img_path.name.split('.')
                    method = 'pad'
                    if len(parts) >= 3:
                        show_end = -1
                        if parts[-2] in {'pad', 'fit'}:
                            method = parts[-2]
                            show_end = -2
                        show = '.'.join(parts[:show_end])
                    else:
                        show = parts[0]

                    with Image.open(img_path) as orig_img:
                        img = orig_img.convert(image_mode)

                    if method == 'pad':
                        img = ImageOps.pad(img, self.out.resolution, color=(255, 255, 255, 0))
                    elif method == 'fit':
                        img = ImageOps.fit(img, self.out.resolution)
                    else:
                        raise RuntimeError(f'Unknown method {method}')

                    # TODO
                    # if self.text:
                    #     img = self.info.write(img, img_path.name)
                    self.show_img(img)
                    success = True
                except Exception:
                    traceback.print_exc()
                    img_paths.remove(img_path)
            interval = log.pop_time()
            if success:
                self.sleep_min(log, interval)

    def generate_and_show(self, loop):
        while True:
            log = Log()
            try:
                self._generate_and_show(log)
            except Exception:
                if loop:
                    traceback.print_exc()
                else:
                    raise
            if not loop:
                break
            self.show_image_file(log)


def parse_size(s):
    return [int(n) for n in s.split('x')]


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--size', type=parse_size)
    arg_parser.add_argument('--skip-size', type=parse_size)
    arg_parser.add_argument('--count', '-c', type=int, default=15)
    arg_parser.add_argument('--skip', '-s', type=int)
    arg_parser.add_argument('--min-interval', '-i', type=int)
    arg_parser.add_argument('--seed', type=int)
    arg_parser.add_argument('--no-text', action='store_false', dest='text')
    arg_parser.add_argument('--workers', '-j', type=int, default=None)
    arg_parser.add_argument('-p', type=int, default=2)
    arg_parser.add_argument('--no-smooth', action='store_false', dest='smooth')
    arg_parser.add_argument('--no-sat', action='store_false', dest='sat')
    arg_parser.add_argument('--no-dynamic-color', action='store_false', dest='dynamic_color')
    arg_parser.add_argument('--force-python-impl', action='store_true')
    arg_parser.add_argument('--inky', action='store_true')
    arg_parser.add_argument('--loop', action='store_true', default=False)
    arg_parser.add_argument('--data', type=Path, default=Path(__file__).parent)
    args = arg_parser.parse_args()

    if args.force_python_impl:
        use_mandelbrot_native_impl = False

    if args.inky:
        from inky.auto import auto as Inky

        out = Inky(ask_user=True, verbose=True)
        skip_default = 3
        min_interval_default = 60 * 10

        try:
            setup_pi_buttons()
        except Exception:
            traceback.print_exc()
    else:
        out = FileOutput()
        skip_default = 0
        min_interval_default = 0
    resolution = args.size or out.resolution
    min_skip_res = (50, 30)
    skip_resolution = args.skip_size or \
        [max(min_skip_res[i], n // 10) for i, n in enumerate(resolution)]

    skip = skip_default if args.skip is None else args.skip
    min_interval = min_interval_default if args.min_interval is None else args.min_interval
    seq = Sequence(args.data, resolution, skip_resolution, out, args.count, skip, min_interval,
        args.seed, args.text, args.workers, args.p, args.smooth, args.sat, args.dynamic_color)

    print(f'{seq.resolution[0]}x{seq.resolution[1]}')
    print(f'seed: {seq.seed}')
    if use_mandelbrot_native_impl:
        print('Using Native Implementation')
    else:
        print('Using Python Implementation')

    seq.generate_and_show(args.loop)
