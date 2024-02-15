import pickle as pkl
import torch
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFont
import os, cv2
import json


class Draw:
    maxalpha = 255
    def __init__(self, width=22, height=22, **kw):
        self.width, self.height = width, height
        
    def draw(self, canvas, box_x=0, box_y=0, opacity=1.0): 
        localcanvas = canvas.crop((box_x, box_y, box_x + self.get_width(), box_y + self.get_height()))
        self._draw(localcanvas, opacity=opacity)
        canvas.paste(localcanvas, (box_x, box_y))
        
    def _draw(self, canvas, opacity=1.0): pass
    def drawself(self):
        canvas = Image.new('RGBA', (self.get_width(), self.get_height()), (255, 255, 255, 0))
        self.draw(canvas, 0, 0, 1)
        return canvas
    
    def get_width(self): return self.width
    def get_height(self): return self.height
    
        
class OverlayDraw(Draw):
    def __init__(self, base:Draw, overlay:Draw, *args, overlay_opacity=1.0, **kw):
        self.base, self.overlay = base, overlay
        assert self.base.get_width() == self.overlay.get_width()
        assert self.base.get_height() == self.overlay.get_height()
        self.overlay_opacity = overlay_opacity
        
    def get_width(self): return self.base.get_width()
    def get_height(self): return self.base.get_height()
    
    def _draw(self, canvas, opacity=1.0):
        self.base._draw(canvas, opacity=opacity)
        self.overlay._draw(canvas, opacity=opacity * self.overlay_opacity)

        
class DrawImage(Draw):
    def __init__(self, image, imgsize=256, rotate=0, **kw):
        super().__init__(**kw)
        self.image = image.convert("RGBA")
        if rotate != 0:
            self.image = self.image.rotate(-rotate)
        self.imgsize = (imgsize, imgsize) if isinstance(imgsize, int) else imgsize
        self.width, self.height = self.imgsize
        
    def _draw(self, canvas, opacity=1.0):
        img = self.image.resize(self.imgsize)
        img.putalpha(int(opacity * self.maxalpha))
        canvas.alpha_composite(img)
    
    
class DrawText(Draw):
    def __init__(self, text, fontsize=28, width=3*256+2*3, height=100, offset=(0,0), bold=False, italic=False, condensed=False, color=(0, 0, 0), **kw):
        super().__init__(**kw)
        self.text, self.fontsize, self.width, self.height = text, fontsize, width, height
        
        font_name = "DejaVuSans" if not condensed else "DejaVuSansCondensed"
        font_name_append = ""
        if bold:
            font_name_append += "Bold"
        if italic: 
            font_name_append += "Oblique"
        
        font_path = os.path.join(cv2.__path__[0],'qt','fonts', 
                font_name + ("-" + font_name_append if font_name_append != "" else "") + '.ttf')
        print(font_path)
        
        self.font = font = ImageFont.truetype(font_path, self.fontsize)
        self.textcolor = color
        self.offset = offset
        
    def _draw(self, canvas, opacity=1.0):
        txtimg = Image.new("RGBA", (self.get_width(), self.get_height()), (255,255,255,0))
        draw = ImageDraw.Draw(txtimg)
        x = self.width // 2 + self.offset[0]
        y = self.height // 2 + self.offset[1]
        draw.text((x, y), self.text, fill=self.textcolor + (int(opacity * self.maxalpha),), font=self.font, anchor="mm", align="center")
        canvas.alpha_composite(txtimg)
        
        
class DrawRow(Draw):
    def __init__(self, *items, margin=3, **kw):
        super().__init__(**kw)
        self.kw = kw
        self.items = items
        self.margin = margin
        
    def get_width(self):
        ret = sum([item.get_width() for item in self.items])
        ret += self.margin * (len(self.items) - 1)
        return ret
        
    def get_height(self):
        return max([item.get_height() for item in self.items])
        
    def __add__(self, other):
        ret = type(self)(*(self.items + other.items), margin=self.margin, **self.kw)
        return ret
    
    def draw(self, canvas, x=0, y=0, opacity=1.0):
        for item in self.items:
            item.draw(canvas, x, y, opacity=opacity)
            x += item.get_width() + self.margin
            
            
class DrawCol(DrawRow):
    def get_width(self):
        return max([item.get_width() for item in self.items])
        
    def get_height(self):
        ret = sum([item.get_height() for item in self.items])
        ret += self.margin * (len(self.items) - 1)
        return ret
    
    def draw(self, canvas, x=0, y=0, opacity=1.0):
        for item in self.items:
            item.draw(canvas, x, y, opacity=opacity)
            y += item.get_height() + self.margin