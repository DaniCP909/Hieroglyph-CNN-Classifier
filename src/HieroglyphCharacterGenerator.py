from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import numpy as np
import random
import os

#FUENTE (inicio fin name, utf-8, Decimal)
#EGYPTIAN HIEROGLYPH A001: U+ 13000 , 77824
#
#EGYPTIAN HIEROGLYPH AA032: U+ 1342E , 78894


class HieroglyphCharacterGenerator:

    paths = {1: "/home/dcorr/Documents/Fonts/NOTO SANS EGYPTIAN HIEROGLYPHS/Noto_Sans_Egyptian_Hieroglyphs/NotoSansEgyptianHieroglyphs-Regular.ttf",
             2: "/home/dcorr/Documents/Fonts/NEW GARDINER/NewGardiner/NewGardinerSMP.ttf"
             }
    
    ranges = {1: (0x00013000, 0x0001342E),
              2: (0x0000E000, 0x0000E42E)
              }

    start_hex_noto = 0x00013000
    end_hex_noto = 0x0001342E
    
    start_hex_new_gardiner = 0x0000E000
    end_hex_new_gardiner = 0x0000E42E


    #font 1-2, size 1-1
    def __init__(self, font_id=1 , font_path=None, size=270, start_hex=None, end_hex=None):
        if(font_id <= len(self.paths)):
            self.id_selected_font = font_id  #index in sict
        else:
            self.id_selected_font = 1

        if(None not in (font_path, start_hex, end_hex)):
            if(os.path.exists(font_path)):
                new_idx = len(self.paths) + 1
                self.paths.update({new_idx: font_path})
                self.ranges.update({new_idx: (start_hex, end_hex)})
                self.id_selected_font = new_idx
        self.font = ImageFont.truetype(self.paths.get(self.id_selected_font),
                                 size, 
                                 encoding="unic")
        start, end = self.ranges.get(self.id_selected_font)
        self.hex_range = [x for x in range(start, end)]
        

    #def dec2hexInRange(self, decimal):
    #    if(decimal > len(self.hex_range)):
    #        return -1
    #    map_dec_hex = map(+1, self.hex_range)
    #    print(map_dec_hex)
     
    def getHexRange(self):
        return self.hex_range

    def getCharCanvasByHex(self, hex):
        canvas = Image.new("RGB", (500,500), "black")
        draw = ImageDraw.Draw(canvas)
        ucode = chr(hex)
        draw.text((15,0), ucode, "white", self.font)
        return canvas
    
    def getCharCanvasByDec(self, decimal):
        return self.getCharCanvasByHex(self.dec2hexInRange(decimal))
    
    def getImageByHex(self, hex):
        return np.array(self.getCharCanvasByHex(hex))
    
    def getImageByDec(self, decimal):
        return self.getImageByHex(self.dec2hexInRange(decimal))

