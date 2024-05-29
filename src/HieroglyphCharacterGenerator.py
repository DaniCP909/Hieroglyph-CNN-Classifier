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

    paths = [ 
        "./files/fonts/Noto_Sans_Egyptian_Hieroglyphs/NotoSansEgyptianHieroglyphs-Regular.ttf",
        "./files/fonts/NewGardiner/NewGardinerBMP.ttf",
            ]
    
    
    ranges = [ 
        (0x00013000, 0x0001342E),
        (0x0000E000, 0x0000E42E),
            ]

    start_hex_noto = 0x00013000
    end_hex_noto = 0x0001342E
    
    start_hex_new_gardiner = 0x0000E000
    end_hex_new_gardiner = 0x0000E42E


    #font 1-2, size 1-1
    def __init__(self,
                 font_path: str,
                 start_hex,
                 end_hex,
                 font_size=270,
                 ):
        
        if(None not in (font_path, start_hex, end_hex) and start_hex <= end_hex):
            if(os.path.exists(font_path)):
                self.font_path = font_path
                self.font_size = font_size
                self.font = ImageFont.truetype(font_path,
                                 font_size, 
                                 encoding="unic")
                self.start_hex, self.end_hex = (start_hex, end_hex)
                self.hex_range = [x for x in range(self.start_hex, self.end_hex)]
        else:
            print("Error arguments")

    def getMinFromFont(self):
        return self.start_hex
    
    def getMaxFromFont(self):
        return self.end_hex
    
    def getFontLength(self):
        return self.end_hex - self.start_hex
    
    

    #def dec2hexInRange(self, decimal):
    #    if(decimal > len(self.hex_range)):
    #        return -1
    #    map_dec_hex = map(+1, self.hex_range)
    #    print(map_dec_hex)

    def label2offset(self,
                   label: int,
                   ):
        if(label < 0 or label > self.getMaxFromFont()):     #ojo
            print("Error: label out of range")
            return (-1)
        
        #se devuelve la posicion deseada (desde 0 hasta selected_font_max)
        return self.getMinFromFont() + label
        
     
    def getFontRange(self, id_font):
        return self.hex_range

    def getCharCanvasByOffset(self, offset):
        #para font_size = 270 => tamaño canvas aprox 500x500
        canvas = Image.new("RGB", (2 * (self.font_size + 20), 2 * (self.font_size + 20)), "black")
        draw = ImageDraw.Draw(canvas)
        ucode = chr(offset)

        if("NewGardiner" in self.font_path): text_y = 100
        else: text_y = 0

        draw.text((200,text_y), ucode, "white", self.font)
        return canvas
    
    def getCharCanvasByLabel(self, label):
        return self.getCharCanvasByOffset(self.label2offset(label))
    
    def getImageByOffset(self, offset):
        return np.array(self.getCharCanvasByOffset(offset))
    
    def getImageByLabel(self, label):
        return self.getImageByOffset(self.label2offset(label))

