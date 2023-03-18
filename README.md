# python-
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#进度条：
import imageio as igo
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 读取gif，将每一帧存储在pics数组中
pics=igo.mimread('cxk1.gif')#读取图片(可照片、可gif)
a=np.array(pics)
b=a.shape
e=b[0]
d=int(100/e)
#print("gif图的帧数是{:}，像素是{:}x{:}，尺寸是：{:}",format(inf1,inf2))
print("照片的矩阵规模是:",a.shape)
print("照片的维度是:",a.ndim)
A = []
#使标注x轴的文字可以以中文正常显示
matplotlib.rc("font",family='Microsoft YaHei')
plt.xlabel("转化进度")

# 允许出现的字符数
string = '~!@#$%^&*()_+-{}|":?><[]\;'
count = len(string)


for g in range(1,100):
    y=g+1
    x=np.arange(0,100,d)
    plt.barh(x,y)
    plt.show()
    

# 对每一帧的图片进行处理
for img in pics:
    u,v,_ = img.shape#'_'是为了平衡赋值，储存了一个不关键的信息——gif的尺寸
    c = img*0+255#数组是可以对里面元素进行更改，乘0是里面每个元素都变成0，255是把面板上所以格都涂成白色
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转成灰度图（灰色有深有浅）
    for i in range(0,u,6):#间隔6格赋一次颜色
        for j in range(0,v,6):
            pix = gray[i, j]#取颜色（灰色的深浅）
            b, g, r, _ = img[i, j]#对于BGR读色顺序
            zifu = string[int(((count-1) * pix) / 256)]#0-255，共256个颜色，同程度灰色取同一种符号
            #字符图的组成成分——字符的实现
            #这里这个减一不是很严重，有没有都可以
            #pix实现字符的层次性，而不是乱排
            cv2.putText(c, zifu, (j, i), cv2.FONT_HERSHEY_COMPLEX, 0.2, (int(b), int(g), int(r), 1))
            #对每一帧图像进行处理
       
    # 色度处理的图片存储于数组
    
    A.append(c)
print("已全部完成！")
    
# 存储成新的gif
igo.mimsave('成品.gif',A,'GIF',duration = 0.1)

#if选择：
import cv2
from PIL import Image
import imageio as igo
import numpy as np

# 读取gif，将每一帧存储在pics数组中
pics=igo.mimread('cxk.gif')#读取图片(可照片、可gif)
a=np.array(pics)
#print("gif图的帧数是{:}，像素是{:}x{:}，尺寸是{:}",format(inf1,inf2)
A = []

# 允许出现的字符数
string = '~!@#$%^&*()_+-{}|":?><[]\;'
count = len(string)

# 对每一帧的图片进行处理
jd=0#进度条
for img in pics:
    u,v,_ = img.shape#'_'是为了平衡赋值，储存了一个不关键的信息——gif的尺寸
    c = img*0+255#数组是可以对里面元素进行更改，乘0是里面每个元素都变成0，255是把面板上所以格都涂成白色
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转成灰度图（灰色有深有浅）
    for i in range(0,u,6):#间隔6格赋一次颜色
        for j in range(0,v,6):
            pix = gray[i, j]#取颜色（灰色的深浅）
            b, g, r, _ = img[i, j]#对于BGR读色顺序
            zifu = string[int(((count-1) * pix) / 256)]#0-255，共256个颜色，同程度灰色取同一种符号
            #字符图的组成成分——字符的实现
            #这里这个减一不是很严重，有没有都可以
            #pix实现字符的层次性，而不是乱排
            cv2.putText(c, zifu, (j, i), cv2.FONT_HERSHEY_COMPLEX, 0.2, (int(b), int(g), int(r), 1))
            #对每一帧图像进行处理
    # 色度处理的图片存储于数组
    A.append(c)
    jd+=1
    print("已完成:",jd,"%")
print("已全部完成！")
print("gif图的帧数是:{}，像素是:{}x{}".format(a.shape[0],a.shape[1],a.shape[2]))
    
# 存储成新的gif
igo.mimsave('成品.gif',A,'GIF',duration = 0.1)


char_set = '''*&^%$$#!)(*&^%$#@!'''


im = Image.open('QQ.jpg')
im = im.resize((80, 50), Image.ANTIALIAS)
im = im.convert('L')    # 转为黑白图, 每个像素都一个灰度值,从0到255, 0是黑色, 255是白色
im.save('t.jpeg')


def get_char(gray):
    if gray >= 240:
        return ' '
    else:
        return char_set[int(gray/((256.0 + 1)/len(char_set)))]

text = ''
for i in range(im.height):
    for j in range(im.width):
        gray = im.getpixel((j, i))      # 返回值可能是一个int, 也可能是一个三元组
        if isinstance(gray, tuple):
            gray = int(0.2126 * gray[0] + 0.7152 * gray[1] + 0.0722 * gray[2])

        text += get_char(gray)
    text += '\n'

with open('pic.txt', 'w')as f:
    f.write(text)
    f.close()
with open('pic.txt', 'r')as f:
    for line in f:
        print(line)
    f.close()

#信息打印：
import imageio as igo
import cv2
import numpy as np

# 读取gif，将每一帧存储在pics数组中
pics=igo.mimread('sana.gif')#读取图片(可照片、可gif)
a=np.array(pics)
#print("gif图的帧数是{:}，像素是{:}x{:}，尺寸是{:}",format(inf1,inf2))
print("照片的矩阵规模是:",a.shape)
print("照片的维度是:",a.ndim)
print("gif图的帧数是:{}，像素是:{}x{}".format(a.shape[0],a.shape[1],a.shape[2]))
A = []

# 允许出现的字符数
string = '~!@#$%^&*()_+-{}|":?><[]\;'
count = len(string)

# 对每一帧的图片进行处理
jd=0#进度条
for img in pics:
    u,v,_ = img.shape#'_'是为了平衡赋值，储存了一个不关键的信息——gif的尺寸
    c = img*0+255#数组是可以对里面元素进行更改，乘0是里面每个元素都变成0，255是把面板上所以格都涂成白色
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转成灰度图（灰色有深有浅）
    for i in range(0,u,6):#间隔6格赋一次颜色
        for j in range(0,v,6):
            pix = gray[i, j]#取颜色（灰色的深浅）
            b, g, r, _ = img[i, j]#对于BGR读色顺序
            zifu = string[int(((count-1) * pix) / 256)]#0-255，共256个颜色，同程度灰色取同一种符号
            #字符图的组成成分——字符的实现
            #这里这个减一不是很严重，有没有都可以
            #pix实现字符的层次性，而不是乱排
            cv2.putText(c, zifu, (j, i), cv2.FONT_HERSHEY_COMPLEX, 0.2, (int(b), int(g), int(r), 1))
            #对每一帧图像进行处理
    # 色度处理的图片存储于数组
    A.append(c)
    jd+=1
    print("已完成:",jd,"%")
print("已全部完成！")
    
# 存储成新的gif
igo.mimsave('成品.gif',A,'GIF',duration = 0.1)

import cv2
from PIL import Image

char_set = '''$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. '''
imagess = cv2.imread(r'C:/Users/HONOR/Desktop/noeasy.jpeg',-1)
print("图片的像素是:{}x{}".format(imagess.shape[0],imagess.shape[1]))


im = Image.open('noeasy.jpeg')
im = im.resize((80, 50), Image.ANTIALIAS)
im = im.convert('L')    # 转为黑白图, 每个像素都一个灰度值,从0到255, 0是黑色, 255是白色
im.save('t.jpeg')


def get_char(gray):
    if gray >= 240:
        return ' '
    else:
        return char_set[int(gray/((256.0 + 1)/len(char_set)))]

text = ''
for i in range(im.height):
    for j in range(im.width):
        gray = im.getpixel((j, i))      # 返回值可能是一个int, 也可能是一个三元组
        if isinstance(gray, tuple):
            gray = int(0.2126 * gray[0] + 0.7152 * gray[1] + 0.0722 * gray[2])

        text += get_char(gray)
    text += '\n'

with open('pic.txt', 'w')as f:
    f.write(text)
    f.close()
with open('pic.txt', 'r')as f:
    for line in f:
        print(line)
    f.close()

