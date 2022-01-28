# -*- coding: utf-8 -*-
# 功能描述   ：把labelimg2标注的xml文件转换成dota能识别的txt文件
#             就是把旋转框 cx,cy,w,h,angle，转换成四点坐标x1,y1,x2,y2,x3,y3,x4,y4

import os
import xml.etree.ElementTree as ET
import math

classes=['zhawa']
def edit_xml(xml_file):
    """
    修改xml文件
    :param xml_file:xml文件的路径
    :return:
    """
#    print(xml_file)
    tree = ET.parse(xml_file)
    objs = tree.findall('object')
    out_file = open('labelTxt/%s.txt'%(xml_file.split('/')[-1].split('.')[0]),'w',encoding='UTF-8')
    for ix, obj in enumerate(objs):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)

        obj_bnd = obj.find('robndbox')
        obj_cx = obj_bnd.find('cx')
        obj_cy = obj_bnd.find('cy')
        obj_w = obj_bnd.find('w')
        obj_h = obj_bnd.find('h')
        obj_angle = obj_bnd.find('angle')
        cx = float(obj_cx.text)
        cy = float(obj_cy.text)
        w = float(obj_w.text)
        h = float(obj_h.text)
        angle = float(obj_angle.text)

        x0, y0 = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
        x1, y1 = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
        x2, y2 = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
        x3, y3 = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)
        bb=[x0,y0,x1,y1,x2,y2,x3,y3]
        out_file.write(str(cls_id) +" " +" ".join([str(a) for a in bb])+  '\n')

# 转换成四点坐标
def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc;
    yoff = yp - yc;
    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return str(int(xc + pResx)), str(int(yc + pResy))

if __name__ == '__main__':
    dir='./annotations'
    filelist = os.listdir(dir)
    for file in filelist:
        edit_xml(dir+'/'+file)
