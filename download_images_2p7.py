import urllib2
import urllib
import json
import numpy as np
import cv2
import untangle
import os
import matplotlib.pyplot as plt

# The API template for pulls is given by Safebooru https://safebooru.org/index.php?page=help&topic=dapi
# /index.php?page=dapi&s=post&q=index
maxsize = 512
imagecounter = 3065
maxImages = 10000
pagestepper = 0
pageoffset = 50
tags = 'holding+staff'
savepath = 'Images_holding_staff'

while imagecounter < maxImages:
    #Get a taged page
    safebooruXMLPage = urllib2.urlopen(
        "http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=" +
        tags + "&pid=" + str(pageoffset + pagestepper)).read()
    pagestepper += 1

    xmlreturn = untangle.parse(safebooruXMLPage)
    for post in xmlreturn.posts.post:
        imgurl = post["sample_url"]
        if ("png" in imgurl) or ("jpg" in imgurl):
            resp = urllib.urlopen(imgurl)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if isinstance(image, type(None)):
                # this happens when we run into 404 error on website
                continue
            print('counter: {}. URL: {}'.format(imagecounter, imgurl))

            height, width = image.shape[:2]
            # Resize but preserve aspect ratio-- this requires we crop and cut the image blindly!
            if height > width:
                scalefactor = (maxsize * 1.0) / width
                res = cv2.resize(
                    image,
                    (int(width * scalefactor), int(height * scalefactor)),
                    interpolation=cv2.INTER_CUBIC)
                cropped = res[0:maxsize, 0:maxsize]
            if width > height:
                scalefactor = (maxsize * 1.0) / height
                res = cv2.resize(
                    image,
                    (int(width * scalefactor), int(height * scalefactor)),
                    interpolation=cv2.INTER_CUBIC)
                center_x = int(round(width * scalefactor * 0.5))
                cropped = res[0:maxsize,
                              center_x - maxsize / 2:center_x + maxsize / 2]

            # so we now have resized/cropped image pulled from the website
            cv2.imwrite(
                savepath + '/' + str(imagecounter) + '_page_' +
                str(pagestepper + pageoffset) + ".jpg", cropped)
            if imagecounter == maxImages:
                break
            else:
                imagecounter += 1