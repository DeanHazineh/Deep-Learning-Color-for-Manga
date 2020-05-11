import cv2
import os
from glob import glob
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import shutil
import scipy.signal as scig
from scipy import fftpack

# Helper Functions
def addAxis(thisfig, n1, n2):
    axlist = []
    for i in range(n1 * n2):
        axlist.append(thisfig.add_subplot(n1, n2, i + 1))
    return np.array(axlist)

def groupFormat(axisList):
    # remove ticks and labels on all axis
    for ax in axisList:
        ax.set_xticks([])
        ax.set_yticks([])

def delFold_RemakeFold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    return

# Main Utility Functions
def read_and_reshape(imgpath):
    # This function will load in a single image given the image folder path
    # and will then load, resize to 256x256
    # return the loaded, colored image as np array

    readImage = cv2.imread(imgpath)
    readImage = cv2.cvtColor(readImage, cv2.COLOR_BGR2RGB)
    readImage = np.array(cv2.resize(readImage, (256, 256)))

    return readImage

def cvSketchify(image_np, blockSize=7):
    # This code takes in a single colored image and first converts
    # it to grayscale. After, it then uses CV edge detection to
    # convert to sketches
    # Play around with the sketchify settings until it looks right
    edgetype = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    #edgetype = cv2.ADAPTIVE_THRESH_MEAN_C
    grayscale = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    sketchified = np.array(
        cv2.adaptiveThreshold(grayscale,
                              255,
                              edgetype,
                              cv2.THRESH_BINARY,
                              blockSize=blockSize,
                              C=2)) / 255.0

    return sketchified

def complexityMetric(bwImage):
    return (256 * 256 - np.sum(np.sum(bwImage))) / (256 * 256) * 100

def genColorCues(cimg):
    out = scig.fftconvolve(cimg, kernel3, mode='same', axes=(0,1))
    return out/np.amax(out)

# Commands used to make figures or analyze things for the Report
def imshowSave(img, savepath = [], color=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if color:
        ax.imshow(img)
    else:
        ax.imshow(img,cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if savepath:
            plt.savefig(savepath)
            plt.close()
    else:
        plt.close()
    return

def batchPlotImages(batchedColored, batchedSketchified, savepath=[], numplot=5):

    pickIdx = random.sample(range(0, batchedColored.shape[0]), numplot)

    for i, idx in enumerate(pickIdx):
        fig = plt.figure(figsize=(8, 8))
        axisList = addAxis(fig, 1, 2)
        axisList[0].imshow(batchedColored[idx])
        axisList[1].imshow(batchedSketchified[idx], cmap='gray')
        axisList[1].set_title('Complexity: ' +
                              str(complexityMetric(batchedSketchified[idx])))
        groupFormat(axisList)
        plt.tight_layout
        if savepath:
            plt.savefig(savepath + str(i) + '.png')
            plt.close()
        
        #if not savepath:
        #    plt.show()
    return

def batchPlotImages2(batchedColored, batchedSketchified, batchedColorCues, savepath=[], numplot=5):
    pickIdx = random.sample(range(0, batchedColored.shape[0]), numplot)
    for i, idx in enumerate(pickIdx):
        imshowSave(batchedColored[idx], savepath+str(i)+'_colored.png', color=True)
        imshowSave(batchedSketchified[idx], savepath+str(i)+'_sketch.png', color=False)
        imshowSave(batchedColorCues[idx], savepath+str(i)+'_Cues.png', color=True)

    return

def compareSketchifyBlockSize():
    ## This code following was written to show the sketchify process
    ## and make figures for the paper
    data = 'Images_magic_solo/5.jpg'
    colored = read_and_reshape(data)
    savepath = 'SavedImagesForPaper/CompareSketchifyBlockSize/'
    sketchMetric = np.arange(3, 15, 2)
    for blocksize in sketchMetric:
        sketchified = cvSketchify(colored, blocksize)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(sketchified, cmap='gray')
        ax.set_title('blockSize: {} complexity Percent: {}'.format(
            blocksize, int(100 * complexityMetric(sketchified))))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(savepath + str(blocksize) + '.png')
        plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(colored)
    ax.set_title('Original Colored (Downsampled + Cropped)')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(savepath + 'OriginalColor1.png')
    plt.close()
    return

def showDownloadComplexity(complexityValues, savepath=[]):
    plt.figure()
    plt.hist(complexityValues, bins=50)
    plt.title('Complexity Distribution of Downloaded Tag Set')
    plt.xlabel('Complexity Value')
    plt.ylabel('Number Images')
    if savepath:
        plt.savefig(savepath + 'HistogramComplexity'+ '.png')
        plt.close()
    return


tag = 'RawImages/Images_holding_staff'
data = glob(os.path.join(tag, "*"))
batchsize = len(data)
complexityCeiling = 40
save = False
sigma = 30

# Run Code Processing
batchedColored = np.array(
    [read_and_reshape(imagepath) for imagepath in data[0:batchsize]])
batchedSketchified = np.array(
    [cvSketchify(image, blockSize=7) for image in batchedColored])

complexityValues = np.array(
    [complexityMetric(bwimage) for bwimage in batchedSketchified])

complexityThreshold = np.squeeze(np.array(np.where(complexityValues < complexityCeiling)))
batchedColored_Thresh = batchedColored[complexityThreshold]
batchedSketchified_Thresh = batchedSketchified[complexityThreshold]

# Create Color Cues
x = np.arange(0, 256) - 128
X,Y = np.meshgrid(x,x)
kernel = 1/(sigma**2 *2*np.pi) * np.exp(-1/2 *(X**2 + Y**2)/sigma**2)
kernel3 = np.repeat(np.expand_dims(kernel,2),3,2)
batchedColorCues_Thresh = np.array(
   [genColorCues(cimg) for cimg in batchedColored_Thresh])

if save:
    # save the batched images to file
    print(batchedColored_Thresh.shape)
    print(batchedSketchified_Thresh.shape)
    print(batchedColorCues_Thresh.shape)
    filename = tag + '_CollectionProcessed.pickle'
    with open(filename, 'wb') as f:
        data = {
            'batchedColored': batchedColored_Thresh,
            'batchedSketchified': batchedSketchified_Thresh,
            'batchedColorCues': batchedColorCues_Thresh
        }
        pickle.dump(data, f, protocol=4)
    print('Pickle File Saved')

if save:
    savepath = 'SaveImagesForPaper/' + tag + '/'
    delFold_RemakeFold(savepath)
else: 
    savepath = []
#batchPlotImages(batchedColored_Thresh, batchedSketchified_Thresh, savepath=savepath, numplot=5)
#batchPlotImages2(batchedColored_Thresh, batchedSketchified_Thresh, batchedColorCues_Thresh, savepath=savepath, numplot=5)
showDownloadComplexity(complexityValues, savepath=savepath)
plt.show()



