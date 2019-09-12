from astropy.io import fits
import numpy as np
import math
import time

def deconvolve(dirty, dirtyWidth, psf, psfWidth):
    residual = dirty  
    model = np.zeros((dirty.shape[0]))
    psfPeakVal = 0.0  
    psfPeakPos = 0  
    iteration = 1000  
    threshold = 0.00001 
    gain = 0.1 
    findPeak(psf, psfPeakVal, psfPeakPos)  
    for i in range(iteration): 
        absPeakVal = 0.0
        absPeakPos = 0
        absPeakVal, absPeakPos = findPeak(residual, absPeakVal, absPeakPos)
        if abs(absPeakVal) < threshold: 
            print("Reached stopping threshold")
            break
        model[absPeakPos] += absPeakVal * gain 
        subtractPSF(psf, psfWidth, residual, dirtyWidth, absPeakPos, psfPeakPos, absPeakVal) 
    return model, residual

def findPeak(image, maxVal, maxPos):
    maxVal = maxVal
    maxPos = maxPos
    size = image.size

    for i in range(size):
            if abs(image[i, 0]) > abs(maxVal):
                maxVal = image[i, 0]
                maxPos = i
    return maxVal, maxPos

def idxToPos(idx, width):
    x = idx / width
    y = idx % width
    return x, y

def posToIdx(width, x, y):
    return y * width + x

def subtractPSF(psf, psfWidth, residual, residualWidth, peakPos, psfPeakPos, absPeakVal):
    gain = 0.1
    rx, ry = idxToPos(peakPos, residualWidth)
    px, py = idxToPos(psfPeakPos, psfWidth)
    diffx = rx - px
    diffy = ry - py
    startx = max(0, diffx)
    starty = max(0, diffy)
    stopx = min(residualWidth - 1, rx + (psfWidth - px - 1))
    stopy = min(residualWidth - 1, ry + (psfWidth - py - 1))
    starty = int(starty)
    stopy = int(stopy)
    startx = int(startx)
    stopx = int(stopx)

    for y in range(starty, stopy + 1):
        for x in range(startx, stopx + 1):
            index1 = posToIdx(residualWidth, x, y)
            index2 = posToIdx(psfWidth, x - diffx, y - diffy)
            index1 = int(index1)
            index2 = int(index2)
            residual[index1] -= gain * absPeakVal * psf[index2]


def main():
    dfu = fits.open('256/imaging_dirty.fits')  
    pfu = fits.open('256/imaging_psf.fits')
    dirty1 = np.mat(dfu[0].data)  
    dirty = dirty1.reshape(-1, 1)  
    psf1 = np.mat(pfu[0].data)
    psf = psf1.reshape(-1, 1)
    dirtysize = int(math.sqrt(dirty.size))  
    psfsize = int(math.sqrt(psf.size))
    start = time.time() 
    model, residual = deconvolve(dirty, dirtysize, psf, psfsize)
    end = time.time()
    print("image dimensions = ", dirtysize, "X", dirtysize)
    print("time = "'%.3f' % (end - start)) 

if __name__ == '__main__':
    main()
