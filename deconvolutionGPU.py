from astropy.io import fits
import numpy as np
import math
import time
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

def deconvolve(dirty, dirtyWidth, psf, psfWidth):
    mod = SourceModule("""
        #include<stdio.h>
        #include<stdlib.h>
        __global__ void subpsf_kernel(
        float *psf,
        float *residual,
        int starty,
        int stopy,
        int startx,   
        int stopx,
        int dirtyWidth,
        int psfWidth,
        float diffx,
        int diffy,
        float gain,
        float absPeakVal)
    {
       int idy = blockIdx.y*blockDim.y+threadIdx.y;
       int idx = blockIdx.x*blockDim.x+threadIdx.x;
       int y=idy+starty;
       int x=idx+startx;
       if((y<stopy+1)&&(x<stopx+1)){
          int index1 = y*dirtyWidth+x;
          int index2 = (int)(y-diffy)*psfWidth+x-diffx;        
          residual[index1] -= gain * absPeakVal * psf[index2];
       }
    }
    """)
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
        gain=0.1
        rx, ry = idxToPos(absPeakPos,dirtyWidth)
        px, py = idxToPos(psfPeakPos, psfWidth)
        diffx = rx - px
        diffy = ry - py
        startx = max(0, diffx)
        startx=int(startx)
        starty = max(0, diffy)
        stopx = min(dirtyWidth - 1, rx + (psfWidth - px - 1))
        stopy = min(dirtyWidth - 1, ry + (psfWidth - py - 1))
        startx = int(startx)
        psf_array=np.array(psf)
        residual_array=np.array(residual)
        psf_gpu=drv.mem_alloc_like(psf_array)
        strm = drv.Stream()
        drv.memcpy_htod_async(psf_gpu,psf_array,strm)
        subpsf_kernel = mod.get_function("subpsf_kernel")
        strm.synchronize()
        subpsf_kernel(psf_gpu,
                      drv.Out(residual_array),
                      np.int32(starty),
                      np.int32(stopy),
                      np.int32(startx),
                      np.int32(stopx),
                      np.int32(dirtyWidth),
                      np.int32(psfWidth),
                      np.float32(diffx),
                      np.int32(diffy),
                      np.float32(gain),
                      np.float32(absPeakVal),
                      block=(32, 32, 1),
                      grid=(8, 8, 1)
                      )
        residual=residual_array
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
