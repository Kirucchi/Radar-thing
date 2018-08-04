# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 16:56:35 2018

@author: haoyu
"""

'''
Displays a grayscaled version of arr which contains the values for the pixels
'''
def gray_scale(arr):
    plt.imshow(arr, cmap='gray')

'''
Plots a picture that illustrates the edges of the source image
@param   arr   the 2D array or list of lists that contains the sum magnitude of pulses
@param   diff  the index that dictates how much contrast the algorithm is looking for
'''
def edge_detection(arr):
    '''ED = np.zeros((len(arr),len(arr[0]),3))
    for row in range(len(arr)):
        for col in range(len(arr[0])):
           # if col is not len(arr[0]) - 1:
            if (col < len(arr[0]) - 1 and np.absolute(arr[row][col]-arr[row][col+1]) > diff) or (row < len(arr) - 4 and np.absolute(arr[row][col]-arr[row+4][col]) > diff*4):
                ED[row,col,0] = 255
                ED[row,col,1] = 255
                ED[row,col,2] = 255
            else:
                ED[row,col,0] = 1
                ED[row,col,1] = 1
                ED[row,col,2] = 1
    return ED'''
    edge_horizont = ndimage.sobel(arr, 0)
    edge_vertical = ndimage.sobel(arr, 1)
    return(np.hypot(edge_horizont, edge_vertical))