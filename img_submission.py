# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 10:10:27 2018

@author: haoyu
"""

import pickle 
import numpy as np

def create_img_submission_file(orig_sar_img_arr,proc_sar_img_arr,x_vec,y_vec):
   submission = {}
   submission['orig_sar_img'] = np.array(orig_sar_img_arr)
   submission['proc_sar"img'] = np.array(proc_sar_img_arr)
   submission['x_axis'] = x_vec
   submission['y_axis'] = y_vec
   with open('group_3_SAR_img.pkl', 'wb') as handle:
       pickle.dump(submission, handle, protocol=pickle.HIGHEST_PROTOCOL)


