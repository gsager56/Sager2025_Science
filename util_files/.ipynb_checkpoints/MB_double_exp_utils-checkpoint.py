from neuprint import Client
from tifffile import imread
import numpy as np
from skimage import data, color, measure, morphology
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_erosion, convolve
import pandas as pd 
from os.path import isfile
from os import listdir
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from neuprint import fetch_synapses, NeuronCriteria as NC, SynapseCriteria as SC, MitoCriteria as MC
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.optimize import minimize


home_dir = '/Users/gs697/Research/positioning_paper'


def RelaxationLabeling(init_prob, num_iters = 5, delta = 0.5):
    prob_map = np.zeros( init_prob.shape )
    for i in range(num_iters):
        sup_matrix = 2 * init_prob - 1
        prob_map[:,:,:] = 0

        prob_map[1: ,:  ,:  ] += sup_matrix[:-1,:  ,:  ] / 6
        prob_map[:-1,:  ,:  ] += sup_matrix[1: ,:  ,:  ] / 6
        prob_map[:  ,1: ,:  ] += sup_matrix[:  ,:-1,:  ] / 6
        prob_map[:  ,:-1,:  ] += sup_matrix[:  ,1: ,:  ] / 6
        prob_map[:  ,:  ,1: ] += sup_matrix[:  ,:  ,:-1] / 6
        prob_map[:  ,:  ,:-1] += sup_matrix[:  ,:  ,1: ] / 6

        prob_map = init_prob + delta * prob_map

        init_prob = np.where( prob_map > 1, 1, prob_map)
        init_prob = np.where( prob_map < 0, 0, prob_map)
    return prob_map > 0.5

def is_connected(image_file):
    # some MBs are connected on either side of the brain, so here is a list of files where that happens
    image_files = [home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_1_970_150uW_00001.tif", 
                   home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_2_970_150uW_00001.tif", 
                   home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_5_970_150uW_00001.tif", 
                   home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250301_1_970_150uW_00001.tif", 
                   home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_2_970_150uW_00001.tif", 
                   home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_1_00002.tif", 
                   home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_2_00001.tif", 
                   home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_1_970_150uW_00001.tif", 
                   home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_2_00001.tif", 
                   home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250307_5_970_250uW_00001.tif"]
    if image_file in image_files:
        return True
    return False

def get_intensity_thresh(image_file, i_channel):
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_2_970.tif":
        if i_channel == 0:
            return [1.83251, 1.89763]
        elif i_channel == 1:
            return [2.07918, 2.14922]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_3_970.tif":
        if i_channel == 0:
            return [1.83251, 1.90309]
        elif i_channel == 1:
            return [2.08636, 2.15836]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_1_970nm_00001.tif":
        if i_channel == 0:
            return [1.76343, 1.89763]
        elif i_channel == 1:
            return [2, 2.14922]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_2_970nm_00001.tif":
        if i_channel == 0:
            return [1.80618, 1.86923]
        elif i_channel == 1:
            return [2.07555, 2.13988]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_4_970nm_00001.tif":
        if i_channel == 0:
            return [1.5563, 1.77085]
        elif i_channel == 1:
            return [1.89209, 2.04922]
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_1_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.27646, 3.18127]
        elif i_channel == 1:
            return [2.30535, 3.22089]
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_2_970_150uW_00002.tif":
        if i_channel == 0:
            return [2.28103, 2.44871]
        elif i_channel == 1:
            return [2.3075, 2.48572]
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_2_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.12057, 2.33846]
        elif i_channel == 1:
            return [2.11394, 2.38021]
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_5_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.13672, 2.44404]
        elif i_channel == 1:
            return [2.15836, 2.46687]
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_6_970_150uW_00003.tif":
        if i_channel == 0:
            return [2.13988, 2.47712]
        elif i_channel == 1:
            return [2.17026, 2.51055]
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250301_1_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.11727, 2.31387]
        elif i_channel == 1:
            return [2.12057, 2.35218]
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_1_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.16732, 2.34242]
        elif i_channel == 1:
            return [2.18469, 2.37658]
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_2_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.15836, 2.33846]
        elif i_channel == 1:
            return [2.15534, 2.38561]
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_1_00002.tif":
        if i_channel == 0:
            return [2.1271 , 2.40312]
        elif i_channel == 1:
            return [2.09691, 2.43297]
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_2_00001.tif":
        if i_channel == 0:
            return [2.07188, 2.40483]
        elif i_channel == 1:
            return [1.96379, 2.45179]
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_1_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.14301, 2.32838]
        elif i_channel == 1:
            return [2.11394, 2.35603]
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_2_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.14301, 2.32838]
        elif i_channel == 1:
            return [2.11394, 2.35603]
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_3_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.13033, 2.26007]
        elif i_channel == 1:
            return [2.08636, 2.30535]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_1_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.14301, 2.32428]
        elif i_channel == 1:
            return [2.13033, 2.32015]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_2_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.14922, 2.38561]
        elif i_channel == 1:
            return [2.17319, 2.39445]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_1_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.16137, 2.33244]
        elif i_channel == 1:
            return [2.15534, 2.33846]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_2_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.15534, 2.31806]
        elif i_channel == 1:
            return [2.15229, 2.33041]
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_3_970_150uW_00001.tif":
        if i_channel == 0:
            return [2.16435, 2.41162]
        elif i_channel == 1:
            return [2.21484, 2.43457]
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_2_00001.tif":
        if i_channel == 0:
            return [2.31387, 2.48287]
        elif i_channel == 1:
            return [2.36549, 2.54283]
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_3_00001.tif":
        if i_channel == 0:
            return [2.25768, 2.42325]
        elif i_channel == 1:
            return [2.29885, 2.4609 ]
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250225_1_970_250uW_00001.tif":
        if i_channel == 0:
            return [2.17026, 2.34635]
        elif i_channel == 1:
            return [2.22531, 2.43933]
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250225_2_970_250uW_00001.tif":
        if i_channel == 0:
            return [2.17319, 2.36922]
        elif i_channel == 1:
            return [2.25285, 2.46982]
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250225_3_970_250uW_00001.tif":
        if i_channel == 0:
            return [2.17319, 2.3927 ]
        elif i_channel == 1:
            return [2.26717, 2.51055]
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250307_2_970_250uW_00001.tif":
        if i_channel == 0:
            return [2.17319, 2.39794]
        elif i_channel == 1:
            return [2.29447, 2.54654]
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250307_5_970_250uW_00001.tif":
        if i_channel == 0:
            return [2, 2.35984]
        elif i_channel == 1:
            return [2.15229, 2.50106]

def get_bool_seg_idxs(image_file, seg_idxs, side):
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_2_970.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 50, 
                                     np.all([seg_idxs[:,0] > 40, seg_idxs[:,2] < 150, seg_idxs[:,1] < 185],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 50, 
                                     np.all([seg_idxs[:,0] > 48, seg_idxs[:,2] > 375, seg_idxs[:,1] < 170],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_3_970.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 90, 
                                     np.all([seg_idxs[:,0] > 39, seg_idxs[:,2] < 175, seg_idxs[:,1] < 205],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 90, 
                                     np.all([seg_idxs[:,0] > 43, seg_idxs[:,2] > 375, seg_idxs[:,1] < 200],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_1_970nm_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 150, 
                                     np.all([seg_idxs[:,0] > 35, seg_idxs[:,2] < 90],axis=0), 
                                     np.all([seg_idxs[:,0] > 37, seg_idxs[:,1] < 260, seg_idxs[:,2] < 170],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 150, 
                                     np.all([seg_idxs[:,0] > 45, seg_idxs[:,2] > 425],axis=0), 
                                     np.all([seg_idxs[:,0] > 47, seg_idxs[:,1] < 260, seg_idxs[:,2] > 350],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_2_970nm_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.all([seg_idxs[:,0] > 35, seg_idxs[:,1] < 215, seg_idxs[:,2] < 160],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,2] > 435, 
                                     np.all([seg_idxs[:,0] > 36, seg_idxs[:,1] < 175, seg_idxs[:,2] > 375],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_4_970nm_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 150, 
                                     np.all([seg_idxs[:,0] > 13, seg_idxs[:,2] < 125, seg_idxs[:,1] > 200],axis=0), 
                                     np.all([seg_idxs[:,0] > 25, seg_idxs[:,1] < 255, seg_idxs[:,2] < 170],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 150, 
                                     np.all([seg_idxs[:,0] > 34, seg_idxs[:,2] > 435, seg_idxs[:,1] > 205],axis=0), 
                                     np.all([seg_idxs[:,0] > 40, seg_idxs[:,1] < 250, seg_idxs[:,2] > 350],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_1_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 200, 
                                     seg_idxs[:,2] > 235, 
                                     np.all([seg_idxs[:,0] > 65, seg_idxs[:,1] < 325],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 200, 
                                     seg_idxs[:,2] < 235, 
                                     seg_idxs[:,2] > 405, 
                                     np.all([seg_idxs[:,0] > 70, seg_idxs[:,2] > 380],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_2_970_150uW_00002.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 175, 
                                     seg_idxs[:,2] > 255,
                                     np.all([seg_idxs[:,1] < 335, seg_idxs[:,2] > 150],axis=0), 
                                     np.all([seg_idxs[:,0] > 44, seg_idxs[:,1] < 325],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([np.all([seg_idxs[:,0] < 20, seg_idxs[:,1] > 350],axis=0), 
                                     seg_idxs[:,1] < 200, 
                                     seg_idxs[:,2] < 255,
                                     np.all([seg_idxs[:,2] > 385, seg_idxs[:,1] < 225], axis=0), 
                                     np.all([seg_idxs[:,1] < 300, seg_idxs[:,2] > 410], axis=0), 
                                     np.all([seg_idxs[:,2] > 425, seg_idxs[:,0] > 29],axis=0), 
                                     np.all([seg_idxs[:,1] < 345, seg_idxs[:,2] < 335],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_2_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] > 275, 
                                     seg_idxs[:,1] < 50, 
                                     seg_idxs[:,2] > 275, 
                                     np.all([seg_idxs[:,1] < 175, seg_idxs[:,2] < 115],axis=0), 
                                     np.all([seg_idxs[:,1] < 150, seg_idxs[:,0] > 64],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] > 275, 
                                     seg_idxs[:,2] < 275, 
                                     seg_idxs[:,1] < 50, 
                                     seg_idxs[:,2] > 430, 
                                     np.all([seg_idxs[:,0] > 63, seg_idxs[:,1] < 150],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_5_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,0] < 23, 
                                     seg_idxs[:,1] < 215, 
                                     seg_idxs[:,2] > 250, 
                                     np.all([seg_idxs[:,0] > 66, seg_idxs[:,1] < 335],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,0] < 23, 
                                     seg_idxs[:,1] < 205, 
                                     seg_idxs[:,2] < 250, 
                                     np.all([seg_idxs[:,0] > 59, seg_idxs[:,1] < 335],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_6_970_150uW_00003.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,0] < 23, 
                                     seg_idxs[:,1] < 250, 
                                     seg_idxs[:,2] < 135, 
                                     np.all([seg_idxs[:,0] > 49, seg_idxs[:,1] < 350],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,0] < 23, 
                                     seg_idxs[:,1] < 225, 
                                     np.all([seg_idxs[:,0] > 61, seg_idxs[:,1] < 325],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250301_1_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([np.all([seg_idxs[:,0] < 30, seg_idxs[:,1] < 235], axis=0), 
                                     seg_idxs[:,1] < 215, 
                                     seg_idxs[:,2] > 280, 
                                     np.all([seg_idxs[:,0] > 59, seg_idxs[:,1] < 300],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 185, 
                                     seg_idxs[:,2] < 280, 
                                     np.all([seg_idxs[:,0] > 47, seg_idxs[:,1] < 325],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_1_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([np.all([seg_idxs[:,0] < 24, seg_idxs[:,1] < 185], axis=0), 
                                     seg_idxs[:,1] < 155, 
                                     seg_idxs[:,2] > 290, 
                                     np.all([seg_idxs[:,0] > 51, seg_idxs[:,1] < 250],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 125, 
                                     np.all([seg_idxs[:,1] < 200, seg_idxs[:,2] > 450],axis=0), 
                                     seg_idxs[:,2] < 300, 
                                     np.all([seg_idxs[:,0] > 46, seg_idxs[:,1] < 260],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_2_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 200, 
                                     seg_idxs[:,0] < 28, 
                                     seg_idxs[:,2] > 270, 
                                     np.all([seg_idxs[:,0] > 61, seg_idxs[:,1] < 275],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 210, 
                                     seg_idxs[:,0] < 28, 
                                     seg_idxs[:,2] < 270, 
                                     np.all([seg_idxs[:,0] > 53, seg_idxs[:,1] < 300],axis=0)],axis=0)

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_1_00002.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 175, 
                                     seg_idxs[:,0] < 31, 
                                     seg_idxs[:,2] > 275, 
                                     np.all([seg_idxs[:,0] > 68, seg_idxs[:,1] < 305],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 170, 
                                     seg_idxs[:,0] < 31, 
                                     seg_idxs[:,2] < 275, 
                                     np.all([seg_idxs[:,0] > 62, seg_idxs[:,1] < 300],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_2_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 195, 
                                     seg_idxs[:,0] < 33, 
                                     seg_idxs[:,1] > 400, 
                                     seg_idxs[:,2] > 270, 
                                     np.all([seg_idxs[:,0] > 66, seg_idxs[:,1] < 315],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 195, 
                                     seg_idxs[:,0] < 33, 
                                     seg_idxs[:,1] > 400, 
                                     seg_idxs[:,2] < 270, 
                                     np.all([seg_idxs[:,0] > 66, seg_idxs[:,1] < 315],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_1_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 190, 
                                     seg_idxs[:,2] > 285, 
                                     np.all([seg_idxs[:,0] > 57, seg_idxs[:,1] < 285],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 185, 
                                     seg_idxs[:,2] < 285, 
                                     np.all([seg_idxs[:,0] > 61, seg_idxs[:,1] < 280],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_2_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 135, 
                                     np.all([seg_idxs[:,0] > 70, seg_idxs[:,1] < 225],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 175, 
                                     seg_idxs[:,2] > 385, 
                                     np.all([seg_idxs[:,0] > 63, seg_idxs[:,1] < 285],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_3_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 150,
                                     seg_idxs[:,1] > 345, 
                                     np.all([seg_idxs[:,2] < 200, seg_idxs[:,1] < 225],axis=0), 
                                     np.all([seg_idxs[:,0] > 60, seg_idxs[:,1] < 260],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 100, 
                                     np.all([seg_idxs[:,0] > 59, seg_idxs[:,1] < 180],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_1_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 230,
                                     np.all([seg_idxs[:,0] > 66, seg_idxs[:,1] < 325],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 225, 
                                     seg_idxs[:,2] > 415, 
                                     np.all([seg_idxs[:,0] < 23, seg_idxs[:,1] < 260],axis=0), 
                                     np.all([seg_idxs[:,0] > 56, seg_idxs[:,1] < 275, seg_idxs[:,2] < 225],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_2_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 200,
                                     np.all([seg_idxs[:,0] > 46, seg_idxs[:,2] < 155],axis=0), 
                                     np.all([seg_idxs[:,0] > 46, seg_idxs[:,1] < 275, seg_idxs[:,2] < 225],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 225, 
                                     np.all([seg_idxs[:,0] > 45, seg_idxs[:,1] < 300, seg_idxs[:,2] > 400],axis=0), 
                                     np.all([seg_idxs[:,0] > 45, seg_idxs[:,2] > 470],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_1_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 150,
                                     np.all([seg_idxs[:,0] > 46, seg_idxs[:,2] < 95],axis=0), 
                                     np.all([seg_idxs[:,0] > 46, seg_idxs[:,1] < 175, seg_idxs[:,2] < 270],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 175, 
                                     np.all([seg_idxs[:,0] > 47, seg_idxs[:,1] < 285, seg_idxs[:,2] > 330],axis=0), 
                                     np.all([seg_idxs[:,0] > 47, seg_idxs[:,2] > 415],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_2_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 200,
                                     np.all([seg_idxs[:,0] > 54, seg_idxs[:,2] < 110],axis=0), 
                                     np.all([seg_idxs[:,0] > 54, seg_idxs[:,1] < 290, seg_idxs[:,2] < 190],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 225, 
                                     np.all([seg_idxs[:,0] > 61, seg_idxs[:,1] < 330, seg_idxs[:,2] > 350],axis=0), 
                                     np.all([seg_idxs[:,0] > 61, seg_idxs[:,2] > 425],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_3_970_150uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 180,
                                     np.all([seg_idxs[:,0] > 47, seg_idxs[:,2] < 100],axis=0), 
                                     np.all([seg_idxs[:,0] > 47, seg_idxs[:,1] < 295, seg_idxs[:,2] < 175],axis=0)],axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 175, 
                                     np.all([seg_idxs[:,0] > 42, seg_idxs[:,1] < 290, seg_idxs[:,2] > 350],axis=0), 
                                     np.all([seg_idxs[:,0] > 42, seg_idxs[:,2] > 415],axis=0)],axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_2_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 200,
                                     seg_idxs[:,2] > 260, 
                                     seg_idxs[:,2] < 85, 
                                     np.all([seg_idxs[:,0] > 58, seg_idxs[:,1] < 325, seg_idxs[:,2] < 175],axis=0)], axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 175, 
                                     seg_idxs[:,2] < 260, 
                                     seg_idxs[:,2] > 415, 
                                     np.all([seg_idxs[:,0] > 55, seg_idxs[:,1] < 300, seg_idxs[:,2] > 325],axis=0)], axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_3_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([np.all([seg_idxs[:,0] < 57, seg_idxs[:,1] > 300, seg_idxs[:,2] < 145],axis=0), 
                                     seg_idxs[:,1] < 175,
                                     seg_idxs[:,2] < 105, 
                                     np.all([seg_idxs[:,0] > 67, seg_idxs[:,1] < 275, seg_idxs[:,2] < 215],axis=0)], axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 155, 
                                     seg_idxs[:,2] > 410, 
                                     np.all([seg_idxs[:,0] > 57, seg_idxs[:,1] < 275, seg_idxs[:,2] > 325],axis=0)], axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250225_1_970_250uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,2] < 100, 
                                     seg_idxs[:,0] < 12, 
                                     np.all([seg_idxs[:,1] < 350, seg_idxs[:,2] < 175],axis=0)], axis=0)
        else:
            bool_seg_idxs = seg_idxs[:,2] < 420
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250225_2_970_250uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,2] < 100, 
                                     seg_idxs[:,1] < 330, 
                                     np.all([seg_idxs[:,1] < 360, seg_idxs[:,2] < 150],axis=0)], axis=0)
        else:
            bool_seg_idxs = seg_idxs[:,2] < 400
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250225_3_970_250uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,2] < 125, 
                                     seg_idxs[:,1] < 275, 
                                     np.all([seg_idxs[:,1] < 300, seg_idxs[:,2] < 200],axis=0)], axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 275, 
                                     seg_idxs[:,2] > 450, 
                                     np.all([seg_idxs[:,1] < 300, seg_idxs[:,2] > 350],axis=0)], axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250307_2_970_250uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,2] < 125, 
                                     seg_idxs[:,1] < 190, 
                                     np.all([seg_idxs[:,1] < 225, seg_idxs[:,2] < 200],axis=0)], axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 190, 
                                     np.all([seg_idxs[:,1] < 215, seg_idxs[:,2] > 375],axis=0)], axis=0)
    if image_file == home_dir + "/saved_data/MB_imaging/MB131B_Gamma_m/20250307_5_970_250uW_00001.tif":
        if side == 'L':
            bool_seg_idxs = ~np.any([seg_idxs[:,1] > 325, 
                                     seg_idxs[:,2] > 265, 
                                     seg_idxs[:,1] < 250], axis=0)
        else:
            bool_seg_idxs = ~np.any([seg_idxs[:,1] < 235, 
                                     seg_idxs[:,2] < 265, 
                                     seg_idxs[:,1] > 320], axis=0)
    return bool_seg_idxs


def get_is_bp2(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ idxs[:,2] > 190 ] = 1
        else: comps[ idxs[:,2] < 355 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ idxs[:,2] > 190 ] = 1
        else: comps[ idxs[:,2] < 350 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_1_970nm_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 180 ] = 1
        else: comps[ idxs[:,2] < 335 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_2_970nm_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 200 ] = 1
        else: comps[ idxs[:,2] < 355 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_4_970nm_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 195 ] = 1
        else: comps[ idxs[:,2] < 350 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 200 ] = 1
        else: comps[ idxs[:,2] < 340 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_2_970_150uW_00001.tif":
        if side == 'L':  
            cond_1 = np.all([idxs[:,0] <= 39, idxs[:,2] > 190],axis=0)
            cond_2 = np.all([idxs[:,0] >= 46, idxs[:,2] > 240], axis=0)
            
            comps[ np.any([cond_1, cond_2],axis=0) ] = 1
        else:  
            cond_1 = np.all([idxs[:,0] <= 39, idxs[:,2] < 335],axis=0)
            cond_2 = np.all([idxs[:,0] >= 46, idxs[:,2] < 390], axis=0)
            
            comps[ np.any([cond_1, cond_2],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 180 ] = 1
        else: comps[ idxs[:,2] < 320 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_2_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 200 ] = 1
        else: comps[ idxs[:,2] < 345 ] = 1
            
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_3_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 190 ] = 1
        else: comps[ idxs[:,2] < 340 ] = 1

    return comps

def get_is_bp1(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 150, idxs[:,2] < 190],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 355, idxs[:,2] < 390, idxs[:,1] > 175],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 148, idxs[:,2] < 190],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 350, idxs[:,2] < 390, idxs[:,1] > 210],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_1_970nm_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 140, idxs[:,2] < 180],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 335, idxs[:,2] < 380, idxs[:,1] > 250],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_2_970nm_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 150, idxs[:,2] < 200],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 355, idxs[:,2] < 395, idxs[:,1] > 185],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_4_970nm_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 155, idxs[:,2] < 195],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 350, idxs[:,2] < 385, idxs[:,1] > 250],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 200, idxs[:,1] < 330, idxs[:,2] > 150],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 340, idxs[:,1] > 330, idxs[:,2] < 380],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_2_970_150uW_00001.tif":
        if side == 'L':  
            cond_1 = np.all([idxs[:,0] <= 39, idxs[:,2] < 190, idxs[:,1] < 340, idxs[:,2] > 145],axis=0)
            cond_2 = np.all([idxs[:,0] >= 46, idxs[:,2] < 240, idxs[:,1] < 270, idxs[:,2] > 200], axis=0)
            
            comps[ np.any([cond_1, cond_2],axis=0) ] = 1
        else:  
            cond_1 = np.all([idxs[:,0] <= 39, idxs[:,2] > 335, idxs[:,1] > 340, idxs[:,2] < 382],axis=0)
            cond_2 = np.all([idxs[:,0] >= 46, idxs[:,2] > 390, idxs[:,1] > 290, idxs[:,2] < 470], axis=0)
            
            comps[ np.any([cond_1, cond_2],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 180, idxs[:,1] > 260, idxs[:,2] > 130],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 320, idxs[:,1] > 280, idxs[:,2] < 365],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 200, idxs[:,1] > 300, idxs[:,2] > 155],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 345, idxs[:,1] > 330, idxs[:,2] < 390],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_3_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 190, idxs[:,1] > 290, idxs[:,2] > 140],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 340, idxs[:,1] > 290, idxs[:,2] < 375],axis=0) ] = 1

    return comps

def get_is_ap1(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 150, idxs[:,2] > 110, idxs[:,1] > 175],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 425, idxs[:,2] > 390, idxs[:,1] > 170],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 148, idxs[:,2] > 115, idxs[:,1] > 200],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 420, idxs[:,2] > 390, idxs[:,1] > 205],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_1_970nm_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 140, idxs[:,2] > 105, idxs[:,1] > 260],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 415, idxs[:,2] > 380, idxs[:,1] > 275],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_2_970nm_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 150, idxs[:,2] > 115, idxs[:,1] > 210],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 430, idxs[:,2] > 395, idxs[:,1] > 185],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_4_970nm_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 155, idxs[:,2] > 105, idxs[:,1] > 260],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 425, idxs[:,2] > 385, idxs[:,1] > 255],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 150, idxs[:,1] > 350],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 380, idxs[:,1] > 340],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,0] <= 39, idxs[:,2] < 145, idxs[:,1] > 340],axis=0) ] = 1
        else: comps[ np.all([idxs[:,0] <= 39, idxs[:,2] > 382, idxs[:,1] > 350],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 135, idxs[:,1] > 275],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 365, idxs[:,1] > 280],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 155, idxs[:,2] > 130, idxs[:,1] > 300],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 390, idxs[:,2] < 420, idxs[:,1] > 330],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_3_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 140, idxs[:,2] > 110, idxs[:,1] > 300],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 375, idxs[:,2] < 410, idxs[:,1] > 290],axis=0) ] = 1

    return comps

def get_is_ap2(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 170, idxs[:,1] > 110, idxs[:,1] < 175],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 390, idxs[:,1] < 170, idxs[:,1] > 100],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 170, idxs[:,1] > 148, idxs[:,1] < 200],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 375, idxs[:,1] < 205, idxs[:,1] > 150],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_1_970nm_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 170, idxs[:,1] > 205, idxs[:,1] < 260],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 370, idxs[:,1] < 275, idxs[:,1] > 210],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_2_970nm_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 175, idxs[:,1] > 160, idxs[:,1] < 210],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 370, idxs[:,1] > 130, idxs[:,1] < 185],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_4_970nm_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 175, idxs[:,1] > 200, idxs[:,1] < 260],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 370, idxs[:,1] > 220, idxs[:,1] < 255],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 350, idxs[:,2] < 150, idxs[:,1] > 280],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 340, idxs[:,2] > 350, idxs[:,1] > 280],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,0] <= 39, idxs[:,1] < 340, idxs[:,2] < 160, idxs[:,1] > 270],axis=0) ] = 1
        else: comps[ np.all([idxs[:,0] <= 39, idxs[:,1] < 350, idxs[:,2] > 370, idxs[:,1] > 290],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 275, idxs[:,2] < 140, idxs[:,1] > 215],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 280, idxs[:,2] > 360, idxs[:,1] > 235],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 300, idxs[:,2] > 130, idxs[:,2] < 170, idxs[:,1] > 260],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 330, idxs[:,2] > 380, idxs[:,1] > 290],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_3_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 300, idxs[:,2] < 160, idxs[:,2] > 100, idxs[:,1] > 245],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 290, idxs[:,2] > 360, idxs[:,2] < 405, idxs[:,1] > 235],axis=0) ] = 1

    return comps


def get_is_ap3(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_2_970.tif":
        if side == 'L': comps[ idxs[:,1] < 110 ] = 1
        else: comps[ idxs[:,1] < 100 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/negative offset subtraction/0128_3_970.tif":
        if side == 'L': comps[ idxs[:,1] < 148 ] = 1
        else: comps[ idxs[:,1] < 150 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_1_970nm_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 205 ] = 1
        else: comps[ idxs[:,1] < 210 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_2_970nm_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 160 ] = 1
        else: comps[ idxs[:,1] < 130 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_ap_MB463B/ab_ap_mito_cyto/20250210/green_red/20250210_4_970nm_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 200 ] = 1
        else: comps[ idxs[:,1] < 220 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 280 ] = 1
        else: comps[ idxs[:,1] < 280 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250228_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,0] <= 39, idxs[:,1] < 270],axis=0) ] = 1
        else: comps[ np.all([idxs[:,0] <= 39, idxs[:,1] < 290],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 215 ] = 1
        else: comps[ idxs[:,1] < 235 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_2_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 260 ] = 1
        else: comps[ idxs[:,1] < 290 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/a'b'_VT030604/20250301_3_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 245 ] = 1
        else: comps[ idxs[:,1] < 235 ] = 1

    return comps













def get_is_b2(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 185 ] = 1
        else: comps[ idxs[:,2] < 300 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_2_970_150uW_00002.tif":
        if side == 'L': comps[ idxs[:,2] > 185 ] = 1
        else: comps[ idxs[:,2] < 325 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,0] > 30, idxs[:,2] > 195],axis=0) ] = 1
        else: comps[ np.all([idxs[:,0] > 21, idxs[:,2] < 330],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_5_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 195 ] = 1
        else: comps[ idxs[:,2] < 325 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_6_970_150uW_00003.tif":
        if side == 'L': comps[ idxs[:,2] > 225 ] = 1
        else: comps[ np.all([idxs[:,2] < 360, idxs[:,1] > 345],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 245 ] = 1
        else: comps[ idxs[:,2] < 325 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 250 ] = 1
        else: comps[ idxs[:,2] < 350 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_2_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 225 ] = 1
        else: comps[ idxs[:,2] < 320 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_1_00002.tif":
        if side == 'L': comps[ idxs[:,2] > 220 ] = 1
        else: comps[ idxs[:,2] < 325 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 240 ] = 1
        else: comps[ idxs[:,2] < 330 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_2_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 180 ] = 1
        else: comps[ idxs[:,2] < 270 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_3_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 320 ] = 1
        else: comps[ idxs[:,2] < 405 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_2_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 210 ] = 1
        else: comps[ idxs[:,2] < 310 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_3_00001.tif":
        if side == 'L': comps[ idxs[:,2] > 205 ] = 1
        else: comps[ idxs[:,2] < 310 ] = 1

    return comps


def get_is_b1(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 130, idxs[:,2] < 185, idxs[:,1] > 325],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 300, idxs[:,2] < 375, idxs[:,1] > 325],axis=0) ] = 1
    
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_2_970_150uW_00002.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 125, idxs[:,2] < 185, idxs[:,1] > 325],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 325, idxs[:,2] < 375, idxs[:,1] > 325],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,0] > 30, idxs[:,2] > 140, idxs[:,2] < 195, idxs[:,1] > 185],axis=0) ] = 1
        else: comps[ np.all([idxs[:,0] > 21, idxs[:,2] > 330, idxs[:,2] < 380, idxs[:,1] > 195],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_5_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 125, idxs[:,2] < 195, idxs[:,1] > 350],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 325, idxs[:,2] < 375, idxs[:,1] > 350],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_6_970_150uW_00003.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 225, idxs[:,1] > 355, idxs[:,2] > 170],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 360, idxs[:,1] > 340, idxs[:,2] < 430],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] > 150, idxs[:,2] < 245, idxs[:,1] > 350],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 325, idxs[:,2] < 395, idxs[:,1] > 340],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 245, idxs[:,1] > 285, idxs[:,2] > 155],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 350, idxs[:,1] > 275, idxs[:,2] < 425],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 225, idxs[:,1] > 300, idxs[:,2] > 135],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 320, idxs[:,1] > 335, idxs[:,2] < 385],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_1_00002.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 220, idxs[:,1] > 300, idxs[:,2] > 135],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 325, idxs[:,1] > 300, idxs[:,2] < 395],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 240, idxs[:,1] > 320, idxs[:,2] > 155],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 330, idxs[:,1] > 310, idxs[:,2] < 410],axis=0) ] = 1 

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 180, idxs[:,1] > 265, idxs[:,2] > 80],axis=0) ] = 1
        else: 
            cond_1 = np.all([idxs[:,1] > 325, idxs[:,2] > 340],axis=0)
            cond_2 = np.all([idxs[:,2] > 270, idxs[:,1] > 300, idxs[:,2] < 350],axis=0)
            comps[ np.any([cond_1, cond_2],axis=0)  ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_3_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 320, idxs[:,1] > 290, idxs[:,2] > 225],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 405, idxs[:,1] > 250, idxs[:,2] < 480],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_2_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 210, idxs[:,1] > 320, idxs[:,2] > 130],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 310, idxs[:,1] > 320, idxs[:,2] < 370],axis=0) ] = 1 
    
    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_3_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 205, idxs[:,1] > 300, idxs[:,2] > 145],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 310, idxs[:,1] > 300, idxs[:,2] < 375],axis=0) ] = 1 

    return comps

def get_is_a1(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 105, idxs[:,1] > 325],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 400, idxs[:,2] > 375, idxs[:,1] > 320],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_2_970_150uW_00002.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 125, idxs[:,1] > 325],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 410, idxs[:,2] > 385, idxs[:,1] > 325],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,0] > 30, idxs[:,2] < 140, idxs[:,1] > 200],axis=0) ] = 1
        else: comps[ np.all([idxs[:,0] > 21, idxs[:,2] < 423, idxs[:,2] > 380, idxs[:,1] > 190],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_5_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 125, idxs[:,1] > 360],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 415, idxs[:,2] > 375, idxs[:,1] > 335],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_6_970_150uW_00003.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 170, idxs[:,1] < 410, idxs[:,1] > 360],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 430, idxs[:,1] > 330],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 150, idxs[:,1] > 355],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] < 435, idxs[:,2] > 395, idxs[:,1] > 340],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 155, idxs[:,1] < 290, idxs[:,1] > 250],axis=0) ] = 1
        else: comps[ idxs[:,2] > 427 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 135, idxs[:,1] < 325, idxs[:,1] > 300],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 385, idxs[:,1] > 310],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_1_00002.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 135, idxs[:,1] > 300],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 395, idxs[:,1] > 300],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 155, idxs[:,1] > 290],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 410, idxs[:,1] < 335, idxs[:,1] > 270],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 120, idxs[:,1] < 250, idxs[:,1] > 225],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 350, idxs[:,1] < 325, idxs[:,1] > 285],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_3_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 240, idxs[:,1] < 300, idxs[:,1] > 275],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 480, idxs[:,1] < 230, idxs[:,1] > 195],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_2_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 130, idxs[:,2] > 100, idxs[:,1] > 325],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 380, idxs[:,1] > 300],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_2_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 130, idxs[:,2] > 100, idxs[:,1] > 325],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 380, idxs[:,1] > 300],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_3_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 140, idxs[:,2] > 110, idxs[:,1] < 300, idxs[:,1] > 275],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 320, idxs[:,1] < 330, idxs[:,1] > 275],axis=0) ] = 1

    return comps

def get_is_a2(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 175, idxs[:,1] > 250, idxs[:,1] < 325],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 350, idxs[:,1] < 320, idxs[:,1] > 275],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_2_970_150uW_00002.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 150, idxs[:,1] > 260, idxs[:,1] < 325],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 350, idxs[:,2] < 405, idxs[:,1] < 325, idxs[:,1] > 275],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,0] > 30, idxs[:,2] < 150, idxs[:,1] > 150, idxs[:,1] < 200],axis=0) ] = 1
        else: comps[ np.all([idxs[:,0] > 21, idxs[:,2] > 375, idxs[:,2] < 415, idxs[:,1] < 190, idxs[:,1] > 125],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_5_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 150, idxs[:,1] > 280, idxs[:,1] < 360],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 350, idxs[:,2] < 410, idxs[:,1] < 335, idxs[:,1] > 275],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_6_970_150uW_00003.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 360, idxs[:,2] < 190, idxs[:,1] > 305],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 330, idxs[:,2] > 400, idxs[:,1] > 280],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,2] < 175, idxs[:,1] > 275, idxs[:,1] < 355],axis=0) ] = 1
        else: comps[ np.all([idxs[:,2] > 350, idxs[:,2] < 425, idxs[:,1] < 340, idxs[:,1] > 265],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 250, idxs[:,2] < 200, idxs[:,1] > 200],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 265, idxs[:,1] > 200],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 300, idxs[:,2] < 175, idxs[:,1] > 255],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 310, idxs[:,1] > 260],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_1_00002.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 300, idxs[:,2] < 150, idxs[:,1] > 255],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 300, idxs[:,1] > 250],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 270, idxs[:,2] < 170, idxs[:,0] < 60, idxs[:,1] > 225],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 270, idxs[:,1] > 240],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 225, idxs[:,2] < 130, idxs[:,1] > 200],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 285, idxs[:,1] > 250],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_3_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 275, idxs[:,2] < 250, idxs[:,1] > 225, idxs[:,0] > 22],axis=0) ] = 1
        else: 
            comp_1 = np.any([np.all([idxs[:,0] >= 22, idxs[:,1] > 175, idxs[:,0] < 29],axis=0), 
                             np.all([idxs[:,0] >= 29, idxs[:,1] > 150, idxs[:,0] < 37, idxs[:,1] < 225],axis=0), 
                             np.all([idxs[:,0] >= 37, idxs[:,1] > 130, idxs[:,1] < 200],axis=0)],axis=0)
            comps[ comp_1 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_2_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 325, idxs[:,2] > 105, idxs[:,2] < 150, idxs[:,1] > 260],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 300, idxs[:,2] > 350, idxs[:,1] > 250],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_3_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 275, idxs[:,2] < 160, idxs[:,1] > 230],axis=0) ] = 1
        else: comps[ np.all([idxs[:,1] < 275, idxs[:,2] > 350, idxs[:,1] > 220],axis=0) ] = 1

    return comps


def get_is_a3(image_file, side, idxs):
    comps = np.zeros(len(idxs))
    assert idxs.shape[1] == 3
    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 250 ] = 1
        else: comps[ idxs[:,1] < 275 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250226_2_970_150uW_00002.tif":
        if side == 'L': comps[ idxs[:,1] < 260 ] = 1
        else: comps[ idxs[:,1] < 275 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_2_970_150uW_00001.tif":
        if side == 'L': comps[ np.any([idxs[:,0] < 30, idxs[:,1] < 150],axis=0) ] = 1
        else: comps[ np.any([idxs[:,0] < 21, idxs[:,1] < 125],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_5_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 280 ] = 1
        else: comps[ idxs[:,1] < 275 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB477B_ab_s/20250228_6_970_150uW_00003.tif":
        if side == 'L': comps[ idxs[:,1] < 305 ] = 1
        else: comps[ idxs[:,1] < 280 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250301_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 275 ] = 1
        else: comps[ idxs[:,1] < 265 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_1_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 200 ] = 1
        else: comps[ idxs[:,1] < 200 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250302_2_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 255 ] = 1
        else: comps[ idxs[:,1] < 260 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250320_1_00002.tif":
        if side == 'L': comps[ idxs[:,1] < 255 ] = 1
        else: comps[ idxs[:,1] < 250 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_1_970_150uW_00001.tif":
        if side == 'L': comps[ np.all([idxs[:,1] < 225, idxs[:,0] < 60],axis=0) ] = 1
        else: comps[ idxs[:,1] < 240 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_2_970_150uW_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 200 ] = 1
        else: comps[ np.any([idxs[:,1] < 250, idxs[:,0] < 25],axis=0) ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/MB594B_ab_c/20250303_3_970_150uW_00001.tif":
        if side == 'L': comps[ np.any([idxs[:,0] < 22, idxs[:,1] < 225],axis=0) ] = 1
        else: 
            comp_1 = np.any([idxs[:,0] < 22, 
                             np.all([idxs[:,0] >= 22, idxs[:,1] < 175, idxs[:,0] < 29],axis=0), 
                             np.all([idxs[:,0] >= 29, idxs[:,1] < 150, idxs[:,0] < 37],axis=0), 
                             np.all([idxs[:,0] >= 37, idxs[:,1] < 130],axis=0)],axis=0)
                             
            comps[ comp_1 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_2_00001.tif":
        if side == 'L': comps[ idxs[:,1] < 260 ] = 1
        else: comps[ idxs[:,1] < 250 ] = 1

    if image_file == home_dir + "/saved_data/MB_imaging/5D_old_fly/185B_20250324_3_00001.tif":
        if side == 'L': comps[ np.any([idxs[:,0] < 20, idxs[:,1] < 230],axis=0) ] = 1
        else: comps[ idxs[:,1] < 220 ] = 1

    return comps






def get_is_g12345(side, seg_idxs):
    threshes = np.quantile(seg_idxs[:,2], np.linspace(0,1,6))
    is_g12345 = np.array([ np.all([seg_idxs[:,2] > threshes[ii], seg_idxs[:,2] < threshes[ii+1]],axis=0) for ii in range(len(threshes)-1) ])
    if side == 'R':
        is_g12345 = np.flip(is_g12345, axis=0)
    return is_g12345
    












