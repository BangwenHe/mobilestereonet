import glob
from itertools import product
import os

import cv2
import numpy as np
import numpy.ma as ma

from utils import read_pfm, convert_disparity_to_depth, load_stereo_coefficients, draw_depth_map


def load_roi_file(roi_file_path):
    result = {}
    with open(roi_file_path, "r") as f:
        result = {i.split()[0]: [int(j) if j.isdigit() else float(j) for j in i.split()[1:]] for i in f.readlines()}
    
    return result


def calculate_precision(result_folder, calibration_file, roi_file, min_depth=0, max_depth=8):
    disparity_files = glob.glob(f"{result_folder}/*.pfm")
    Q = load_stereo_coefficients(calibration_file)[-1]

    filename_to_bbox = load_roi_file(roi_file)
    result = []

    for disparity_file in disparity_files:
        disp, _ = read_pfm(disparity_file)
        depth = convert_disparity_to_depth(disp, Q)
        depth = np.clip(depth, min_depth, max_depth)
        depth_map = draw_depth_map(depth)
        
        filename = os.path.split(disparity_file)[1].split(".")[0]
        cv2.imwrite(f"{result_folder}/{filename}_depth.png", depth_map)

        if "left" not in filename: continue

        filename_ext = "%s.png" % filename
        filename_ext = filename_ext[filename_ext.find("left"):]
        x, y, w, h, gt = filename_to_bbox[filename_ext]
        roi_depth = depth[y:y+h, x:x+w]
        roi_depth_mask = ma.masked_outside(roi_depth, min_depth, max_depth)

        error = np.abs(roi_depth_mask - gt) / gt
        result.append([
            filename_ext, gt,
            np.min(roi_depth_mask), np.max(roi_depth_mask),np.mean(roi_depth_mask), 
            np.min(error), np.max(error), np.mean(error)
        ])

    
    return result


def summarize_precision(precision_results, depth_gts):
    gt_to_error_means = {depth_gt: [] for depth_gt in depth_gts}
    results = []

    for precision_result in precision_results:
        gt = precision_result[1]
        if gt not in depth_gts: continue

        gt_to_error_means[gt].append(precision_result[-1])
    
    for gt, error_means in gt_to_error_means.items():
        if len(error_means) is 0: continue

        results.append([
            None, str(gt), 0, 0, 0, 0, 0, np.mean(error_means)
        ])
    
    return results


def save_precision_results(precision_results, result_filepath):
    precision_results = [[str(i) for i in precision_result] for precision_result in precision_results]
    
    with open(result_filepath, "w") as f:
        f.write("filename,gt,min_depth,max_depth,avg_depth,min_error,max_error,avg_error\n")
        for line in precision_results:
            f.write(",".join(line))
            f.write("\n")


if __name__ == "__main__":
    depth_gts = [0.5, 1, 3, 5]
    moving_states = ['static', 'slow', 'fast']
    # calibration_file = "data/p30_1.03855444_1.05697224.yml"
    calibration_file = "data/mate40pro_1.01841479_1.01802223.yml"

    exp_name = "mate40pro_220719-3"
    save_folder = f"predictions/{exp_name}"

    results = {moving_state: [] for moving_state in moving_states}

    for depth_gt, moving_state in product(depth_gts, moving_states):
        result_folder = f"predictions/{exp_name}/{depth_gt}/processed_{moving_state}"
        roi_file = f"data/{exp_name}/{depth_gt}/processed_{moving_state}_{depth_gt}.roi"

        res = calculate_precision(result_folder, calibration_file, roi_file)
        results[moving_state].extend(res)
    
    for moving_state, moving_results in results.items():
        precision_summary = summarize_precision(moving_results, depth_gts)
        moving_results.extend(precision_summary)

        save_precision_results(moving_results, os.path.join(save_folder, f"{moving_state}.csv"))
