from itertools import product
import os
import glob


if __name__ == "__main__":
    # dataroot = "data/p30_220720-2"
    dataroot = "data/mate40pro_220719-3"
    depth_gts = [0.5, 1, 3, 5]
    moving_states = ["static", "slow", "fast"]
    prefix = "processed_"

    for depth_gt, moving_state in product(depth_gts, moving_states):
        src_folder = os.path.join(dataroot, str(depth_gt), prefix + moving_state)
        dst_filepath = os.path.join(dataroot, f"{depth_gt}_{moving_state}.txt")

        left_files = sorted(glob.glob(os.path.join(src_folder, "left_*.png")))
        right_files = sorted(glob.glob(os.path.join(src_folder, "right_*.png")))

        with open(dst_filepath, "w") as f:
            for left_file, right_file in zip(left_files, right_files):
                left_file = os.sep.join(left_file.split(os.sep)[1:])
                right_file = os.sep.join(right_file.split(os.sep)[1:])
                f.write(f"{left_file} {right_file}\n")
