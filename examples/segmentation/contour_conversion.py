import json
import argparse
import numpy as np
import nibabel as nib
import bia

def main(img_path, contour_path):

    # load image for image domain shape
    nifti_img = nib.load(img_path)
    img_domain = np.array(nifti_img.dataobj).shape
    affine = np.array(nifti_img.affine)

    # load contour voxel coordinates
    with open(contour_path) as f:
        data = json.load(f)
        print(data.keys())
    contour_points = data["voxel_coordinates"]

    # convert contour to binary mask
    binary_mask = bia.naively_convert_high_res_contour_to_mask_3d(img_domain=img_domain, contour_points=contour_points)

    # save binary mask as NIFTY image
    nifti_binary_mask = nib.Nifti1Image(binary_mask, affine)
    nib.save(nifti_binary_mask, "binary_mask.nii.gz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Contour Conversion',
        description='Convert contour to binary mask')

    parser.add_argument('-i', '--img_path')
    parser.add_argument('-c', '--contour_path')
    args = parser.parse_args()

    main(img_path=args.img_path, contour_path=args.contour_path)