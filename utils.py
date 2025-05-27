import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import io
from scipy.ndimage import label
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter1d


def load_dicom(path_folder: str):
    slices = []

    sorted_files = sorted(os.listdir(path_folder))
    for file in sorted_files:
        if file.endswith('.dcm'):
            file_path = os.path.join(path_folder, file)
            slices.append(pydicom.dcmread(file_path))

    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices


def create_3d_array_from_dicom(slices):
    img_shape = slices[0].pixel_array.shape
    img_3d = np.zeros((len(slices), *img_shape), dtype=np.int16)

    for i, slice in enumerate(slices):
        img_3d[i] = slice.pixel_array

    return img_3d


def convert_to_hounsfield(slices):
    """
    Converts a list of DICOM slices into a 3D volume in Hounsfield Units (HU).

    Parameters:
    - slices: list of pydicom.Dataset objects, properly sorted

    Returns:
    - image_hu: 3D volume (np.ndarray) with intensities in HU
    - spacing: tuple (spacing between slices Z, Y, X) in mm
    """
    # Apilar el volumen
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Opcional: fijar fuera de campo a 0 si es necesario
    image[image == -2000] = 0

    # Rescale (HU conversion)
    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    # Espaciado físico entre vóxeles (en mm)
    slice_thickness = np.abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
    pixel_spacing = slices[0].PixelSpacing  # [dy, dx]
    spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))  # (Z, Y, X)

    return image, spacing

def generate_mip_gif(ct_volume, liver_mask, tumor_mask, output_path="mip_rotation.gif",
                     alpha_liver=0.3, alpha_tumor=0.4,
                     cmap_liver='Blues', cmap_tumor='Reds',
                     cmap_ct='gray', aspect=4, duration=100, steps=36):
    frames = []
    angles = np.linspace(0, 360, steps, endpoint=False)

    for angle in angles:
        fig, ax = plt.subplots(figsize=(5, 5))

        # Rotate volume around the z-axis
        rotated_ct = ndimage.rotate(ct_volume, angle, axes=(1, 2), reshape=False, order=1)
        rotated_liver = ndimage.rotate(liver_mask, angle, axes=(1, 2), reshape=False, order=0)
        rotated_tumor = ndimage.rotate(tumor_mask, angle, axes=(1, 2), reshape=False, order=0)

        # Apply sagittal MIP (axis=2 for X)
        mip_ct = np.max(rotated_ct, axis=1)
        mip_liver = np.max(rotated_liver, axis=1)
        mip_tumor = np.max(rotated_tumor, axis=1)

        # Optional: apply contrast enhancement (e.g., CLAHE or windowing)
        mip_ct_display = mip_ct  # assuming already processed

        # Alpha masks only where mask == 1
        alpha_l = np.where(mip_liver == 1, alpha_liver, 0)
        alpha_t = np.where(mip_tumor == 1, alpha_tumor, 0)

        # Plot base image
        ax.imshow(mip_ct_display, cmap=cmap_ct, aspect=aspect)
        ax.imshow(mip_liver, cmap=cmap_liver, alpha=alpha_l, aspect=aspect)
        ax.imshow(mip_tumor, cmap=cmap_tumor, alpha=alpha_t, aspect=aspect)
        ax.axis('off')

        # Save frame to memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        frame = Image.open(buf)
        frames.append(frame.copy())
        plt.close(fig)

    # Save frames to GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"✅ GIF saved at: {output_path}")


def show_mip_with_mask(mip_img, mask, c_img="gray", c_mask='Reds', figsize=(10, 6), aspect=1, alpha_value=0.4):
    alpha_mask = np.where(mask == 1, alpha_value, 0)  # Alpha only where mask is 1
    plt.figure(figsize=(10, 6))
    plt.imshow(mip_img, cmap=c_img)  # Base image
    plt.imshow(mask, cmap=c_mask, alpha=alpha_mask,aspect=aspect)  # Mask overlay with conditional alpha
    plt.show()

def show_mip_with_two_masks(mip_img, mask_liver, mask_tumor,
                            alpha_liver=0.3, alpha_tumor=0.4,
                            cmap_mip='gray', cmap_liver='Blues', cmap_tumor='Reds',
                            bbox=None, centroid=None, plane='coronal', aspect=1):

    alpha_l = np.where(mask_liver == 1, alpha_liver, 0) if mask_liver is not None else None
    alpha_t = np.where(mask_tumor == 1, alpha_tumor, 0) if mask_tumor is not None else None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(mip_img, cmap=cmap_mip, aspect=aspect)
    
    if mask_liver is not None:
        ax.imshow(mask_liver, cmap=cmap_liver, alpha=alpha_l, aspect=aspect)
    if mask_tumor is not None:
        ax.imshow(mask_tumor, cmap=cmap_tumor, alpha=alpha_t, aspect=aspect)

    # Dibujar múltiples bounding boxes
    if bbox is not None and isinstance(bbox, list):
        for b in bbox:
            if plane == 'coronal':
                y_min, y_max = b[1]
                x_min, x_max = b[2]
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     edgecolor='lime', facecolor='none', linewidth=1)
                ax.add_patch(rect)
            elif plane == 'sagittal':
                z_min, z_max = b[0]
                y_min, y_max = b[1]
                rect = plt.Rectangle((y_min, z_min), y_max - y_min, z_max - z_min,
                                     edgecolor='lime', facecolor='none', linewidth=1)
                ax.add_patch(rect)
            elif plane == 'axial':
                z_min, z_max = b[0]
                x_min, x_max = b[2]
                rect = plt.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min,
                                     edgecolor='lime', facecolor='none', linewidth=1)
                ax.add_patch(rect)
    elif bbox is not None:
        raise TypeError("bbox debe ser una lista de bounding boxes")

    # Dibujar múltiples centroides
    if centroid is not None and isinstance(centroid, list):
        for c in centroid:
            if plane == 'coronal':
                y_c, x_c = c[1], c[2]
                ax.plot(x_c, y_c, 'ro', markersize=3)
            elif plane == 'sagittal':
                z_c, y_c = c[0], c[1]
                ax.plot(y_c, z_c, 'ro', markersize=3)
            elif plane == 'axial':
                z_c, x_c = c[0], c[2]
                ax.plot(x_c, z_c, 'ro', markersize=3)
    elif centroid is not None:
        raise TypeError("centroid must be a list of centroids")

    # ax.axis('off')
    # ax.set_title(f"{plane.capitalize()} MIP with Masks, BBoxes and Centroids")
    # plt.legend(['Centroid'])
    plt.show()

def window_image(img, center, width):
    vmin = center - width / 2
    vmax = center + width / 2
    img_clipped = np.clip(img, vmin, vmax)
    img_normalized = (img_clipped - vmin) / (vmax - vmin)
    return img_normalized


def gaussian_derivative_3d(volume, sigma=1.0):
    gx = gaussian_filter1d(volume, sigma=sigma, axis=2, order=1)
    gy = gaussian_filter1d(volume, sigma=sigma, axis=1, order=1)
    gz = gaussian_filter1d(volume, sigma=sigma, axis=0, order=1)

    gradient_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
    return gradient_magnitude


def sobel_gradient_magnitude(volume):
    volume = np.nan_to_num(volume, nan=0.0)
    
    gx = sobel(volume, axis=2)
    gy = sobel(volume, axis=1)
    gz = sobel(volume, axis=0)

    # Cuadrado de gradientes
    grad_sq = gx**2 + gy**2 + gz**2
 
    # Reemplaza valores negativos si aparecen por numéricos válidos
    grad_sq[grad_sq < 0] = 0

    magnitude = np.sqrt(grad_sq)
    return magnitude


def laplacian_edges(volume):
    return laplace(volume)

def tumor_contrast_enhancement(volume, liver_mask, sigma=2):
    blurred = gaussian_filter(volume, sigma=sigma)
    enhanced = volume - blurred  # zonas diferentes al hígado se amplifican
    enhanced[liver_mask == 0] = 0  # fuera del hígado = 0
    return enhanced

def liver_zscore(volume, liver_mask):
    liver_vals = volume[liver_mask == 1]
    mean = np.mean(liver_vals)
    std = np.std(liver_vals)
    zmap = (volume - mean) / std
    return zmap

def get_component_bbox(region_mask):
    z, y, x = np.where(region_mask)
    return ((z.min(), z.max()), (y.min(), y.max()), (x.min(), x.max()))


def get_component_bbox(region_mask):
    z, y, x = np.where(region_mask)
    return ((z.min(), z.max()), (y.min(), y.max()), (x.min(), x.max()))

def analyze_tumor_vs_liver_per_component(volume, tumor_mask, liver_mask, min_voxels=10, k=1.0):
    """
    For each tumor component:
    - calculates bounding box
    - extracts tumor and liver intensities within the bbox
    - compares histograms
    - calculates suggested threshold (mean ± k·std)
    """
    labeled, n = label(tumor_mask)
    print(f"🔍 Detected {n} disconnected tumor regions.")

    all_stats = []

    for i in range(1, n + 1):
        region = (labeled == i)
        if np.sum(region) < min_voxels:
            continue

        bbox = get_component_bbox(region)
        (z0, z1), (y0, y1), (x0, x1) = bbox

        tumor_crop = region[z0:z1+1, y0:y1+1, x0:x1+1]
        liver_crop = liver_mask[z0:z1+1, y0:y1+1, x0:x1+1]
        volume_crop = volume[z0:z1+1, y0:y1+1, x0:x1+1]

        tumor_vals = volume_crop[tumor_crop == 1]
        liver_vals = volume_crop[(tumor_crop == 0) & (liver_crop == 1)]

        if len(tumor_vals) == 0 or len(liver_vals) == 0:
            continue

        t_mean = np.mean(tumor_vals)
        t_std = np.std(tumor_vals)

        threshold_min = t_mean - k * t_std
        threshold_max = t_mean + k * t_std

        stats = {
            'region': i,
            'bbox': bbox,
            'tumor_voxels': len(tumor_vals),
            'liver_voxels': len(liver_vals),
            'tumor_mean': t_mean,
            'liver_mean': np.mean(liver_vals),
            'tumor_std': t_std,
            'liver_std': np.std(liver_vals),
            'threshold_min': threshold_min,
            'threshold_max': threshold_max
        }
        all_stats.append(stats)

        # Compare histograms
        plt.figure(figsize=(8, 4))
        plt.hist(liver_vals, bins=50, alpha=0.5, label='Liver (bbox)')
        plt.hist(tumor_vals, bins=50, alpha=0.5, label='Tumor')
        # plt.axvline(threshold_min, color='red', linestyle='--', label=f"min: {threshold_min:.3f}")
        # plt.axvline(threshold_max, color='red', linestyle='--', label=f"max: {threshold_max:.3f}")
        plt.title(f"Region #{i} | Tumor vs Liver (BBox)")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"🧠 Region #{i}: tumor_mean={t_mean:.3f} ± {k}·std = [{threshold_min:.3f}, {threshold_max:.3f}]")
        print("Number of tumor voxels:", stats['tumor_voxels'])
        print("Number of liver voxels:", stats['liver_voxels'])
    return all_stats


def plot_slices_with_two_masks(volume, mask1=None, mask2=None,
                                cmap1='Blues', cmap2='Reds',
                                alpha1=0.3, alpha2=0.4,
                                save_path=None, max_slices=100):
    z_slices = volume.shape[0]
    step = max(1, z_slices // max_slices)
    selected_slices = range(0, z_slices, step)

    ncols = 6
    nrows = int(np.ceil(len(selected_slices) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axs = axs.flatten()

    for ax in axs:
        ax.axis('off')

    for i, z in enumerate(selected_slices):
        ax = axs[i]
        ax.imshow(volume[z], cmap='gray')

        if mask1 is not None:
            alpha_mask1 = np.where(mask1[z] == 1, alpha1, 0)
            ax.imshow(mask1[z], cmap=cmap1, alpha=alpha_mask1)

        if mask2 is not None:
            alpha_mask2 = np.where(mask2[z] == 1, alpha2, 0)
            ax.imshow(mask2[z], cmap=cmap2, alpha=alpha_mask2)

        ax.set_title(f"Slice {z}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ File saved as: {save_path}")
        plt.close()
    else:
        plt.show()
