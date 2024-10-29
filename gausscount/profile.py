"""
profile.py
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def randomized_scan_order(nrows, ncols):
    indices = [(i, j) for i in range(1, nrows-1) for j in range(1, ncols-1)]
    np.random.shuffle(indices)
    return indices

def extract_3x3_patches(original_frames, processed_frames):
    """
    Extract 3x3 patches from original frames based on hits identified in processed frames.
    
    Args:
    - original_frames: numpy array of original frames (non-baseline-subtracted, non-thresholded).
    - processed_frames: numpy array of frames after baseline subtraction and thresholding.
    
    Returns:
    - numpy array of 3x3 patches centered around identified hits.
    """
    hit_patches = []
    nrows, ncols = processed_frames.shape[1], processed_frames.shape[2]  # Frame dimensions

    for frame_index in range(processed_frames.shape[0]):
        processed_frame = processed_frames[frame_index]
        original_frame = original_frames[frame_index]
        
        # Iterate through all pixels in the processed frame to find hits
        indices = randomized_scan_order(nrows, ncols)
        for i, j in indices:
            if processed_frame[i, j] > 0:  # Center of a hit in processed frame
                
                # Extract the corresponding 3x3 patch from the original frame
                patch = original_frame[i-1:i+2, j-1:j+2]
                hit_patches.append(patch)

    return np.array(hit_patches)

def gaussian_profile(file_path, nframes, baseline, th_single_elec, plot_results=True):
    """
    Process frames to extract the average 3x3 patch of electron hits, fit it to a Gaussian, 
    and return the optimized Gaussian parameters and patches.
    
    Args:
    - file_path (str): Path to the HDF5 file containing the frames.
    - nframes (int): Number of frames to process.
    - baseline (float): Baseline value to subtract from the frames.
    - th_single_elec (float): Threshold for identifying single electron hits.
    - plot_results (bool): If True, plots the original average patch, the optimized Gaussian, and their difference.

    Returns:
    - avg_patch (np.ndarray): The average 3x3 patch computed from the identified hits.
    - optimized_patch (np.ndarray): The 3x3 patch after fitting to the Gaussian model.
    - A_opt (float): The optimized amplitude of the Gaussian.
    - sigma_opt (float): The optimized standard deviation (width) of the Gaussian.
    """
    
    print(f"Processing scan at {file_path}...")
    
    with h5py.File(file_path, 'r') as f0:
        data = f0['frames']
        frames = data[0:nframes, :, :]
        
        # Subtract baseline and apply threshold
        sub_frames = frames - baseline
        sub_frames_th = np.where(sub_frames >= th_single_elec, sub_frames, 0)
        
        # Extract the 3x3 patches
        hit_patches = extract_3x3_patches(sub_frames, sub_frames_th)
        avg_patch = np.mean(hit_patches, axis=0)

        # Define the Gaussian function for fitting
        def gaussian(x, y, A, sigma):
            return A * np.exp(-((x-1)**2 + (y-1)**2) / (2*sigma**2))

        # Objective function for optimization
        def objective(params):
            A, sigma = params
            predicted = np.array([[gaussian(x, y, A, sigma) for y in range(3)] for x in range(3)])
            return np.mean((avg_patch - predicted)**2)

        # Initial guess for the Gaussian parameters
        initial_guess = [np.max(avg_patch), 1.0]
        
        # Perform the optimization to find the best-fit Gaussian parameters
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(0, None), (0, None)])
        
        A_opt, sigma_opt = result.x
        
        # Generate the optimized 3x3 Gaussian patch
        optimized_patch = np.array([[gaussian(x, y, A_opt, sigma_opt) for y in range(3)] for x in range(3)])
        
        if plot_results:
            # Plotting
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original averaged patch
            ax0 = axs[0].imshow(avg_patch, cmap='viridis')
            fig.colorbar(ax0, ax=axs[0])
            axs[0].set_title('Original Avg. Patch')
            axs[0].axis('off')
            
            # Optimized Gaussian patch
            ax1 = axs[1].imshow(optimized_patch, cmap='viridis')
            fig.colorbar(ax1, ax=axs[1])
            axs[1].set_title('Optimized Gaussian Patch')
            axs[1].axis('off')
            
            # Difference
            ax2 = axs[2].imshow(avg_patch - optimized_patch, cmap='viridis')
            fig.colorbar(ax2, ax=axs[2])
            axs[2].set_title('Difference (Original - Optimized)')
            axs[2].axis('off')
            
            fig2, axs2 = plt.subplots(1, 1, figsize=(15, 5))
            axs2.hist(hit_patches[:,1,1],bins=100)
            axs2.set_yscale('log')

            plt.show()
        
        # Return the key results
        return avg_patch, optimized_patch, A_opt, sigma_opt
 