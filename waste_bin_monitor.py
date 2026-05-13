import cv2
import numpy as np
import argparse
import os

def calculate_fill_level(image_path, bin_mask_path=None, save_path="output_result.jpg", show_window=False):
    """
    Calculates the fill level of a waste bin using HSV thresholding and pixel ratio.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # 1. Define the Bin Area (Region of Interest)
    # If no mask is provided, we assume a central crop is the bin.
    # In a real IoT/Smart City application, you'd have a fixed camera and a predefined mask.
    height, width = img.shape[:2]
    
    if bin_mask_path and os.path.exists(bin_mask_path):
        bin_mask = cv2.imread(bin_mask_path, cv2.IMREAD_GRAYSCALE)
        _, bin_mask = cv2.threshold(bin_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        # Default: Use the central 80% width and bottom 90% height as the bin area
        bin_mask = np.zeros((height, width), dtype=np.uint8)
        margin_y_top = int(height * 0.1)
        margin_y_bottom = int(height * 0.05)
        margin_x = int(width * 0.1)
        bin_mask[margin_y_top:height-margin_y_bottom, margin_x:width-margin_x] = 255

    total_bin_pixels = cv2.countNonZero(bin_mask)

    # Apply the mask to the image to ignore background outside the bin
    masked_img = cv2.bitwise_and(img, img, mask=bin_mask)

    # 2. Convert to HSV color space
    hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

    # 3. HSV Waste Thresholding
    # Assuming the inside of an empty bin is dark, we threshold for "non-dark" colors
    # and typical waste colors (like white plastics, colorful paper, etc.)
    
    # Adjust these HSV ranges based on your specific camera and bin color!
    # Range 1: Colorful waste (Saturation > 20, Value > 40)
    lower_waste = np.array([0, 20, 40])
    upper_waste = np.array([179, 255, 255])
    
    # Range 2: White/Grey waste (Low saturation, High value)
    lower_white_waste = np.array([0, 0, 100])
    upper_white_waste = np.array([179, 40, 255])

    mask1 = cv2.inRange(hsv, lower_waste, upper_waste)
    mask2 = cv2.inRange(hsv, lower_white_waste, upper_white_waste)
    
    # Combine masks to detect all potential waste
    waste_mask = cv2.bitwise_or(mask1, mask2)
    
    # Only consider waste within the defined bin area
    waste_mask = cv2.bitwise_and(waste_mask, bin_mask)

    # Morphological operations to remove noise and fill small holes
    kernel = np.ones((5,5), np.uint8)
    waste_mask = cv2.morphologyEx(waste_mask, cv2.MORPH_OPEN, kernel) # Remove small noise
    waste_mask = cv2.morphologyEx(waste_mask, cv2.MORPH_CLOSE, kernel) # Fill small holes

    # 4. Calculate Pixel Ratio
    waste_pixels = cv2.countNonZero(waste_mask)
    fill_ratio = waste_pixels / total_bin_pixels if total_bin_pixels > 0 else 0
    fill_percentage = fill_ratio * 100

    print("-" * 30)
    print("Waste Bin Monitoring Results")
    print("-" * 30)
    print(f"Total Bin Area (Pixels): {total_bin_pixels}")
    print(f"Waste Detected (Pixels): {waste_pixels}")
    print(f"Estimated Fill Level:    {fill_percentage:.2f}%")
    
    if fill_percentage > 80:
        print("ALERT: Bin is almost full! Dispatch collection vehicle.")
    print("-" * 30)

    if save_path or show_window:
        # Visualize the result
        result_img = img.copy()
        
        # Highlight waste area in red
        result_img[waste_mask == 255] = [0, 0, 255]
        
        # Draw bin boundary (green box based on mask)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)

        # Add text overlay
        text = f"Fill Level: {fill_percentage:.1f}%"
        color = (0, 0, 255) if fill_percentage > 80 else (0, 255, 0)
        cv2.putText(result_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Resize for display if the image is too large
        display_scale = 1.0
        if height > 600:
            display_scale = 600 / height
            
        new_w = int(width * display_scale)
        new_h = int(height * display_scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        mask_resized = cv2.resize(cv2.cvtColor(waste_mask, cv2.COLOR_GRAY2BGR), (new_w, new_h))
        res_resized = cv2.resize(result_img, (new_w, new_h))

        # Add labels to the combined image
        cv2.putText(img_resized, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(mask_resized, "Waste Mask (HSV)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(res_resized, "Result Overlay", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined = np.hstack((img_resized, mask_resized, res_resized))
        
        if save_path:
            # Save the result image
            cv2.imwrite(save_path, combined)
            print(f"Saved visualization to {save_path}")

        if show_window:
            # Try to show window (might fail in some headless environments)
            try:
                cv2.imshow('Smart City Waste Bin Monitor', combined)
                print("Press any key on the image window to close it...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print("Could not display window (headless mode?). Check saved output instead.")

    return fill_percentage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart City Waste Bin Fill Level Monitor using HSV & Pixel Ratio")
    parser.add_argument("--image", required=True, help="Path to the input image of the bin")
    parser.add_argument("--mask", default=None, help="Optional path to a binary mask image defining the bin area")
    parser.add_argument("--no-show", action="store_true", help="Do not attempt to display the result window")
    args = parser.parse_args()

    calculate_fill_level(args.image, args.mask, show_window=not args.no_show)
