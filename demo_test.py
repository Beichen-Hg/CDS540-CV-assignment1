import cv2
import pytesseract
import numpy as np
import itertools
import time

def preprocess_image(image_path, blur_kernel, thresh_method):

    # Preprocess the image for OCR.  
      
    # Parameters:  
    # - image_path: str, path to the image file.  
    # - blur_kernel: int, size of the Gaussian blur kernel.  
    # - thresh_method: str, thresholding method ('otsu', 'adaptive', 'fixed').  
      
    # Returns:  
    # - binary_image: np.ndarray, the preprocessed binary image. 

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded, check the file path.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying blur
    if blur_kernel > 0:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Applying threshold
    if thresh_method == 'otsu':
        _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_method == 'adaptive':
        binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
    else:
        _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    return binary_image



def run_ocr(binary_image, psm):

    # Run OCR on the binary image.  
      
    # Parameters:  
    # - binary_image: np.ndarray, the preprocessed binary image.  
    # - psm: int, page segmentation mode for Pytesseract.  
      
    # Returns:  
    # - text: str, recognized text from the image.  

    config = f'--oem 3 --psm {psm}'
    text = pytesseract.image_to_string(binary_image, config=config)
    return text

def test_params(image_path, ground_truth, params_combinations):

    # Test different parameter combinations for preprocessing and OCR.  
      
    # Parameters:  
    # - image_path: str, path to the image file.  
    # - ground_truth: str, expected text for accuracy calculation.  
    # - params_combinations: list of tuples, parameter combinations to test.  
      
    # Returns:  
    # - results: list of tuples, each containing parameters, accuracy, and duration.

    results = []
    for params in params_combinations:
        blur_kernel, thresh_method, psm = params
        try:
            binary_image = preprocess_image(image_path, blur_kernel, thresh_method)
            start_time = time.time()
            detected_text = run_ocr(binary_image, psm)
            duration = time.time() - start_time
            accuracy = np.mean([gt == dt for gt, dt in zip(ground_truth, detected_text)])
            results.append((params, accuracy, duration))
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            continue
    return results

def main():

    # Main function to run OCR parameter testing.

    image_path = 'pic2.jpg'
    ground_truth = """
Introduction
Whenever you create a variable in Python, it has a value with a corresponding data type. 
There are many different data types, such as integers, floats, booleans, and strings, 
all of which we'll cover in this lesson. (This is just a small subset of the available data types 
-- there are also dictionaries, sets, lists, tuples, and much more.)
"""
    
    # Define parameter ranges
    blur_kernels = [1, 3, 5, 7]  # Example kernel sizes for Gaussian blur
    thresh_methods = ['otsu', 'adaptive', 'fixed']
    psm_options = [3, 6, 11]

    # Create combinations of parameters to test
    params_combinations = list(itertools.product(blur_kernels, thresh_methods, psm_options))
    
    # Run the tests
    results = test_params(image_path, ground_truth, params_combinations)
    
    # Analyze results to find the best parameters
    best_params = max(results, key=lambda x: x[1])  # Sort by accuracy
    print(f"Best parameters: {best_params[0]}")
    print(f"Accuracy: {best_params[1]*100:.2f}%")
    print(f"Duration: {best_params[2]:.4f} seconds")

if __name__ == "__main__":
    main() # Run the main function
