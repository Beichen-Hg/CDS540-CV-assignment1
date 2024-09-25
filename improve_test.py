import cv2  
import pytesseract  
import numpy as np  
import itertools  
import time  
from Levenshtein import distance as levenshtein_distance  
  
# Function to preprocess the image  
def preprocess_image(image_path, blur_kernel, thresh_method):  
    image = cv2.imread(image_path)  # Load the image  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale  
    if blur_kernel > 1:  
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)  # Apply Gaussian blur  
    # Apply thresholding based on the method  
    if thresh_method == 'otsu':  
        _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    elif thresh_method == 'adaptive':  
        binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  
    else:  
        _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  
    return binary_image, gray  # Return the binary and grayscale images  
  
# Function to run OCR on the preprocessed image  
def run_ocr(binary_image, psm):  
    config = f'--oem 3 --psm {psm}'  # Set OCR configuration  
    data = pytesseract.image_to_data(binary_image, config=config, output_type=pytesseract.Output.DICT)  # Run OCR  
    return data  # Return OCR data  
  
# Function to draw boxes around recognized words with high confidence  
def draw_boxes_around_words(image, data):  
    for i in range(len(data['text'])):  
        if int(data['conf'][i]) > 60:  # Check confidence level  
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])  
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle  
    return image  # Return the image with boxes  
  
# Function to test different OCR parameters and find the best combination  
def test_params(image_path, ground_truth, params_combinations):  
    best_accuracy = 0  # Initialize best accuracy  
    best_image = None  # Initialize best image  
    best_params = None  # Initialize best parameters  
    for params in params_combinations:  # Iterate through parameter combinations  
        blur_kernel, thresh_method, psm = params  
        binary_image, original_image = preprocess_image(image_path, blur_kernel, thresh_method)  # Preprocess image  
        ocr_data = run_ocr(binary_image, psm)  # Run OCR  
        # Join recognized text with confidence over 60  
        detected_text = ' '.join([ocr_data['text'][i] for i in range(len(ocr_data['text'])) if int(ocr_data['conf'][i]) > 60])  
        # Calculate accuracy using Levenshtein distance  
        accuracy = 1 - levenshtein_distance(ground_truth, detected_text) / max(len(ground_truth), len(detected_text))  
        # Update best accuracy, parameters, and image if current accuracy is higher  
        if accuracy > best_accuracy:  
            best_accuracy = accuracy  
            best_params = params  
            best_image = draw_boxes_around_words(original_image.copy(), ocr_data)  
    return best_params, best_accuracy, best_image  # Return best parameters, accuracy, and image  
  
# Main function  
def main():  
    image_path = 'pic2.jpg'  # Path to the input image  
    ground_truth = """
Introduction
Whenever you create a variable in Python, it has a value with a corresponding data type. 
There are many different data types, such as integers, floats, booleans, and strings, 
all of which we'll cover in this lesson. (This is just a small subset of the available data types 
-- there are also dictionaries, sets, lists, tuples, and much more.)
"""  # Ground truth text (shortened for brevity)  
    # Define parameter ranges  
    blur_kernels = [1, 3, 5, 7]  
    thresh_methods = ['otsu', 'adaptive', 'fixed']  
    psm_options = [3, 6, 11]  
    # Generate all parameter combinations  
    params_combinations = list(itertools.product(blur_kernels, thresh_methods, psm_options))  
    # Test parameters and get the best result  
    best_params, best_accuracy, best_image = test_params(image_path, ground_truth, params_combinations)  
    # Output the best parameters and accuracy  
    if best_image is not None:  
        print("Best Parameters:", best_params)  
        print("Highest Accuracy: {:.2f}%".format(best_accuracy * 100))  
        # Optionally save the best result image (code commented out)  
  
if __name__ == "__main__":  
    main()  # Run the main function