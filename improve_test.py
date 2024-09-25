import cv2
import pytesseract
import numpy as np
import itertools
import time
from Levenshtein import distance as levenshtein_distance

def preprocess_image(image_path, blur_kernel, thresh_method):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    if thresh_method == 'otsu':
        _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_method == 'adaptive':
        binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary_image, gray

def run_ocr(binary_image, psm):
    config = f'--oem 3 --psm {psm}'
    data = pytesseract.image_to_data(binary_image, config=config, output_type=pytesseract.Output.DICT)
    return data

def draw_boxes_around_words(image, data):
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:  # Confidence level over 60
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def test_params(image_path, ground_truth, params_combinations):
    best_accuracy = 0
    best_image = None
    best_params = None
    for params in params_combinations:
        blur_kernel, thresh_method, psm = params
        binary_image, original_image = preprocess_image(image_path, blur_kernel, thresh_method)
        ocr_data = run_ocr(binary_image, psm)
        detected_text = ' '.join([ocr_data['text'][i] for i in range(len(ocr_data['text'])) if int(ocr_data['conf'][i]) > 60])
        accuracy = 1 - levenshtein_distance(ground_truth, detected_text) / max(len(ground_truth), len(detected_text))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_image = draw_boxes_around_words(original_image.copy(), ocr_data)
    return best_params, best_accuracy, best_image

def main():
    image_path = 'pic2.jpg'
    ground_truth = """
Introduction
Whenever you create a variable in Python, it has a value with a corresponding data type. 
There are many different data types, such as integers, floats, booleans, and strings, 
all of which we'll cover in this lesson. (This is just a small subset of the available data types 
-- there are also dictionaries, sets, lists, tuples, and much more.)
"""
    blur_kernels = [1, 3, 5, 7]
    thresh_methods = ['otsu', 'adaptive', 'fixed']
    psm_options = [3, 6, 11]
    params_combinations = list(itertools.product(blur_kernels, thresh_methods, psm_options))
    best_params, best_accuracy, best_image = test_params(image_path, ground_truth, params_combinations)
    
    if best_image is not None:
        #save_path = f"ocr_results/best_result.jpg"
        #cv2.imwrite(save_path, best_image)
        print("Best Parameters:", best_params)
        print("Highest Accuracy: {:.2f}%".format(best_accuracy * 100))
        #print(f"Best result image saved to {save_path}")

if __name__ == "__main__":
    main()
