import cv2  
import pytesseract  
  
def apply_ocr(image_path, blur_kernel, thresh_method, psm):  
    # Load the image  
    image = cv2.imread(image_path)  
    # Convert the image to grayscale  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    # Apply Gaussian blur if the kernel size is greater than 1  
    if blur_kernel > 1:  
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)  
      
    # Apply thresholding based on the selected method  
    if thresh_method == 'otsu':  
        _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    elif thresh_method == 'adaptive':  
        binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  
    else:  
        _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  
  
    # Perform OCR using Tesseract  
    config = f'--oem 3 --psm {psm}'  
    detected_text = pytesseract.image_to_string(binary_image, config=config)  
      
    # Print the detected text  
    print("Detected Text:")  
    print(detected_text)  
      
    # Get bounding boxes around the recognized text  
    boxes = pytesseract.image_to_boxes(binary_image, config=config)  
    h, w = binary_image.shape  
    for b in boxes.splitlines():  
        b = b.split(' ')  
        x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])  
        # Draw rectangles around the recognized text  
        image = cv2.rectangle(image, (x, h - y), (x2, h - y2), (0, 255, 0), 2)  
      
    # Display the image with OCR results  
    cv2.imshow('OCR Results', image)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  
  
# Assuming optimal parameters are found  
optimal_blur_kernel = 1  
optimal_thresh_method = 'otsu'  
optimal_psm = 11  
image_path = 'pic2.jpg'  
  
# Apply OCR with optimal parameters  
apply_ocr(image_path, optimal_blur_kernel, optimal_thresh_method, optimal_psm)