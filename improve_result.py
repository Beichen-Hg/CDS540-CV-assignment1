import cv2  
import pytesseract  
  
def apply_ocr(image_path, blur_kernel, thresh_method, psm):  
    # Image preprocessing  
    image = cv2.imread(image_path)  # Read the image from the file  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale  
    if blur_kernel > 1:  
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)  # Apply Gaussian blur if kernel size is greater than 1  
      
    # Thresholding the grayscale image  
    if thresh_method == 'otsu':  
        _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply Otsu's thresholding  
    elif thresh_method == 'adaptive':  
        binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Apply adaptive thresholding  
    else:  
        _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # Apply simple binary thresholding with a default value of 128  
  
    # OCR processing  
    config = f'--oem 3 --psm {psm}'  # Set OCR configuration (OEM and PSM)  
    data = pytesseract.image_to_data(binary_image, config=config, output_type=pytesseract.Output.DICT)  # Run OCR on the binary image  
      
    # Output the recognized text  
    print("Detected Text:")  
    # Print text with confidence higher than 60  
    print("\n".join([data['text'][i] for i in range(len(data['text'])) if int(data['conf'][i]) > 60]))  
      
    # Visualization: Draw bounding boxes around words  
    n_boxes = len(data['text'])  
    for i in range(n_boxes):  
        if int(data['conf'][i]) > 60:  # Confidence filtering  
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])  
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around the text  
      
    # Display the image  
    cv2.imshow('OCR Results', image)  # Show the image with OCR results  
    cv2.waitKey(0)  # Wait for a key press  
    cv2.destroyAllWindows()  # Close all windows  
  
# Assuming optimal parameters are found  
optimal_blur_kernel = 1  
optimal_thresh_method = 'otsu'  
optimal_psm = 11  
image_path = 'pic2.jpg'  
  
# Apply OCR with optimal parameters  
apply_ocr(image_path, optimal_blur_kernel, optimal_thresh_method, optimal_psm)
