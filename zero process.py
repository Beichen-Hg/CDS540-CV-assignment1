import cv2
import pytesseract
import time

# Step 1: Load an image
image_path = 'pic2.jpg'
image = cv2.imread(image_path)

# Step 2: Pre-process the image (convert to grayscale)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Step 3: Text Detection using Tesseract
custom_config = r'--oem 3 --psm 6'

# Measure the processing time
start_time = time.time()
detected_text = pytesseract.image_to_string(gray, config=custom_config)
end_time = time.time()

# Print detected text
print("Detected text:")
print(detected_text)

# Step 4: Performance Evaluation - Measure speed
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.4f} seconds")

#Step 5: Draw bounding boxes around detected text using basic Tesseract functions
h, w, _ = image.shape
boxes = pytesseract.image_to_boxes(gray, config=custom_config)

for b in boxes.splitlines():
    b = b.split(' ')
    x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    image = cv2.rectangle(image, (x, h - y), (x2, h - y2), (0, 255, 0), 2)

# Step 6: Display the result
cv2.imshow('Detected Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Step 7: Accuracy Evaluation (example with known ground truth)
ground_truth_text = """
Introduction
Whenever you create a variable in Python, it has a value with a corresponding data type. 
There are many different data types, such as integers, floats, booleans, and strings, 
all of which we'll cover in this lesson. (This is just a small subset of the available data types 
-- there are also dictionaries, sets, lists, tuples, and much more.)
"""

def calculate_accuracy(extracted_text, ground_truth_text):

    extracted_text = extracted_text.replace('\n', ' ').replace('\r', '').strip()
    ground_truth_text = ground_truth_text.replace('\n', ' ').replace('\r', '').strip()
    
    correct = 0
    total = len(ground_truth_text)

    for i in range(total):
        if i < len(extracted_text) and extracted_text[i] == ground_truth_text[i]:
            correct += 1

    accuracy = correct / total
    return accuracy

accuracy = calculate_accuracy(detected_text, ground_truth_text)
print(f"Accuracy: {accuracy * 100:.2f}%")