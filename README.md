demo_test:
This script reads an image, preprocesses it (grayscale, blur, thresholding), runs OCR, and tests different combinations of parameters (blur kernel size, threshold method, and page segmentation modes) to find the most accurate OCR settings using a ground truth text for comparison. Results are printed for the best parameters based on accuracy and processing time.
Second Script:

demo_result:
Reads an image, applies specified preprocessing, and performs OCR. Additionally, this script visualizes the OCR results by drawing bounding boxes around recognized words and displays the image with these annotations.

improve_test:
Includes Levenshtein Distance: Improves upon the initial testing method by employing Levenshtein distance to compute the accuracy of OCR results more precisely. Draws bounding boxes around words based on confidence scores.

improve_result:
Detailed Output and Visualization: Similar to the demo_result script but extends the output details by listing recognized words filtered by confidence scores. It visualizes the results in a similar manner with bounding boxes around high-confidence words.

Use the picture (pic2.png) as an example, the improve_test compared with the demo_test the accuracy has been improve from 55.43% to 96.94%.
