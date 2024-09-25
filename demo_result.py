import cv2
import pytesseract

def apply_ocr(image_path, blur_kernel, thresh_method, psm):
    # 图像预处理
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

    # OCR 处理
    config = f'--oem 3 --psm {psm}'
    detected_text = pytesseract.image_to_string(binary_image, config=config)
    
    # 输出识别的文本
    print("Detected Text:")
    print(detected_text)
    
    # 可视化处理：绘制边界框
    boxes = pytesseract.image_to_boxes(binary_image, config=config)
    h, w = binary_image.shape
    for b in boxes.splitlines():
        b = b.split(' ')
        x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        image = cv2.rectangle(image, (x, h - y), (x2, h - y2), (0, 255, 0), 2)
    
    # 显示图像
    cv2.imshow('OCR Results', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 假设已经找到最优参数
optimal_blur_kernel = 1
optimal_thresh_method = 'otsu'
optimal_psm = 11
image_path = 'pic2.jpg'

# 应用最优参数
apply_ocr(image_path, optimal_blur_kernel, optimal_thresh_method, optimal_psm)
