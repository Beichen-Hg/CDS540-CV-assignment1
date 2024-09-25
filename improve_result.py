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
    data = pytesseract.image_to_data(binary_image, config=config, output_type=pytesseract.Output.DICT)
    
    # 输出识别的文本
    print("Detected Text:")
    print("\n".join([data['text'][i] for i in range(len(data['text'])) if int(data['conf'][i]) > 60]))
    
    # 可视化处理：绘制单词边界框
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # 置信度过滤
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
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

