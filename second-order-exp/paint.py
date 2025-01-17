import numpy as np
import matplotlib.pyplot as plt
import cv2

def draw_rectangle_on_mask(H, W, a, r, h, rotate=0):
    # 根据给定的长度和长宽比计算矩形的宽度
    w = int(h * r)
    
    # 创建一个全为0的mask
    mask = np.zeros((H, W))
    
    # 随机选取矩形的左上角，确保矩形不会超出mask的边界
    max_x = H - h - a
    max_y = W - w - a
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    
    # 画矩形
    mask[x:x+a, y:y+w+a] = 1  # 上边框
    mask[x+h:x+h+a, y:y+w+a] = 1  # 下边框
    mask[x:x+h+a, y:y+a] = 1  # 左边框
    mask[x:x+h+a, y+w:y+w+a] = 1  # 右边框

    # 如果指定了旋转角度，则旋转mask
    if rotate != 0:
        center = (W // 2, H // 2)
        matrix = cv2.getRotationMatrix2D(center, rotate, 1)
        mask = cv2.warpAffine(mask, matrix, (W, H))
    
    return mask


def draw_triangle_on_mask(H, W, side1, side2, angle, a, rotate=0):
    mask = np.zeros((H, W), dtype=np.uint8)

    # 使用余弦定理计算第三边的长度
    side3 = np.sqrt(side1**2 + side2**2 - 2*side1*side2*np.cos(np.radians(angle)))

    # 如果第三边的长度与基线相同，则返回空的mask，因为这会导致一个线段，而不是三角形
    if np.isclose(side3, side1):
        return mask

    # 使用三角形的基本属性计算三个顶点
    x1, y1 = W // 2 - side1 // 2, H // 2
    x2, y2 = W // 2 + side1 // 2, H // 2

    # 使用夹角和另一边的长度确定第三个点
    x3 = x1 + side2 * np.cos(np.radians(angle))
    y3 = y1 - side2 * np.sin(np.radians(angle))

    # 使用OpenCV的line函数绘制三条边
    cv2.line(mask, (x1, y1), (x2, y2), 255, a)
    cv2.line(mask, (x1, y1), (int(x3), int(y3)), 255, a)
    cv2.line(mask, (int(x3), int(y3)), (x2, y2), 255, a)

    # 如果指定了旋转角度，则旋转
    if rotate != 0:
        center = (W // 2, H // 2)
        matrix = cv2.getRotationMatrix2D(center, -rotate, 1)  # 注意：OpenCV的旋转是逆时针的，因此我们使用-rotate
        mask = cv2.warpAffine(mask, matrix, (W, H))

    return mask

def draw_trapezoid_on_mask(H, W, top_width, bottom_width, height, a, rotate=0):
    mask = np.zeros((H, W), dtype=np.uint8)

    # 确定梯形的中心
    center_x = W // 2
    center_y = H // 2

    # 计算梯形的四个顶点
    top_left = (int(center_x - top_width / 2), int(center_y - height / 2))
    top_right = (int(center_x + top_width / 2), int(center_y - height / 2))
    bottom_left = (int(center_x - bottom_width / 2), int(center_y + height / 2))
    bottom_right = (int(center_x + bottom_width / 2), int(center_y + height / 2))

    # 使用OpenCV的line函数绘制梯形的边框
    cv2.line(mask, top_left, top_right, 255, a)
    cv2.line(mask, bottom_left, bottom_right, 255, a)
    cv2.line(mask, top_left, bottom_left, 255, a)
    cv2.line(mask, top_right, bottom_right, 255, a)

    # 旋转（如果指定了旋转角度）
    if rotate != 0:
        center = (W // 2, H // 2)
        matrix = cv2.getRotationMatrix2D(center, -rotate, 1)  # 注意：OpenCV的旋转是逆时针的，因此我们使用-rotate
        mask = cv2.warpAffine(mask, matrix, (W, H))

    return mask


def draw_circle_on_mask(H, W, radius, eccentricity, a, rotate=0):
    mask = np.zeros((H, W), dtype=np.uint8)

    # 确定椭圆的中心
    center = (W // 2, H // 2)

    # 根据偏心率计算椭圆的主轴和次轴
    if eccentricity < 0.5:
        major_axis = radius
        minor_axis = int(major_axis * np.sqrt(1 - eccentricity**2))
    else:
        minor_axis = radius
        major_axis = int(minor_axis / np.sqrt(1 - eccentricity**2))

    # 使用OpenCV的ellipse函数绘制椭圆
    cv2.ellipse(mask, center, (major_axis, minor_axis), 0, 0, 360, 255, a)

    # 旋转（如果指定了旋转角度）
    if rotate != 0:
        matrix = cv2.getRotationMatrix2D(center, -rotate, 1)  # 注意：OpenCV的旋转是逆时针的，因此我们使用-rotate
        mask = cv2.warpAffine(mask, matrix, (W, H))

    return mask


def draw_shape_on_mask(H, W, a, rotate=0):
    shape_type = np.random.choice(['rectangle', 'triangle', 'trapezoid', 'circle'])
    
    if shape_type == 'rectangle':
        r = np.random.uniform(0.5, 2)
        h = np.random.randint(H//10, H//3)
        mask =  draw_rectangle_on_mask(H, W, a, r, h, rotate)
        print(1)
    elif shape_type == 'triangle':
        side1 = np.random.randint(H//10, H//2)
        side2 = np.random.randint(0.5,2) * side1
        angle = np.random.randint(30, 120)
        mask =  draw_triangle_on_mask(H, W, side1, side2, angle, a, rotate)
        print(2)
    elif shape_type == 'trapezoid':
        top_width = np.random.randint(H//10, W//2)
        bottom_width = np.random.randint(0.5, 2) * top_width
        height = np.random.randint(H//10, H//2)
        mask =  draw_trapezoid_on_mask(H, W, top_width, bottom_width, height, a, rotate)
        print(3)
    elif shape_type == 'circle':
        radius = np.random.randint(min(H, W)//12, min(H, W)//8)
        eccentricity = np.random.uniform(0.7, 1.0)
        mask =  draw_circle_on_mask(H, W, radius, eccentricity, a, rotate)
        print(4)
    # find index of non-zero elements
    idx = np.argwhere(mask)
    # judege if the idx is close to the edge
    if len(idx) == 0 or np.min(idx[:,0]) < 5 or np.min(idx[:,1]) < 5 or np.max(idx[:,0]) > H-5 or np.max(idx[:,1]) > W-5 :
        mask = draw_shape_on_mask(H, W, a, rotate)
    return mask
while True:
    width = np.random.randint(4,20)
    angle = np.random.randint(0,360)
    mask = draw_shape_on_mask(256, 256, width, angle)
    plt.imshow(mask, cmap='gray')
    plt.show()