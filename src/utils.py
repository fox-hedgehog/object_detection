import cv2
import numpy as np
from typing import Tuple, List, Dict


def preprocess_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    预处理输入图像，将其转换为模型所需的格式。

    Args:
        image_path (str): 输入图像的路径

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]:
            - 处理后的图像数组 (NCHW格式)
            - 原始图像的尺寸 (height, width)

    Raises:
        ValueError: 当图像无法读取时抛出异常
    """
    # 读取原始图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 保存原始尺寸
    original_shape = original_image.shape[:2]  # (height, width)

    # 图像预处理
    # 1. 调整图像尺寸为模型输入尺寸 (640x640)
    resized_image = cv2.resize(original_image, (640, 640))

    # 2. 将BGR格式转换为RGB格式
    # OpenCV默认使用BGR格式，而大多数深度学习模型使用RGB格式
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # 3. 像素值归一化到[0,1]范围
    # 将uint8类型(0-255)转换为float32类型(0-1)
    normalized_image = rgb_image.astype(np.float32) / 255.0

    # 4. 调整维度顺序
    # 从HWC(Height, Width, Channel)转换为CHW(Channel, Height, Width)
    # 这是PyTorch等深度学习框架常用的格式
    transposed_image = np.transpose(normalized_image, (2, 0, 1))

    # 5. 添加batch维度
    # 从CHW转换为NCHW格式，其中N=1表示batch size为1
    batched_image = np.expand_dims(transposed_image, axis=0)

    return batched_image, original_shape


def postprocess_detections(
    model_output: np.ndarray,
    original_shape: Tuple[int, int],
    confidence_threshold: float = 0.1,
    nms_threshold: float = 0.5,
) -> List[Dict[str, int]]:
    """
    处理模型输出，执行非极大值抑制(NMS)，并转换为指定格式的检测框。

    Args:
        model_output (np.ndarray): 模型输出的检测结果
        original_shape (Tuple[int, int]): 原始图像的尺寸 (height, width)
        confidence_threshold (float): 置信度阈值，默认0.1
        nms_threshold (float): NMS的IoU阈值，默认0.5

    Returns:
        List[Dict[str, int]]: 检测框列表，每个检测框格式为:
            {
                "x": int,  # 左上角x坐标
                "y": int,  # 左上角y坐标
                "w": int,  # 框的宽度
                "h": int   # 框的高度
            }
    """
    # 提取预测框信息
    predicted_boxes = model_output[0][0, :4, :].T  # 转置为 [N, 4] 格式

    # 将中心点坐标和宽高转换为左上角和右下角坐标
    predicted_boxes[:, 2:4] /= 2  # 将宽高除以2
    boxes = np.zeros_like(predicted_boxes, dtype=np.float32)
    boxes[:, 0] = (
        predicted_boxes[:, 0] - predicted_boxes[:, 2]
    )  # x1 = center_x - width/2
    boxes[:, 1] = (
        predicted_boxes[:, 1] - predicted_boxes[:, 3]
    )  # y1 = center_y - height/2
    boxes[:, 2] = (
        predicted_boxes[:, 0] + predicted_boxes[:, 2]
    )  # x2 = center_x + width/2
    boxes[:, 3] = (
        predicted_boxes[:, 1] + predicted_boxes[:, 3]
    )  # y2 = center_y + height/2

    # 执行非极大值抑制
    confidence_scores = model_output[0][0, 4, :]
    nms_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=confidence_scores,
        score_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
    )

    # 将检测框转换回原始图像尺寸并格式化输出
    detection_boxes = []
    for idx in nms_indices:
        x1, y1, x2, y2 = boxes[idx]
        # 将坐标映射回原始图像尺寸
        x1 = x1 * original_shape[1] / 640
        x2 = x2 * original_shape[1] / 640
        y1 = y1 * original_shape[0] / 640
        y2 = y2 * original_shape[0] / 640

        detection_boxes.append(
            {
                "x": round(x1),
                "y": round(y1),
                "w": round(x2 - x1),
                "h": round(y2 - y1),
            }
        )

    return detection_boxes
