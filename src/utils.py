import cv2
import numpy as np

# 预处理图片，将图片转换为模型输入的格式
def preprocess(img_path):
    img = cv2.imread(img_path)  # 读取图片
    img = cv2.resize(img, (640, 640))  # 改为模型输入尺寸
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB
    img = img.astype(np.float32) / 255.0  # 归一化
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # 添加 batch 维度 NCHW
    return img

# 处理网络模型的输出，并调整为指定的格式
def postprocess(results, conf_threshold=0.1, nms_threshold=0.5):
    pboxes = results[0][0, :4, :].T
    pboxes[:, 2:4] /= 2
    boxes = np.zeros_like(pboxes, dtype=np.float32)
    boxes[:, 0] = pboxes[:, 0] - pboxes[:, 2]
    boxes[:, 1] = pboxes[:, 1] - pboxes[:, 3]
    boxes[:, 2] = pboxes[:, 0] + pboxes[:, 2]
    boxes[:, 3] = pboxes[:, 1] + pboxes[:, 3]

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,  # 边界框列表
        scores=results[0][0, 4, :],  # 置信度列表
        score_threshold=conf_threshold,  # 置信度阈值
        nms_threshold=nms_threshold  # IoU 阈值
    )
    # 将边界框转换为指定格式
    output = []
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        output.append({
            "x": round(x1.item()),
            "y": round(y1.item() / 4 * 3),
            "w": round(x2.item() - x1.item()),
            "h": round((y2.item() - y1.item()) / 4 * 3),
        })
    return output


# 将YOLO模型的输出转化为规定的格式
def process_output(results):
    output = []
    for result in results:
        for box in result.boxes:
            output.append({
                "x": round(box.xyxy[0][0].item()),
                "y": round(box.xyxy[0][1].item()),
                "w": round(box.xywh[0][2].item()),
                "h": round(box.xywh[0][3].item()),
            })
    return output






