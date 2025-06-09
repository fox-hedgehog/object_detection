import os
import time
from ultralytics import YOLO
import onnxruntime as ort

from src.utils import *

#
# 参数:
#   img_path: 要识别的图片的路径
#
# 返回:
#   返回结果为各赛题中要求的识别结果，具体格式可参考提供压缩包中的 “图片对应输出结果.txt” 中一张图片对应的结果
#


def process_img(img_path):
    # 加载模型
    if not hasattr(process_img, "session"):
        process_img.session = ort.InferenceSession("mushroom_detect.onnx", providers=["CPUExecutionProvider"])
        # process_img.session = ort.InferenceSession("detection_int8.onnx", providers=["CPUExecutionProvider"])
        process_img.input_name = process_img.session.get_inputs()[0].name

    session = process_img.session
    input_name = process_img.input_name

    # 图片预处理
    img = preprocess(img_path)

    # 执行推理
    results = session.run(None, {input_name: img})

    # 后处理
    output = postprocess(results)

    return output

def process_img_test(img_path):
    load_time = time.time()
    # 加载微调后的模型进行蘑菇检测

    if not hasattr(process_img_test, "session"):
        process_img_test.session = ort.InferenceSession("mushroom_detect.onnx", providers=["CPUExecutionProvider"])
        # process_img_test.session = ort.InferenceSession("detection_int8.onnx", providers=["CPUExecutionProvider"])
        process_img_test.input_name = process_img_test.session.get_inputs()[0].name

    session = process_img_test.session
    input_name = process_img_test.input_name

    load_time = time.time() - load_time

    # 预处理图片
    preprocess_time = time.time()
    img = preprocess(img_path)
    preprocess_time = time.time() - preprocess_time

    # 执行推理
    iterence_time = time.time()
    results = session.run(None, {input_name: img})
    inference_time = time.time() - iterence_time
    
    # 处理模型输出并调整为指定格式
    postprocess_time = time.time()
    output = postprocess(results)
    postprocess_time = time.time() - postprocess_time
    
    return output, inference_time, postprocess_time, preprocess_time, load_time

#
# 以下代码仅作为选手测试代码时使用，仅供参考，可以随意修改
# 但是最终提交代码后，process.py文件是作为模块进行调用，而非作为主程序运行
# 因此提交时请根据情况删除不必要的额外代码
#

if __name__ == "__main__":
    imgs_folder = "./imgs/"
    img_paths = os.listdir(imgs_folder)

    def now():
        return int(time.time() * 1000)

    last_time = count_time = max_time =  0
    min_time = now()
    inf_time = post_time = pre_time = lo_time = 0

    for img_path in img_paths:
        print(img_path, ":")
        last_time = now()
        result, inference_time, postprocess_time, preprocess_time, load_time = process_img_test(imgs_folder + img_path)
        inf_time += inference_time
        post_time += postprocess_time
        pre_time += preprocess_time
        lo_time += load_time
        run_time = now() - last_time
        print("result:\n", result)
        print("run time: ", run_time, "ms")
        print()
        count_time += run_time
        if run_time > max_time:
            max_time = run_time
        if run_time < min_time:
            min_time = run_time
    
    print("\n")
    print("avg time: ", int(count_time / len(img_paths)), "ms")
    print("max time: ", max_time, "ms")
    print("min time: ", min_time, "ms")
    print("load time: ", lo_time * 1000 / len(img_paths), "ms")
    print("preprocess time: ", pre_time * 1000 / len(img_paths), "ms")
    print("inference time: ", inf_time * 1000 / len(img_paths), "ms")
    print("postprocess time: ", post_time * 1000 / len(img_paths), "ms")