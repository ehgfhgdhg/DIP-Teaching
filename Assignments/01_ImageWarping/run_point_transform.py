import cv2
import numpy as np
import gradio as gr
from typing import Callable

def verticals(p:np.floating):
    q = p.copy()
    if len(p.shape) == 1:
        q[0], q[1] = -q[1], q[0]
    elif len(p.shape) == 2:
        q[:, 0], q[:, 1] = -q[:, 1].copy(), q[:, 0].copy()
    else:
        assert False
    return q

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 对数组组成的图像执行任意变换。输入为所需执行变换的逆变换，即
# $$\texttt{dst} (x,y) =  \texttt{src} (map_x(x,y),map_y(x,y))$$
# * mapping: [float, float] -> [float, float] 映射输出图像上的点到输入图像。
def np_remap(image:np.floating, mapping:Callable):
    N = 50 # sample number
    small_m = np.array([[mapping(i,j) for j in np.linspace(0,image.shape[1]-1,N)] for i in np.linspace(0,image.shape[0]-1,N)], np.float32)

    step0, step1 = (image.shape[0]-1) / (N-1), (image.shape[1]-1) / (N-1)
    m0 = np.arange(image.shape[0], dtype=np.float32).reshape(-1,1).repeat(image.shape[1], axis=1)
    m1 = np.arange(image.shape[1], dtype=np.float32).reshape(1,-1).repeat(image.shape[0], axis=0)
    m0, m1 = m0.flatten()/step0, m1.flatten()/step1
    m0_s, m1_s = np.fmin(N-2,np.int32(np.floor(m0))), np.fmin(N-2,np.int32(np.floor(m1)))
    m0_s_1, m1_s_1 = m0_s+1, m1_s+1
    map_0 = small_m[m0_s_1, m1_s_1, 0] * (m0 - m0_s) * (m1 - m1_s) + \
            small_m[m0_s_1, m1_s, 0] * (m0 - m0_s) * (m1_s_1 - m1) + \
            small_m[m0_s, m1_s_1, 0] * (m0_s_1 - m0) * (m1 - m1_s) + \
            small_m[m0_s, m1_s, 0] * (m0_s_1 - m0) * (m1_s_1 - m1)
    map_1 = small_m[m0_s_1, m1_s_1, 1] * (m0 - m0_s) * (m1 - m1_s) + \
            small_m[m0_s_1, m1_s, 1] * (m0 - m0_s) * (m1_s_1 - m1) + \
            small_m[m0_s, m1_s_1, 1] * (m0_s_1 - m0) * (m1 - m1_s) + \
            small_m[m0_s, m1_s, 1] * (m0_s_1 - m0) * (m1_s_1 - m1)
    map_0, map_1 = np.float32(map_0.reshape(image.shape[:2])), np.float32(map_1.reshape(image.shape[:2]))

    return cv2.remap(image, map_1, map_0, cv2.INTER_LINEAR)

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    ### FILL: 基于MLS or RBF 实现 image warping
    def weight(p, v, alpha):
        return (eps + np.linalg.norm(p-v, ord=2, axis=1)) ** (-2 * alpha)
    def centroid(w, v) -> float:
        return np.inner(w.transpose(), v.transpose()).transpose() / w.sum();

    def affine_map_A(w, p_hat, p_star, v):
        try:
            return (v - p_star) @ np.linalg.inv((w * p_hat.transpose()) @ p_hat) @ (w * p_hat.transpose())
        except np.linalg.LinAlgError:
            return np.float32([1])
    def get_transformation_affine(p, v):
        w = weight(p, v, alpha)
        p_star = centroid(w, p)
        p_hat = p - p_star
        A = affine_map_A(w, p_hat, p_star, v)
        def transformation(q):
            q_star = centroid(w, q)
            q_hat = q - q_star
            return (A * q_hat.transpose()).transpose().sum(axis=0) + q_star
        return transformation

    def similarity_rigid_map_A(w, p_hat, p_star, v):
        try:
            p_hat_v = verticals(p_hat)
            v_p_star = (v - p_star).reshape(1,2)
            v_p_star_v = verticals(v_p_star)
            p_hat = p_hat.reshape(1,-1,2).transpose(0,2,1)
            p_hat_v = p_hat_v.reshape(1,-1,2).transpose(0,2,1)
            P = np.concatenate((p_hat,-p_hat_v), 0);
            Q = np.concatenate((v_p_star,-v_p_star_v), 0).transpose()
            # w.reshape(1,1,-1) * np.einsum('pqi,qr', P, Q).transpose(0,2,1)
            return w.reshape(1,1,-1) * np.tensordot(P, Q, axes=(1,0)).transpose(0,2,1)
        except np.linalg.LinAlgError:
            return np.float32([[1, 0], [0, 1]]).repeat(p_hat.shape[0], 2)
    def get_transformation_similarity(p, v):
        w = weight(p, v, alpha)
        p_star = centroid(w, p)
        p_hat = p - p_star
        A = similarity_rigid_map_A(w, p_hat, p_star, v)
        mu_s = w @ (p_hat ** 2).sum(axis=1) + eps
        def transformation(q):
            q_star = centroid(w, q)
            q_hat = q - q_star
            # np.einsum('kj,jik', (q_hat, A / mu_s)) + q_star
            return np.tensordot(q_hat, A / mu_s, axes=([1,0],[0,2])) + q_star
        return transformation

    def get_transformation_rigid(p, v):
        w = weight(p, v, alpha)
        p_star = centroid(w, p)
        p_hat = p - p_star
        A = similarity_rigid_map_A(w, p_hat, p_star, v)
        def transformation(q):
            q_star = centroid(w, q)
            q_hat = q - q_star
            # np.einsum('kj,jik', (q_hat, A)) + q_star
            f_r = np.tensordot(q_hat, A, axes=([1,0],[0,2]))
            return np.linalg.norm(v - p_star) / (np.linalg.norm(f_r) + eps) * f_r + q_star
        return transformation

    def mapping(x, y):
        transformation = get_transformation_rigid(target_pts, np.float32([y, x])) # reversed!
        return np.flip(transformation(source_pts).flatten()) # reversed!

    warped_image = np_remap(image, mapping)
    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    if len(points_dst) > 0:
        points_src_trimmed = points_src[:len(points_dst)]
        warped_image = point_guided_deformation(np.array(image), np.array(points_src_trimmed), np.array(points_dst))
        return warped_image
    else:
        return image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
