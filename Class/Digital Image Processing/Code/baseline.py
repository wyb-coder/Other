"""
数字图像处理作业2：人体部件检测与增强现实系统
基线系统 - 人脸及器官检测模块
"""

import cv2
import numpy as np
import os

# ======================== 配置 ========================
CASCADE_DIR = os.path.join(os.path.dirname(__file__), "cascade_files")

# Haar级联分类器路径
CASCADE_PATHS = {
    "face": os.path.join(CASCADE_DIR, "haarcascade_frontalface_alt.xml"),
    "eye": os.path.join(CASCADE_DIR, "haarcascade_eye.xml"),
    "nose": os.path.join(CASCADE_DIR, "haarcascade_mcs_nose.xml"),
    "mouth": os.path.join(CASCADE_DIR, "haarcascade_mcs_mouth.xml"),
    "left_ear": os.path.join(CASCADE_DIR, "haarcascade_mcs_leftear.xml"),
    "right_ear": os.path.join(CASCADE_DIR, "haarcascade_mcs_rightear.xml"),
}


class FaceDetector:
    """
    人体部件检测器
    基于Haar级联分类器实现人脸及面部器官检测
    """
    
    def __init__(self):
        """初始化：加载所有级联分类器"""
        self.cascades = {}
        for name, path in CASCADE_PATHS.items():
            if os.path.exists(path):
                self.cascades[name] = cv2.CascadeClassifier(path)
                print(f"[INFO] 已加载分类器: {name}")
            else:
                print(f"[WARNING] 未找到分类器: {path}")
    
    def detect_face(self, frame, gray=None):
        """
        全局面部检测
        
        Args:
            frame: BGR彩色图像
            gray: 灰度图像(可选,如果不提供会自动转换)
            
        Returns:
            faces: 检测到的人脸列表 [(x, y, w, h), ...]
        """
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if "face" not in self.cascades:
            return []
        
        # 使用detectMultiScale进行多尺度检测
        faces = self.cascades["face"].detectMultiScale(
            gray,
            scaleFactor=1.1,      # 每次图像尺寸减小的比例
            minNeighbors=5,       # 每个目标至少被检测到的次数
            minSize=(30, 30),     # 目标的最小尺寸
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def detect_eyes(self, frame, face_roi, gray_roi=None):
        """
        眼睛检测 - 仅在人脸ROI的上半部分搜索
        
        Args:
            frame: 原始BGR图像
            face_roi: 人脸区域 (x, y, w, h)
            gray_roi: 人脸区域的灰度图
            
        Returns:
            eyes: 眼睛位置列表(相对于原图坐标) [(x, y, w, h), ...]
        """
        if "eye" not in self.cascades:
            return []
        
        x, y, w, h = face_roi
        
        # 仅在人脸上半部分搜索眼睛(眼睛通常在脸部上半区域)
        upper_half_h = int(h * 0.6)
        
        if gray_roi is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_roi = gray[y:y+upper_half_h, x:x+w]
        else:
            gray_roi = gray_roi[:upper_half_h, :]
        
        eyes_local = self.cascades["eye"].detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # 转换为原图坐标
        eyes = []
        for (ex, ey, ew, eh) in eyes_local:
            eyes.append((x + ex, y + ey, ew, eh))
        
        return eyes
    
    def detect_nose(self, frame, face_roi, gray_roi=None):
        """
        鼻子检测 - 在人脸ROI内搜索
        
        Args:
            frame: 原始BGR图像
            face_roi: 人脸区域 (x, y, w, h)
            gray_roi: 人脸区域的灰度图
            
        Returns:
            noses: 鼻子位置列表(相对于原图坐标)
        """
        if "nose" not in self.cascades:
            return []
        
        x, y, w, h = face_roi
        
        if gray_roi is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_roi = gray[y:y+h, x:x+w]
        
        noses_local = self.cascades["nose"].detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        # 转换为原图坐标
        noses = []
        for (nx, ny, nw, nh) in noses_local:
            noses.append((x + nx, y + ny, nw, nh))
        
        return noses
    
    def detect_mouth(self, frame, face_roi, gray_roi=None):
        """
        嘴巴检测 - 在人脸ROI下半部分搜索
        
        Args:
            frame: 原始BGR图像
            face_roi: 人脸区域 (x, y, w, h)
            gray_roi: 人脸区域的灰度图
            
        Returns:
            mouths: 嘴巴位置列表(相对于原图坐标)
        """
        if "mouth" not in self.cascades:
            return []
        
        x, y, w, h = face_roi
        
        # 嘴巴通常在脸部下半部分
        lower_start = int(h * 0.5)
        
        if gray_roi is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_roi = gray[y+lower_start:y+h, x:x+w]
        else:
            gray_roi = gray_roi[lower_start:, :]
        
        mouths_local = self.cascades["mouth"].detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=10,  # 提高阈值减少误检
            minSize=(25, 15)
        )
        
        # 转换为原图坐标
        mouths = []
        for (mx, my, mw, mh) in mouths_local:
            mouths.append((x + mx, y + lower_start + my, mw, mh))
        
        return mouths
    
    def detect_ears(self, frame, face_roi, gray_roi=None):
        """
        耳朵检测 - 左右耳分别检测（在人脸ROI内）
        
        Args:
            frame: 原始BGR图像
            face_roi: 人脸区域 (x, y, w, h)
            gray_roi: 人脸区域的灰度图
            
        Returns:
            ears: {"left": [...], "right": [...]}
        """
        x, y, w, h = face_roi
        
        if gray_roi is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_roi = gray[y:y+h, x:x+w]
        
        ears = {"left": [], "right": []}
        
        # 左耳(通常在图像右侧)
        if "left_ear" in self.cascades:
            left_ears = self.cascades["left_ear"].detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(15, 25)
            )
            for (ex, ey, ew, eh) in left_ears:
                ears["left"].append((x + ex, y + ey, ew, eh))
        
        # 右耳(通常在图像左侧)
        if "right_ear" in self.cascades:
            right_ears = self.cascades["right_ear"].detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(15, 25)
            )
            for (ex, ey, ew, eh) in right_ears:
                ears["right"].append((x + ex, y + ey, ew, eh))
        
        return ears
    
    def detect_ears_global(self, frame, gray=None):
        """
        全局耳朵检测 - 在整幅图像中搜索耳朵（侧脸场景）
        
        当正脸检测失败时，仍然可以独立检测耳朵
        
        Args:
            frame: 原始BGR图像
            gray: 灰度图像
            
        Returns:
            ears: {"left": [...], "right": [...]}
        """
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 直方图均衡化增强对比度
        gray = cv2.equalizeHist(gray)
        
        ears = {"left": [], "right": []}
        
        # 左耳检测 - 降低阈值提高检测率
        if "left_ear" in self.cascades:
            left_ears = self.cascades["left_ear"].detectMultiScale(
                gray,
                scaleFactor=1.05,    # 更小的缩放比例，更精细的搜索
                minNeighbors=1,      # 降低阈值
                minSize=(15, 20)     # 更小的最小尺寸
            )
            for (ex, ey, ew, eh) in left_ears:
                ears["left"].append((ex, ey, ew, eh))
        
        # 右耳检测 - 降低阈值提高检测率
        if "right_ear" in self.cascades:
            right_ears = self.cascades["right_ear"].detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=1,
                minSize=(15, 20)
            )
            for (ex, ey, ew, eh) in right_ears:
                ears["right"].append((ex, ey, ew, eh))
        
        return ears
    
    def detect_pupil(self, eye_roi):
        """
        瞳孔检测 - 基于形状分析(不使用分类器)
        
        算法流程:
        1. 图像反相(使黑色瞳孔变为高亮)
        2. 二值化阈值处理
        3. 轮廓查找
        4. 几何筛选(面积、圆形度)
        
        Args:
            eye_roi: 眼睛区域的BGR图像
            
        Returns:
            pupil: 瞳孔中心和半径 (cx, cy, radius) 或 None
        """
        if eye_roi is None or eye_roi.size == 0:
            return None
        
        # 检查图像尺寸是否太小
        if eye_roi.shape[0] < 10 or eye_roi.shape[1] < 10:
            return None
        
        # 转为灰度
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi.copy()
        
        # 高斯模糊去噪（减小核大小以保留更多细节）
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 图像反相 - 使黑色瞳孔变为白色高亮
        inverted = cv2.bitwise_not(blurred)
        
        # 使用自适应阈值或降低固定阈值
        # 方法1：降低阈值（原来200太高）
        _, thresh = cv2.threshold(inverted, 120, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 几何筛选 - 寻找最接近圆形的轮廓
        best_pupil = None
        best_score = 0
        
        roi_h, roi_w = gray.shape[:2]
        roi_area = roi_h * roi_w
        roi_center_x = roi_w // 2
        roi_center_y = roi_h // 2
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 放宽面积约束
            if area < roi_area * 0.005 or area > roi_area * 0.6:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # 圆形度计算
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # 放宽圆形度约束（从0.3降到0.2）
            if circularity < 0.2:
                continue
            
            # 计算轮廓中心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 优先选择靠近眼睛中心的轮廓
            dist_to_center = np.sqrt((cx - roi_center_x)**2 + (cy - roi_center_y)**2)
            max_dist = np.sqrt(roi_center_x**2 + roi_center_y**2)
            center_score = 1 - (dist_to_center / max_dist) if max_dist > 0 else 0
            
            # 综合评分：圆形度 + 中心距离
            score = circularity * 0.5 + center_score * 0.5
            
            if score > best_score:
                best_score = score
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                best_pupil = (int(cx), int(cy), max(2, int(radius)))
        
        return best_pupil


def draw_detections(frame, faces, detector, draw_config=None):
    """
    在图像上绘制所有检测结果
    
    Args:
        frame: BGR图像
        faces: 检测到的人脸列表
        detector: FaceDetector实例
        draw_config: 绘制配置字典
        
    Returns:
        frame: 绘制后的图像
        stats: 检测统计 {"face": n, "eyes": n, "nose": n, "mouth": n, "left_ear": n, "right_ear": n, "pupil": n}
    """
    if draw_config is None:
        draw_config = {
            "face": True,
            "eyes": True,
            "nose": True,
            "mouth": True,
            "ears": True,
            "pupil": True
        }
    
    # 检测统计
    stats = {
        "face": len(faces),
        "eyes": 0,
        "nose": 0,
        "mouth": 0,
        "left_ear": 0,
        "right_ear": 0,
        "pupil": 0
    }
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for (x, y, w, h) in faces:
        # 绘制人脸框
        if draw_config.get("face", True):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        face_roi = (x, y, w, h)
        gray_face = gray[y:y+h, x:x+w]
        
        # 检测并绘制眼睛
        if draw_config.get("eyes", True):
            eyes = detector.detect_eyes(frame, face_roi, gray_face.copy())
            stats["eyes"] += len(eyes)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                cv2.putText(frame, "Eye", (ex, ey-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # 瞳孔检测
                if draw_config.get("pupil", True):
                    eye_img = frame[ey:ey+eh, ex:ex+ew]
                    pupil = detector.detect_pupil(eye_img)
                    if pupil:
                        stats["pupil"] += 1
                        px, py, pr = pupil
                        # 转换到原图坐标
                        cv2.circle(frame, (ex+px, ey+py), pr, (255, 0, 255), 2)
                        cv2.circle(frame, (ex+px, ey+py), 2, (255, 0, 255), -1)
        
        # 检测并绘制鼻子
        if draw_config.get("nose", True):
            noses = detector.detect_nose(frame, face_roi, gray_face.copy())
            stats["nose"] += len(noses)
            for (nx, ny, nw, nh) in noses:
                cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (0, 255, 255), 2)
                cv2.putText(frame, "Nose", (nx, ny-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 检测并绘制嘴巴
        if draw_config.get("mouth", True):
            mouths = detector.detect_mouth(frame, face_roi, gray_face.copy())
            stats["mouth"] += len(mouths)
            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                cv2.putText(frame, "Mouth", (mx, my-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 检测并绘制耳朵
        if draw_config.get("ears", True):
            ears = detector.detect_ears(frame, face_roi, gray_face.copy())
            stats["left_ear"] += len(ears["left"])
            stats["right_ear"] += len(ears["right"])
            for ear in ears["left"]:
                ex, ey, ew, eh = ear
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 128, 0), 2)
                cv2.putText(frame, "L-Ear", (ex, ey-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
            for ear in ears["right"]:
                ex, ey, ew, eh = ear
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 128, 255), 2)
                cv2.putText(frame, "R-Ear", (ex, ey-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1)
    
    # 当没有检测到人脸时，进行全局耳朵检测（侧脸场景）
    if len(faces) == 0 and draw_config.get("ears", True):
        ears = detector.detect_ears_global(frame, gray)
        stats["left_ear"] += len(ears["left"])
        stats["right_ear"] += len(ears["right"])
        for ear in ears["left"]:
            ex, ey, ew, eh = ear
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 128, 0), 2)
            cv2.putText(frame, "L-Ear", (ex, ey-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
        for ear in ears["right"]:
            ex, ey, ew, eh = ear
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 128, 255), 2)
            cv2.putText(frame, "R-Ear", (ex, ey-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 255), 1)
    
    return frame, stats


def draw_status_panel(frame, stats):
    """
    在图像上绘制检测状态面板
    
    Args:
        frame: BGR图像
        stats: 检测统计字典
        
    Returns:
        frame: 绘制后的图像
    """
    h, w = frame.shape[:2]
    
    # 状态面板背景
    panel_h = 180
    overlay = frame.copy()
    cv2.rectangle(overlay, (w-200, 0), (w, panel_h), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # 标题
    cv2.putText(frame, "Detection Status", (w-190, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(frame, (w-195, 35), (w-5, 35), (255, 255, 255), 1)
    
    # 各部位状态
    status_items = [
        ("Face", stats["face"], (255, 0, 0)),
        ("Eyes", stats["eyes"], (0, 255, 0)),
        ("Nose", stats["nose"], (0, 255, 255)),
        ("Mouth", stats["mouth"], (0, 0, 255)),
        ("L-Ear", stats["left_ear"], (255, 128, 0)),
        ("R-Ear", stats["right_ear"], (0, 128, 255)),
        ("Pupil", stats["pupil"], (255, 0, 255)),
    ]
    
    y_offset = 55
    for name, count, color in status_items:
        # 状态指示器 (绿点=检测到, 红点=未检测到)
        indicator_color = (0, 255, 0) if count > 0 else (0, 0, 150)
        cv2.circle(frame, (w-185, y_offset), 5, indicator_color, -1)
        
        # 部位名称和数量
        text = f"{name}: {count}"
        cv2.putText(frame, text, (w-170, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        y_offset += 20
    
    return frame


def test_with_image(image_path):
    """使用静态图片测试检测功能"""
    print(f"\n[INFO] 使用图片测试模式: {image_path}")
    
    detector = FaceDetector()
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_face(frame, gray)
    
    print(f"[INFO] 检测到 {len(faces)} 张人脸")
    
    frame, stats = draw_detections(frame, faces, detector)
    frame = draw_status_panel(frame, stats)
    
    # 打印检测统计
    print("\n[检测统计]")
    print(f"  Face: {stats['face']}")
    print(f"  Eyes: {stats['eyes']}")
    print(f"  Nose: {stats['nose']}")
    print(f"  Mouth: {stats['mouth']}")
    print(f"  L-Ear: {stats['left_ear']}")
    print(f"  R-Ear: {stats['right_ear']}")
    print(f"  Pupil: {stats['pupil']}")
    
    # 显示结果
    cv2.imshow("Face Detection - Image Test", frame)
    print("[INFO] 按任意键保存并退出...")
    cv2.waitKey(0)
    
    # 保存结果
    output_path = image_path.replace(".", "_result.")
    cv2.imwrite(output_path, frame)
    print(f"[INFO] 结果已保存: {output_path}")
    
    cv2.destroyAllWindows()


def main():
    """主函数 - 实时摄像头检测演示"""
    print("=" * 50)
    print("数字图像处理作业2 - 人体部件检测系统(基线)")
    print("=" * 50)
    
    # 初始化检测器
    detector = FaceDetector()
    
    # 尝试使用DirectShow后端打开摄像头(Windows专用,更稳定)
    print("\n[INFO] 尝试使用DirectShow后端打开摄像头...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("[WARNING] DirectShow失败,尝试默认后端...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头!")
        print("[TIP] 请检查摄像头是否被其他程序占用")
        print("[TIP] 或使用图片测试模式: python face_detector.py --image <图片路径>")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n[INFO] 摄像头已开启")
    print("[INFO] 按 'q' 退出程序")
    print("[INFO] 按 's' 保存当前帧截图")
    
    frame_count = 0
    fail_count = 0
    max_fails = 10
    
    while True:
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            print(f"[WARNING] 读取帧失败 ({fail_count}/{max_fails})")
            if fail_count >= max_fails:
                print("[ERROR] 连续读取失败,退出程序!")
                break
            continue
        
        fail_count = 0  # 重置失败计数
        
        # 水平翻转(镜像效果更自然)
        frame = cv2.flip(frame, 1)
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = detector.detect_face(frame, gray)
        
        # 绘制所有检测结果
        frame, stats = draw_detections(frame, faces, detector)
        
        # 绘制状态面板
        frame = draw_status_panel(frame, stats)
        
        # 显示结果
        cv2.imshow("Face Detection - Baseline", frame)
        
        # 键盘事件处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] 截图已保存: {filename}")
            frame_count += 1
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] 程序已退出")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2 and sys.argv[1] == "--image":
        # 图片测试模式
        test_with_image(sys.argv[2])
    else:
        # 摄像头模式
        main()
