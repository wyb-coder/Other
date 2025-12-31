"""
数字图像处理作业2 - 创新模块2：几何约束过滤器
Geometric Constraints Filter

核心技术：逻辑校验与坐标变换
通过解剖学常识过滤误检，提升检测鲁棒性
"""

import cv2
import numpy as np
import os

# 导入基线检测器
from baseline import FaceDetector, draw_status_panel

# ======================== 配置 ========================
# 几何约束阈值（相对于人脸高度的比例）
CONSTRAINTS = {
    "eye_max_y": 0.55,      # 眼睛不能超过人脸下半部分55%
    "eye_min_y": 0.10,      # 眼睛不能在人脸最上方10%
    "nose_min_y": 0.25,     # 鼻子至少在人脸25%以下
    "nose_max_y": 0.75,     # 鼻子不能超过人脸75%
    "mouth_min_y": 0.50,    # 嘴巴至少在人脸50%以下
    "mouth_max_y": 0.95,    # 嘴巴不能超过人脸95%
}


class GeometricFilter:
    """
    几何约束过滤器
    基于解剖学规则过滤误检
    """
    
    def __init__(self, constraints=None):
        """
        初始化过滤器
        
        Args:
            constraints: 约束参数字典
        """
        self.constraints = constraints or CONSTRAINTS
        self.filter_stats = {
            "eyes_filtered": 0,
            "nose_filtered": 0,
            "mouth_filtered": 0,
            "total_filtered": 0
        }
    
    def reset_stats(self):
        """重置过滤统计"""
        for key in self.filter_stats:
            self.filter_stats[key] = 0
    
    def check_eye(self, eye_rect, face_rect):
        """
        检查眼睛位置是否合理
        
        规则：
        1. 眼睛应该在人脸上半部分（y < 55% face_h）
        2. 眼睛不应该在人脸最顶端（y > 10% face_h）
        
        Args:
            eye_rect: 眼睛相对于人脸的坐标 (x, y, w, h)
            face_rect: 人脸区域 (x, y, w, h)
            
        Returns:
            bool: True表示位置合理，False表示应过滤
        """
        _, ey, _, eh = eye_rect
        _, _, _, fh = face_rect
        
        # 计算眼睛中心y相对于人脸的比例
        eye_center_y = (ey + eh / 2) / fh
        
        # 检查约束
        if eye_center_y > self.constraints["eye_max_y"]:
            return False
        if eye_center_y < self.constraints["eye_min_y"]:
            return False
        
        return True
    
    def check_nose(self, nose_rect, face_rect, eye_rects=None):
        """
        检查鼻子位置是否合理
        
        规则：
        1. 鼻子应该在人脸中部（25% < y < 75%）
        2. 如果有眼睛检测，鼻子应该在眼睛下方
        
        Args:
            nose_rect: 鼻子相对于人脸的坐标
            face_rect: 人脸区域
            eye_rects: 眼睛位置列表（可选）
            
        Returns:
            bool: True表示位置合理
        """
        _, ny, _, nh = nose_rect
        _, _, _, fh = face_rect
        
        # 计算鼻子中心y相对于人脸的比例
        nose_center_y = (ny + nh / 2) / fh
        
        # 基本位置约束
        if nose_center_y < self.constraints["nose_min_y"]:
            return False
        if nose_center_y > self.constraints["nose_max_y"]:
            return False
        
        # 如果有眼睛检测结果，鼻子应该在眼睛下方
        if eye_rects and len(eye_rects) > 0:
            min_eye_bottom = min(ey + eh for (_, ey, _, eh) in eye_rects)
            if ny < min_eye_bottom * 0.8:  # 鼻子顶部应该低于眼睛底部
                return False
        
        return True
    
    def check_mouth(self, mouth_rect, face_rect, nose_rects=None):
        """
        检查嘴巴位置是否合理
        
        规则：
        1. 嘴巴应该在人脸下半部分（y > 50%）
        2. 如果有鼻子检测，嘴巴应该在鼻子下方
        
        Args:
            mouth_rect: 嘴巴相对于人脸的坐标
            face_rect: 人脸区域
            nose_rects: 鼻子位置列表（可选）
            
        Returns:
            bool: True表示位置合理
        """
        _, my, _, mh = mouth_rect
        _, _, _, fh = face_rect
        
        # 计算嘴巴中心y相对于人脸的比例
        mouth_center_y = (my + mh / 2) / fh
        
        # 基本位置约束
        if mouth_center_y < self.constraints["mouth_min_y"]:
            return False
        if mouth_center_y > self.constraints["mouth_max_y"]:
            return False
        
        # 如果有鼻子检测结果，嘴巴应该在鼻子下方
        if nose_rects and len(nose_rects) > 0:
            min_nose_bottom = min(ny + nh for (_, ny, _, nh) in nose_rects)
            if my < min_nose_bottom * 0.9:  # 嘴巴顶部应该低于鼻子底部
                return False
        
        return True
    
    def filter_detections(self, eyes, noses, mouths, face_rect):
        """
        过滤所有检测结果
        
        Args:
            eyes: 眼睛检测列表（相对于人脸ROI的坐标）
            noses: 鼻子检测列表
            mouths: 嘴巴检测列表
            face_rect: 人脸区域
            
        Returns:
            filtered_eyes, filtered_noses, filtered_mouths: 过滤后的结果
        """
        fx, fy, fw, fh = face_rect
        
        # 过滤眼睛
        filtered_eyes = []
        for eye in eyes:
            ex, ey, ew, eh = eye
            # 转换为相对坐标
            rel_eye = (ex - fx, ey - fy, ew, eh)
            if self.check_eye(rel_eye, (0, 0, fw, fh)):
                filtered_eyes.append(eye)
            else:
                self.filter_stats["eyes_filtered"] += 1
                self.filter_stats["total_filtered"] += 1
        
        # 过滤鼻子（使用过滤后的眼睛作为参考）
        filtered_noses = []
        rel_filtered_eyes = [(ex - fx, ey - fy, ew, eh) for (ex, ey, ew, eh) in filtered_eyes]
        for nose in noses:
            nx, ny, nw, nh = nose
            rel_nose = (nx - fx, ny - fy, nw, nh)
            if self.check_nose(rel_nose, (0, 0, fw, fh), rel_filtered_eyes):
                filtered_noses.append(nose)
            else:
                self.filter_stats["nose_filtered"] += 1
                self.filter_stats["total_filtered"] += 1
        
        # 过滤嘴巴（使用过滤后的鼻子作为参考）
        filtered_mouths = []
        rel_filtered_noses = [(nx - fx, ny - fy, nw, nh) for (nx, ny, nw, nh) in filtered_noses]
        for mouth in mouths:
            mx, my, mw, mh = mouth
            rel_mouth = (mx - fx, my - fy, mw, mh)
            if self.check_mouth(rel_mouth, (0, 0, fw, fh), rel_filtered_noses):
                filtered_mouths.append(mouth)
            else:
                self.filter_stats["mouth_filtered"] += 1
                self.filter_stats["total_filtered"] += 1
        
        return filtered_eyes, filtered_noses, filtered_mouths


def draw_detections_with_geometry(frame, faces, detector, geo_filter, geometry_enabled=True):
    """
    绘制检测结果，可选地应用几何约束
    
    Args:
        frame: BGR图像
        faces: 检测到的人脸列表
        detector: FaceDetector实例
        geo_filter: GeometricFilter实例
        geometry_enabled: 是否启用几何约束
        
    Returns:
        frame: 处理后的图像
        stats: 检测统计
        filter_stats: 过滤统计
    """
    stats = {
        "face": len(faces),
        "eyes": 0,
        "nose": 0,
        "mouth": 0,
        "left_ear": 0,
        "right_ear": 0,
        "pupil": 0
    }
    
    geo_filter.reset_stats()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for (x, y, w, h) in faces:
        # 绘制人脸框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        face_roi = (x, y, w, h)
        gray_face = gray[y:y+h, x:x+w]
        
        # 检测器官
        eyes = detector.detect_eyes(frame, face_roi, gray_face.copy())
        noses = detector.detect_nose(frame, face_roi, gray_face.copy())
        mouths = detector.detect_mouth(frame, face_roi, gray_face.copy())
        
        # 应用几何约束过滤
        if geometry_enabled:
            eyes, noses, mouths = geo_filter.filter_detections(
                eyes, noses, mouths, face_roi
            )
        
        # 统计
        stats["eyes"] += len(eyes)
        stats["nose"] += len(noses)
        stats["mouth"] += len(mouths)
        
        # 绘制眼睛
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(frame, "Eye", (ex, ey-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # 瞳孔检测
            eye_img = frame[ey:ey+eh, ex:ex+ew]
            pupil = detector.detect_pupil(eye_img)
            if pupil:
                stats["pupil"] += 1
                px, py, pr = pupil
                cv2.circle(frame, (ex+px, ey+py), pr, (255, 0, 255), 2)
                cv2.circle(frame, (ex+px, ey+py), 2, (255, 0, 255), -1)
        
        # 绘制鼻子
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(frame, (nx, ny), (nx+nw, ny+nh), (0, 255, 255), 2)
            cv2.putText(frame, "Nose", (nx, ny-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 绘制嘴巴
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
            cv2.putText(frame, "Mouth", (mx, my-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 绘制耳朵（不受几何约束影响）
        ears = detector.detect_ears(frame, face_roi, gray_face.copy())
        stats["left_ear"] += len(ears["left"])
        stats["right_ear"] += len(ears["right"])
        for ear in ears["left"]:
            ex, ey, ew, eh = ear
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 128, 0), 2)
        for ear in ears["right"]:
            ex, ey, ew, eh = ear
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 128, 255), 2)
    
    return frame, stats, geo_filter.filter_stats


def draw_filter_stats(frame, filter_stats, geometry_enabled):
    """绘制过滤统计信息"""
    y_start = 200
    
    # 模式状态
    mode_text = "GEOMETRY: ON" if geometry_enabled else "GEOMETRY: OFF"
    mode_color = (0, 255, 0) if geometry_enabled else (0, 0, 255)
    cv2.putText(frame, mode_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    if geometry_enabled:
        # 过滤统计
        cv2.putText(frame, f"Filtered: {filter_stats['total_filtered']}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(frame, f"  Eyes: {filter_stats['eyes_filtered']}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"  Nose: {filter_stats['nose_filtered']}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, f"  Mouth: {filter_stats['mouth_filtered']}", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return frame


def main():
    """主函数 - 几何约束演示"""
    print("=" * 50)
    print("数字图像处理作业2 - 创新模块2：几何约束过滤器")
    print("=" * 50)
    
    # 初始化
    detector = FaceDetector()
    geo_filter = GeometricFilter()
    
    # 打开摄像头
    print("\n[INFO] 尝试打开摄像头...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n[INFO] 摄像头已开启")
    print("[INFO] 按键说明：")
    print("  [g] - 开关几何约束")
    print("  [s] - 保存截图")
    print("  [q] - 退出程序")
    
    geometry_enabled = True
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 水平翻转
        frame = cv2.flip(frame, 1)
        
        # 检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_face(frame, gray)
        
        # 绘制检测结果（可选几何约束）
        frame, stats, filter_stats = draw_detections_with_geometry(
            frame, faces, detector, geo_filter, geometry_enabled
        )
        
        # 绘制状态面板
        frame = draw_status_panel(frame, stats)
        
        # 绘制过滤统计
        frame = draw_filter_stats(frame, filter_stats, geometry_enabled)
        
        # 显示结果
        cv2.imshow("Geometric Constraint - Innovation Module 2", frame)
        
        # 键盘事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            geometry_enabled = not geometry_enabled
            print(f"[INFO] 几何约束: {'开启' if geometry_enabled else '关闭'}")
        elif key == ord('s'):
            filename = f"geometry_screenshot_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] 截图已保存: {filename}")
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] 程序已退出")


if __name__ == "__main__":
    main()
