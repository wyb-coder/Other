"""
数字图像处理作业2 - 创新模块3：隐私马赛克盾
Privacy Mosaic Shield

核心技术：ROI降采样与最近邻插值
实现人脸隐私保护，模拟新闻报道中的马赛克效果
"""

import cv2
import numpy as np
import os

# 导入基线检测器
from baseline import FaceDetector, draw_status_panel


class PrivacyMosaic:
    """
    隐私马赛克处理器
    通过降采样和最近邻插值实现马赛克效果
    """
    
    def __init__(self, default_level=15):
        """
        初始化马赛克处理器
        
        Args:
            default_level: 默认马赛克等级（像素块大小）
        """
        self.level = default_level
        self.min_level = 5
        self.max_level = 30
    
    def increase_level(self):
        """增加马赛克强度（更模糊）"""
        if self.level < self.max_level:
            self.level += 2
            print(f"[INFO] 马赛克等级: {self.level}")
    
    def decrease_level(self):
        """降低马赛克强度（更清晰）"""
        if self.level > self.min_level:
            self.level -= 2
            print(f"[INFO] 马赛克等级: {self.level}")
    
    def apply_mosaic(self, frame, roi_rect):
        """
        对指定区域应用马赛克效果
        
        算法原理：
        1. 降采样：将ROI缩小为原来的 1/level
        2. 升采样：使用最近邻插值放大回原尺寸
        3. 结果：产生明显的方块效果
        
        Args:
            frame: 原始BGR图像
            roi_rect: 需要马赛克的区域 (x, y, w, h)
            
        Returns:
            frame: 处理后的图像
        """
        x, y, w, h = roi_rect
        
        # 边界检查
        frame_h, frame_w = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        
        if w <= 0 or h <= 0:
            return frame
        
        try:
            # 1. 提取ROI
            roi = frame[y:y+h, x:x+w]
            
            # 2. 计算缩小后的尺寸
            small_w = max(1, w // self.level)
            small_h = max(1, h // self.level)
            
            # 3. 降采样（使用INTER_LINEAR平滑采样）
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            
            # 4. 升采样（使用INTER_NEAREST最近邻插值，产生方块效果）
            mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 5. 放回原图
            frame[y:y+h, x:x+w] = mosaic
            
        except Exception as e:
            print(f"[ERROR] 马赛克应用失败: {e}")
        
        return frame
    
    def apply_mosaic_face(self, frame, face_rect, expand_ratio=0.1):
        """
        对人脸区域应用马赛克（带扩展）
        
        Args:
            frame: 原始BGR图像
            face_rect: 人脸区域 (x, y, w, h)
            expand_ratio: 扩展比例（向外扩展以确保完全覆盖）
            
        Returns:
            frame: 处理后的图像
        """
        x, y, w, h = face_rect
        
        # 扩展区域
        expand_w = int(w * expand_ratio)
        expand_h = int(h * expand_ratio)
        
        new_x = x - expand_w
        new_y = y - expand_h
        new_w = w + 2 * expand_w
        new_h = h + 2 * expand_h
        
        return self.apply_mosaic(frame, (new_x, new_y, new_w, new_h))
    
    def apply_mosaic_eyes(self, frame, eye_rects, expand_ratio=0.3):
        """
        仅对眼睛区域应用马赛克（常用于保护隐私但保留部分面部特征）
        
        Args:
            frame: 原始BGR图像
            eye_rects: 眼睛区域列表
            expand_ratio: 扩展比例
            
        Returns:
            frame: 处理后的图像
        """
        for (ex, ey, ew, eh) in eye_rects:
            # 扩展眼睛区域
            expand_w = int(ew * expand_ratio)
            expand_h = int(eh * expand_ratio)
            
            new_x = ex - expand_w
            new_y = ey - expand_h
            new_w = ew + 2 * expand_w
            new_h = eh + 2 * expand_h
            
            frame = self.apply_mosaic(frame, (new_x, new_y, new_w, new_h))
        
        return frame


def draw_detections_with_mosaic(frame, faces, detector, mosaic_processor, 
                                 mosaic_mode="off"):
    """
    绘制检测结果，并可选地应用马赛克
    
    Args:
        frame: BGR图像
        faces: 检测到的人脸列表
        detector: FaceDetector实例
        mosaic_processor: PrivacyMosaic实例
        mosaic_mode: "off" / "face" / "eyes"
        
    Returns:
        frame: 处理后的图像
        stats: 检测统计
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
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for (x, y, w, h) in faces:
        face_roi = (x, y, w, h)
        gray_face = gray[y:y+h, x:x+w]
        
        # 检测眼睛（用于eyes模式和统计）
        eyes = detector.detect_eyes(frame, face_roi, gray_face.copy())
        stats["eyes"] += len(eyes)
        
        # 应用马赛克
        if mosaic_mode == "face":
            # 整脸马赛克
            frame = mosaic_processor.apply_mosaic_face(frame, face_roi)
        elif mosaic_mode == "eyes":
            # 仅眼睛马赛克
            frame = mosaic_processor.apply_mosaic_eyes(frame, eyes)
            # 绘制人脸框（眼睛模式下显示人脸框）
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            # 正常模式：绘制所有检测框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # 绘制眼睛
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # 其他器官检测（用于统计）
        noses = detector.detect_nose(frame, face_roi, gray_face.copy())
        stats["nose"] += len(noses)
        
        mouths = detector.detect_mouth(frame, face_roi, gray_face.copy())
        stats["mouth"] += len(mouths)
        
        ears = detector.detect_ears(frame, face_roi, gray_face.copy())
        stats["left_ear"] += len(ears["left"])
        stats["right_ear"] += len(ears["right"])
    
    return frame, stats


def main():
    """主函数 - 隐私马赛克演示"""
    print("=" * 50)
    print("数字图像处理作业2 - 创新模块3：隐私马赛克盾")
    print("=" * 50)
    
    # 初始化
    detector = FaceDetector()
    mosaic_processor = PrivacyMosaic(default_level=15)
    
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
    print("  [p] - 切换马赛克模式 (关闭 → 整脸 → 仅眼睛)")
    print("  [+] - 增加马赛克强度")
    print("  [-] - 降低马赛克强度")
    print("  [s] - 保存截图")
    print("  [q] - 退出程序")
    
    mosaic_modes = ["off", "face", "eyes"]
    mosaic_mode_names = {"off": "关闭", "face": "整脸", "eyes": "仅眼睛"}
    current_mode_idx = 0
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
        
        # 绘制检测结果（可选马赛克）
        current_mode = mosaic_modes[current_mode_idx]
        frame, stats = draw_detections_with_mosaic(
            frame, faces, detector, mosaic_processor, current_mode
        )
        
        # 绘制状态面板
        frame = draw_status_panel(frame, stats)
        
        # 绘制模式提示
        mode_text = f"PRIVACY: {mosaic_mode_names[current_mode]}"
        mode_color = (0, 255, 0) if current_mode != "off" else (0, 0, 255)
        cv2.putText(frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        if current_mode != "off":
            level_text = f"Level: {mosaic_processor.level}"
            cv2.putText(frame, level_text, (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示结果
        cv2.imshow("Privacy Mosaic - Innovation Module 3", frame)
        
        # 键盘事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            current_mode_idx = (current_mode_idx + 1) % len(mosaic_modes)
            print(f"[INFO] 马赛克模式: {mosaic_mode_names[mosaic_modes[current_mode_idx]]}")
        elif key == ord('+') or key == ord('='):
            mosaic_processor.increase_level()
        elif key == ord('-') or key == ord('_'):
            mosaic_processor.decrease_level()
        elif key == ord('s'):
            filename = f"privacy_screenshot_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] 截图已保存: {filename}")
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] 程序已退出")


if __name__ == "__main__":
    main()
