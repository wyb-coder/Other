"""
数字图像处理作业2 - 创新模块1：AR面具系统
"真我假面" (AR Masquerade)

核心技术：掩膜生成与位运算(Bitwise Operations)
实现面具纹理与人脸背景的像素级融合
"""

import cv2
import numpy as np
import os

# 导入基线检测器
from baseline import FaceDetector, draw_status_panel

# ======================== 配置 ========================
MASK_DIR = os.path.join(os.path.dirname(__file__), "Face")


class ARMaskApplier:
    """
    AR面具应用器
    实现面具与人脸的无缝融合
    """
    
    def __init__(self):
        """初始化：加载所有可用面具"""
        self.masks = []
        self.mask_names = []
        self.current_mask_idx = 0
        
        # 加载Face目录下所有图片作为面具
        if os.path.exists(MASK_DIR):
            for filename in sorted(os.listdir(MASK_DIR)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    mask_path = os.path.join(MASK_DIR, filename)
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask_img is not None:
                        self.masks.append(mask_img)
                        self.mask_names.append(filename)
                        print(f"[INFO] 已加载面具: {filename}")
        
        if len(self.masks) == 0:
            print("[WARNING] 未找到面具图片，请在 Face/ 目录下放置面具图片")
    
    def switch_mask(self):
        """切换到下一个面具"""
        if len(self.masks) > 0:
            self.current_mask_idx = (self.current_mask_idx + 1) % len(self.masks)
            print(f"[INFO] 切换面具: {self.mask_names[self.current_mask_idx]}")
    
    def get_current_mask(self):
        """获取当前面具"""
        if len(self.masks) > 0:
            return self.masks[self.current_mask_idx]
        return None
    
    def apply_mask(self, frame, face_roi):
        """
        将面具应用到人脸区域
        
        技术实现：
        1. 提取人脸ROI
        2. 将面具缩放到人脸大小
        3. 生成二值掩膜
        4. 使用位运算融合
        
        Args:
            frame: 原始BGR图像
            face_roi: 人脸区域 (x, y, w, h)
            
        Returns:
            frame: 处理后的图像
        """
        mask_img = self.get_current_mask()
        if mask_img is None:
            return frame
        
        x, y, w, h = face_roi
        
        # 确保坐标有效
        if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
            return frame
        
        try:
            # 1. 将面具缩放到人脸大小
            mask_resized = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_AREA)
            
            # 2. 检查面具是否有Alpha通道
            if mask_resized.shape[2] == 4:
                # 有Alpha通道的情况（PNG透明图）
                alpha = mask_resized[:, :, 3] / 255.0
                mask_bgr = mask_resized[:, :, :3]
            else:
                # 无Alpha通道，使用颜色阈值生成掩膜
                # 假设白色或接近白色为背景
                mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(mask_gray, 240, 255, cv2.THRESH_BINARY_INV)
                alpha = binary_mask / 255.0
                mask_bgr = mask_resized
            
            # 3. 获取人脸ROI
            roi = frame[y:y+h, x:x+w]
            
            # 4. Alpha混合
            # 公式: output = alpha * mask + (1-alpha) * background
            for c in range(3):
                roi[:, :, c] = (alpha * mask_bgr[:, :, c] + 
                               (1 - alpha) * roi[:, :, c]).astype(np.uint8)
            
            # 5. 将处理后的ROI放回原图
            frame[y:y+h, x:x+w] = roi
            
        except Exception as e:
            print(f"[ERROR] 面具应用失败: {e}")
        
        return frame
    
    def apply_mask_bitwise(self, frame, face_roi, scale=1.15, y_offset_ratio=-0.1):
        """
        使用位运算实现面具融合（教材方法）
        
        技术实现：
        1. 生成二值掩膜(mask)和逆掩膜(mask_inv)
        2. 使用bitwise_and抠出背景
        3. 使用bitwise_and提取面具
        4. 使用add融合
        
        Args:
            frame: 原始BGR图像
            face_roi: 人脸区域 (x, y, w, h)
            scale: 面具缩放比例（相对于人脸大小），默认1.3倍
            y_offset_ratio: y方向偏移比例（负值向上），默认-0.2
            
        Returns:
            frame: 处理后的图像
        """
        mask_img = self.get_current_mask()
        if mask_img is None:
            return frame
        
        x, y, w, h = face_roi
        
        # 计算放大后的面具尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 计算新的位置（居中并向上偏移）
        new_x = x - (new_w - w) // 2
        new_y = y - (new_h - h) // 2 + int(h * y_offset_ratio)
        
        # 边界检查和裁剪
        frame_h, frame_w = frame.shape[:2]
        
        # 计算有效区域
        src_x1 = max(0, -new_x)
        src_y1 = max(0, -new_y)
        src_x2 = min(new_w, frame_w - new_x)
        src_y2 = min(new_h, frame_h - new_y)
        
        dst_x1 = max(0, new_x)
        dst_y1 = max(0, new_y)
        dst_x2 = min(frame_w, new_x + new_w)
        dst_y2 = min(frame_h, new_y + new_h)
        
        # 检查有效性
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return frame
        
        try:
            # 1. 将面具缩放到放大后的尺寸
            mask_resized = cv2.resize(mask_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # 2. 裁剪面具到有效区域
            mask_cropped = mask_resized[src_y1:src_y2, src_x1:src_x2]
            
            # 如果有Alpha通道，转为3通道
            if mask_cropped.shape[2] == 4:
                mask_bgr = mask_cropped[:, :, :3]
                alpha_channel = mask_cropped[:, :, 3]
            else:
                mask_bgr = mask_cropped
                # 创建掩膜：非白色区域为前景
                mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                _, alpha_channel = cv2.threshold(mask_gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # 3. 生成二值掩膜
            mask_binary = alpha_channel
            mask_inv = cv2.bitwise_not(mask_binary)
            
            # 4. 获取目标ROI
            roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
            
            # 确保尺寸匹配
            if roi.shape[:2] != mask_bgr.shape[:2]:
                return frame
            
            # 5. 使用bitwise_and在ROI中抠出面具形状的黑色区域
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            
            # 6. 使用bitwise_and提取面具的前景部分
            mask_fg = cv2.bitwise_and(mask_bgr, mask_bgr, mask=mask_binary)
            
            # 7. 使用add融合背景和前景
            result = cv2.add(roi_bg, mask_fg)
            
            # 8. 放回原图
            frame[dst_y1:dst_y2, dst_x1:dst_x2] = result
            
        except Exception as e:
            print(f"[ERROR] 面具应用失败: {e}")
        
        return frame


def draw_detections_with_mask(frame, faces, detector, mask_applier, mask_enabled=True):
    """
    绘制检测结果，并可选地应用面具
    
    Args:
        frame: BGR图像
        faces: 检测到的人脸列表
        detector: FaceDetector实例
        mask_applier: ARMaskApplier实例
        mask_enabled: 是否启用面具
        
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
        
        # 应用面具
        if mask_enabled and mask_applier is not None:
            frame = mask_applier.apply_mask_bitwise(frame, face_roi)
        else:
            # 不应用面具时绘制人脸框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 检测其他器官（即使戴了面具也可以检测，用于统计）
        gray_face = gray[y:y+h, x:x+w]
        
        eyes = detector.detect_eyes(frame, face_roi, gray_face.copy())
        stats["eyes"] += len(eyes)
        
        noses = detector.detect_nose(frame, face_roi, gray_face.copy())
        stats["nose"] += len(noses)
        
        mouths = detector.detect_mouth(frame, face_roi, gray_face.copy())
        stats["mouth"] += len(mouths)
        
        ears = detector.detect_ears(frame, face_roi, gray_face.copy())
        stats["left_ear"] += len(ears["left"])
        stats["right_ear"] += len(ears["right"])
    
    return frame, stats


def main():
    """主函数 - AR面具演示"""
    print("=" * 50)
    print("数字图像处理作业2 - 创新模块1：AR面具系统")
    print("=" * 50)
    
    # 初始化检测器和面具应用器
    detector = FaceDetector()
    mask_applier = ARMaskApplier()
    
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
    print("  [m] - 开关面具模式")
    print("  [n] - 切换下一个面具")
    print("  [s] - 保存截图")
    print("  [q] - 退出程序")
    
    mask_enabled = True
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
        
        # 绘制检测结果（可选面具）
        frame, stats = draw_detections_with_mask(
            frame, faces, detector, mask_applier, mask_enabled
        )
        
        # 绘制状态面板
        frame = draw_status_panel(frame, stats)
        
        # 绘制模式提示
        mode_text = "MASK: ON" if mask_enabled else "MASK: OFF"
        mode_color = (0, 255, 0) if mask_enabled else (0, 0, 255)
        cv2.putText(frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        if mask_enabled and len(mask_applier.mask_names) > 0:
            mask_text = f"Current: {mask_applier.mask_names[mask_applier.current_mask_idx]}"
            cv2.putText(frame, mask_text, (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示结果
        cv2.imshow("AR Mask - Innovation Module 1", frame)
        
        # 键盘事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mask_enabled = not mask_enabled
            print(f"[INFO] 面具模式: {'开启' if mask_enabled else '关闭'}")
        elif key == ord('n'):
            mask_applier.switch_mask()
        elif key == ord('s'):
            filename = f"ar_mask_screenshot_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] 截图已保存: {filename}")
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] 程序已退出")


if __name__ == "__main__":
    main()
