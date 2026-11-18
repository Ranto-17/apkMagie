import cv2
import numpy as np
from PIL import Image as PILImage, ImageEnhance, ImageFilter, ImageChops
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.utils import platform

if platform == 'android':
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE])
    from kivy.uix.camera import Camera

class MagicCardApp(App):
    def build(self):
        self.frames = []
        self.current_step = 0
        self.is_animating = False
        self.current_frame_bgr = None
        
        Window.bind(on_key_down=self.on_key_down)
        self.layout = BoxLayout(orientation='vertical')
        
        if platform == 'android':
            # Caméra native Android (caméra arrière index 0)
            self.camera = Camera(index=0, play=True, resolution=(1280, 720), allow_stretch=True, keep_ratio=True)
            self.layout.add_widget(self.camera)
            self.display_view = Image(allow_stretch=True, keep_ratio=True)
        else:
            # Caméra PC Windows
            self.display_view = Image(allow_stretch=True, keep_ratio=True)
            self.layout.add_widget(self.display_view)
            self.capture = cv2.VideoCapture(0)
            self.cam_event = Clock.schedule_interval(self.update_camera_pc, 1.0 / 30.0)
            
        self.btn_capture = Button(text="Prendre la photo", size_hint_y=0.12)
        self.btn_capture.bind(on_press=self.take_photo)
        self.layout.add_widget(self.btn_capture)
        
        return self.layout

    def update_camera_pc(self, dt):
        ret, frame = self.capture.read()
        if ret:
            self.current_frame_bgr = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_pil_image(PILImage.fromarray(frame_rgb))

    def take_photo(self, instance):
        if platform == 'android':
            photo_path = "temp_capture.png"
            self.camera.export_to_png(photo_path)
            img_bgr = cv2.imread(photo_path)
            if img_bgr is None:
                return
            # Correction de rotation fréquente sur capteur Android
            if img_bgr.shape[1] > img_bgr.shape[0]:
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
            self.current_frame_bgr = img_bgr
        else:
            if self.current_frame_bgr is None:
                return

        success = self.prepare_magic_frames(self.current_frame_bgr, "3_coeur.jpg")
        
        if success:
            if platform == 'android':
                self.camera.play = False
                self.layout.remove_widget(self.camera)
                self.layout.remove_widget(self.btn_capture)
                self.layout.add_widget(self.display_view)
            else:
                self.cam_event.cancel()
                self.capture.release()
                self.layout.remove_widget(self.btn_capture)
            self.display_pil_image(self.frames[int(0)])
        else:
            self.btn_capture.text = "Carte non vue - Réessayer"

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[int(0)] = pts[np.argmin(s)]
        rect[int(2)] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[int(1)] = pts[np.argmin(diff)]
        rect[int(3)] = pts[np.argmax(diff)]
        return rect

    def prepare_magic_frames(self, img_bg, card_path):
        img_card = cv2.imread(card_path)
        if img_card is None:
            return False

        gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        card_corners = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < 3000:
                continue
            rect = cv2.minAreaRect(c)
            (w_rect, h_rect) = rect[1]
            if w_rect == 0 or h_rect == 0:
                continue
            aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)
            if 1.25 <= aspect_ratio <= 1.75:
                box = cv2.boxPoints(rect)
                card_corners = self.order_points(np.array(box, dtype="float32"))
                break
                
        if card_corners is None:
            return False

        pil_bg = PILImage.fromarray(cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB))
        h_c, w_c = img_card.shape[:2]
        h_bg, w_bg = img_bg.shape[:2]
        
        p0 = [float(0), float(0)]
        p1 = [float(w_c - 1), float(0)]
        p2 = [float(w_c - 1), float(h_c - 1)]
        p3 = [float(0), float(h_c - 1)]
        src_pts = np.array([p0, p1, p2, p3], dtype="float32")
        
        matrix = cv2.getPerspectiveTransform(src_pts, card_corners)
        warped_card = cv2.warpPerspective(img_card, matrix, (w_bg, h_bg))
        
        lighting_map = cv2.GaussianBlur(img_bg, (45, 45), 0)
        lighting_map = cv2.convertScaleAbs(lighting_map, alpha=1.25, beta=25)
        
        noise = np.random.normal(0, 6, warped_card.shape).astype(np.int16)
        warped_noisy = np.clip(warped_card.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        pil_warped = PILImage.fromarray(cv2.cvtColor(warped_noisy, cv2.COLOR_BGR2RGB))
        pil_lighting = PILImage.fromarray(cv2.cvtColor(lighting_map, cv2.COLOR_BGR2RGB))
        
        pil_warped = ImageEnhance.Contrast(pil_warped).enhance(0.85)
        pil_warped = pil_warped.filter(ImageFilter.GaussianBlur(radius=1.0))
        pil_lit_card = ImageChops.multiply(pil_warped, pil_lighting)
        
        poly_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
        cv2.fillConvexPoly(poly_mask, card_corners.astype(int), 255)
        real_mask = cv2.bitwise_and(poly_mask, closed)
        
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        real_mask = cv2.erode(real_mask, erode_kernel, iterations=1)
        pil_mask = PILImage.fromarray(real_mask).filter(ImageFilter.GaussianBlur(radius=1.8))
        
        pil_final = pil_bg.copy()
        pil_final.paste(pil_lit_card, (int(0), int(0)), pil_mask)
        
        self.frames = []
        steps = 25
        for i in range(steps + 1):
            alpha = i / float(steps)
            blended = PILImage.blend(pil_bg, pil_final, alpha)
            self.frames.append(blended)
            
        return True

    def display_pil_image(self, pil_img):
        pil_img = pil_img.transpose(PILImage.FLIP_TOP_BOTTOM)
        data = pil_img.tobytes()
        tex = Texture.create(size=pil_img.size, colorfmt='rgb')
        tex.blit_buffer(data, colorfmt='rgb', bufferfmt='ubyte')
        self.display_view.texture = tex

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        if key in (24, 32) and len(self.frames) > 1 and not self.is_animating:
            self.is_animating = True
            self.current_step = 0
            Clock.schedule_interval(self.animate_reveal, 0.06)
            return True
        return False

    def animate_reveal(self, dt):
        if self.current_step < len(self.frames):
            self.display_pil_image(self.frames[self.current_step])
            self.current_step += 1
        else:
            self.is_animating = False
            return False

    def on_stop(self):
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()

if __name__ == '__main__':
    MagicCardApp().run()
