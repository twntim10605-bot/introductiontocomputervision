mport os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mediapipe as mp

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("高科大資工 - 多功能影像處理專題系統")
        self.root.geometry("1280x800")
        
        self.file_path = None
        self.orig_bgr = None      
        self.current_img = None   
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        self.setup_ui()

    def setup_ui(self):
        top_frame = ttk.LabelFrame(self.root, text=" 1. 影像載入與基礎解析 (Metadata) ")
        top_frame.pack(fill="x", padx=10, pady=5)
        
        btn_load = ttk.Button(top_frame, text="載入影像 (PNG/JPG/BMP)", command=self.load_image)
        btn_load.pack(side="left", padx=10, pady=10)
        
        self.lbl_meta = ttk.Label(top_frame, text="尚未載入影像。請點擊左側按鈕選擇檔案...", font=("Arial", 11))
        self.lbl_meta.pack(side="left", padx=20)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.tab_color = ttk.Frame(self.notebook)
        self.tab_kernel = ttk.Frame(self.notebook)
        self.tab_face = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_color, text=" 2. 色彩空間切換與直方圖 ")
        self.notebook.add(self.tab_kernel, text=" 3. 自訂卷積核濾波與 PSNR 評估 ")
        self.notebook.add(self.tab_face, text=" 4. AI 人臉偵測模組 ")
        
        self.init_tab_color()
        self.init_tab_kernel()
        self.init_tab_face()

    def load_image(self):
        file_types = [("Image files", "*.png *.jpg *.jpeg *.bmp")]
        path = filedialog.askopenfilename(title="選擇影像檔案", filetypes=file_types)
        if not path:
            return
            
        self.file_path = path
        self.orig_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if self.orig_bgr is None:
            messagebox.showerror("錯誤", "影像檔案毀損或不支援該格式！")
            return
            
        file_size_kb = os.path.getsize(path) / 1024
        h, w, c = self.orig_bgr.shape
        color_dim = "24-bit 真彩色 (RGB 各 8-bit)" if c == 3 else f"{c*8}-bit"
        
        meta_info = f"【解析度】: {w} × {h} 像素   |   【色彩維度】: {color_dim}   |   【檔案大小】: {file_size_kb:.2f} KB   |   【路徑】: {path}"
        self.lbl_meta.config(text=meta_info)
        
        self.update_color_tab()
        self.update_kernel_tab()
        self.update_face_tab()

    def init_tab_color(self):
        self.color_img_frame = ttk.Frame(self.tab_color)
        self.color_img_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        mode_frame = ttk.Frame(self.color_img_frame)
        mode_frame.pack(side="top", fill="x", pady=5)
        ttk.Label(mode_frame, text="切換檢視檢視模式：").pack(side="left")
        
        self.combo_mode = ttk.Combobox(mode_frame, values=[
            "原始影像 (Original)", 
            "R 分量頻道", "G 分量頻道", "B 分量頻道",
            "Y 亮度分量 (YUV)", "U 色度分量 (YUV)", "V 色度分量 (YUV)"
        ], state="readonly", width=25)
        self.combo_mode.set("原始影像 (Original)")
        self.combo_mode.pack(side="left", padx=5)
        self.combo_mode.bind("<<ComboboxSelected>>", lambda e: self.update_color_tab())
        
        self.lbl_color_canvas = ttk.Label(self.color_img_frame)
        self.lbl_color_canvas.pack(fill="both", expand=True, pady=5)
        
        self.color_hist_frame = ttk.Frame(self.tab_color, width=450)
        self.color_hist_frame.pack(side="right", fill="both", padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=self.color_hist_frame)
        self.canvas_hist.get_tk_widget().pack(fill="both", expand=True)

    def update_color_tab(self):
        if self.orig_bgr is None: return
        mode = self.combo_mode.get()
        
        b, g, r = cv2.split(self.orig_bgr)
        yuv = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        
        target_img = None
        hist_data = None
        hist_color = 'black'
        
        if mode == "原始影像 (Original)":
            target_img = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2RGB)
            hist_data = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2GRAY)
            hist_color = 'purple'
        elif mode == "R 分量頻道":
            target_img = cv2.merge([r, np.zeros_like(r), np.zeros_like(r)]) 
            hist_data = r
            hist_color = 'red'
        elif mode == "G 分量頻道":
            target_img = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)]) 
            hist_data = g
            hist_color = 'green'
        elif mode == "B 分量頻道":
            target_img = cv2.merge([np.zeros_like(b), np.zeros_like(b), b]) 
            hist_data = b
            hist_color = 'blue'
        elif mode == "Y 亮度分量 (YUV)":
            target_img = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)
            hist_data = y
            hist_color = 'gray'
        elif mode == "U 色度分量 (YUV)":
            target_img = cv2.cvtColor(u, cv2.COLOR_GRAY2RGB)
            hist_data = u
            hist_color = 'orange'
        elif mode == "V 色度分量 (YUV)":
            target_img = cv2.cvtColor(v, cv2.COLOR_GRAY2RGB)
            hist_data = v
            hist_color = 'brown'

        self.display_on_label(target_img, self.lbl_color_canvas)
        
        self.ax.clear()
        hist = cv2.calcHist([hist_data], [0], None, [256], [0, 256])
        self.ax.bar(range(256), hist.ravel(), width=1.0, color=hist_color, edgecolor=hist_color)
        self.ax.set_title(f"{mode}強度分佈直方圖", fontproperties="Microsoft JhengHei")
        self.ax.set_xlim([0, 255])
        self.ax.set_xlabel("像素強度 (0-255)", fontproperties="Microsoft JhengHei")
        self.ax.set_ylabel("像素數量", fontproperties="Microsoft JhengHei")
        self.fig.tight_layout()
        self.canvas_hist.draw()

    def init_tab_kernel(self):
        ctrl_frame = ttk.Frame(self.tab_kernel)
        ctrl_frame.pack(side="top", fill="x", padx=10, pady=5)
        
        ttk.Label(ctrl_frame, text="快速範本設定:").pack(side="left")
        self.combo_tmpl = ttk.Combobox(ctrl_frame, values=["高斯平滑 (Gaussian)", "邊緣銳化 (Sharpen)", "浮雕效果 (Emboss)"], state="readonly", width=18)
        self.combo_tmpl.pack(side="left", padx=5)
        self.combo_tmpl.bind("<<ComboboxSelected>>", self.apply_template)
        
        self.var_norm = tk.BooleanVar(value=True)
        chk_norm = ttk.Checkbutton(ctrl_frame, text="啟用核矩陣正規化 (Normalization)", variable=self.var_norm)
        chk_norm.pack(side="left", padx=15)
        
        btn_apply_filter = ttk.Button(ctrl_frame, text="⚙️ 執行自訂卷積濾波", command=self.apply_custom_filter)
        btn_apply_filter.pack(side="left", padx=5)
        
        self.lbl_psnr = ttk.Label(ctrl_frame, text="PSNR 指標: -- dB", font=("Arial", 12, "bold"), foreground="blue")
        self.lbl_psnr.pack(side="right", padx=20)
        
        matrix_frame = ttk.LabelFrame(self.tab_kernel, text=" 自訂 3x3 濾波矩陣權重輸入 ")
        matrix_frame.pack(side="top", pady=5)
        
        self.entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                ent = ttk.Entry(matrix_frame, width=8, justify="center")
                ent.grid(row=i, column=j, padx=4, pady=4)
                ent.insert(0, "0" if (i!=1 or j!=1) else "1") 
                row_entries.append(ent)
            self.entries.append(row_entries)
            
        view_frame = ttk.Frame(self.tab_kernel)
        view_frame.pack(side="bottom", fill="both", expand=True, pady=5)
        
        self.lbl_img_before = ttk.Label(view_frame, text="【濾波前原始圖】", anchor="center")
        self.lbl_img_before.pack(side="left", fill="both", expand=True, padx=5)
        
        self.lbl_img_after = ttk.Label(view_frame, text="【濾波後影像成果】", anchor="center")
        self.lbl_img_after.pack(side="right", fill="both", expand=True, padx=5)

    def apply_template(self, event):
        tmpl = self.combo_tmpl.get()
        if tmpl == "高斯平滑 (Gaussian)":
            weights = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            self.var_norm.set(True)
        elif tmpl == "邊緣銳化 (Sharpen)":
            weights = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
            self.var_norm.set(False)
        elif tmpl == "浮雕效果 (Emboss)":
            weights = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]
            self.var_norm.set(False)
            
        for i in range(3):
            for j in range(3):
                self.entries[i][j].delete(0, tk.END)
                self.entries[i][j].insert(0, str(weights[i][j]))

    def apply_custom_filter(self):
        if self.orig_bgr is None:
            messagebox.showwarning("警告", "請先載入任何點陣影像。")
            return
            
        try:
            kernel = np.zeros((3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    kernel[i, j] = float(self.entries[i][j].get())
        except ValueError:
            messagebox.showerror("錯誤", "矩陣輸入格只能填寫數值數字！")
            return
            
        if self.var_norm.get():
            k_sum = np.sum(kernel)
            if k_sum != 0:
                kernel /= k_sum
                
        filtered_bgr = cv2.filter2D(self.orig_bgr, -1, kernel)
        
        psnr_val = cv2.PSNR(self.orig_bgr, filtered_bgr)
        self.lbl_psnr.config(text=f"量化指標 PSNR: {psnr_val:.2f} dB")
        
        img_orig_rgb = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2RGB)
        img_filt_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        
        self.display_on_label(img_orig_rgb, self.lbl_img_before)
        mport cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# 初始化 MediaPipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# 讀取下載的圖片
image = cv2.imread('test_faces.jpg')
# 將 BGR 轉為 RGB 格式（MediaPipe 需要 RGB）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 設定 Face Mesh 參數
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=10,  # 設定最多偵測的人臉數量
    refine_landmarks=True,
    min_detection_confidence=0.5
) as face_mesh:
    
    # 執行偵測
    results = face_mesh.process(image_rgb)

    # 複製一份原圖用於繪製輪廓
    annotated_image = image_rgb.copy()
    
    face_count = 0
    if results.multi_face_landmarks:
        face_count = len(results.multi_face_landmarks)
        
        for face_landmarks in results.multi_face_landmarks:
            # 繪製人臉網格輪廓線
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # 繪製五官外框輪廓（眼睛、眉毛、嘴唇、臉部外輪廓）
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
    # 顯示結果與數量
    print(f"偵測到的人臉總數 (Face Count): {face_count}")
    
    # 使用 matplotlib 繪製對比圖
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Detected Faces: {face_count}")
    plt.imshow(annotated_image)
    plt.axis('off')
    
    plt.show()
