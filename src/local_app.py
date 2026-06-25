from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from metrics import bce_dice_loss, dice_coef, iou_coef


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = APP_DIR / "models" / "unet_tusimple_1000.keras"


class LaneSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Сегментация дорожных полос")
        self.root.geometry("1280x760")
        self.root.minsize(1100, 680)
        self.root.configure(bg="#f3f5f7")

        self.model = None
        self.model_path = None
        self.image_path = None
        self.original = None
        self.prediction = None
        self.mask_gray = None
        self.mask_binary = None
        self.overlay = None
        self._panel_images = []

        self.threshold = tk.DoubleVar(value=0.50)
        self.alpha = tk.DoubleVar(value=0.55)
        self.status = tk.StringVar(value="Загрузите модель и выберите изображение дороги.")
        self.model_label = tk.StringVar(value="Модель не загружена")
        self.image_label = tk.StringVar(value="Изображение не выбрано")

        self._configure_style()
        self._build_ui()
        self._try_autoload_model()

    def _configure_style(self):
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")

        style.configure("TFrame", background="#f3f5f7")
        style.configure("Header.TFrame", background="#16202a")
        style.configure("Header.TLabel", background="#16202a", foreground="#ffffff")
        style.configure("Panel.TLabelframe", background="#f3f5f7")
        style.configure("Panel.TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", background="#e9eef3", foreground="#1d2731")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(16, 12))
        header.pack(fill=tk.X)

        ttk.Label(
            header,
            text="Сегментация дорожных полос",
            style="Header.TLabel",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Локальная проверка U-Net модели на изображениях дороги",
            style="Header.TLabel",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 0))

        controls = ttk.Frame(self.root, padding=(12, 10))
        controls.pack(fill=tk.X)

        ttk.Button(
            controls,
            text="Загрузить модель",
            style="Primary.TButton",
            command=self.load_model,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            controls,
            text="Открыть фото",
            style="Primary.TButton",
            command=self.open_image,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            controls,
            text="Сохранить результат",
            command=self.save_overlay,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            controls,
            text="Сохранить маску",
            command=self.save_mask,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            controls,
            text="Сбросить",
            command=self.reset_image,
        ).pack(side=tk.LEFT, padx=6)

        slider_box = ttk.Frame(controls)
        slider_box.pack(side=tk.RIGHT)

        ttk.Label(slider_box, text="Порог маски").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Scale(
            slider_box,
            variable=self.threshold,
            from_=0.10,
            to=0.90,
            orient=tk.HORIZONTAL,
            length=170,
            command=self._on_slider_change,
        ).grid(row=0, column=1, padx=(0, 8))
        self.threshold_value = ttk.Label(slider_box, text="0.50", width=4)
        self.threshold_value.grid(row=0, column=2, sticky="e")

        ttk.Label(slider_box, text="Наложение").grid(row=1, column=0, sticky="w", padx=(0, 8))
        ttk.Scale(
            slider_box,
            variable=self.alpha,
            from_=0.20,
            to=0.85,
            orient=tk.HORIZONTAL,
            length=170,
            command=self._on_slider_change,
        ).grid(row=1, column=1, padx=(0, 8), pady=(6, 0))
        self.alpha_value = ttk.Label(slider_box, text="0.55", width=4)
        self.alpha_value.grid(row=1, column=2, sticky="e")

        info = ttk.Frame(self.root, padding=(12, 0))
        info.pack(fill=tk.X)
        ttk.Label(info, textvariable=self.model_label).pack(side=tk.LEFT, padx=(0, 24))
        ttk.Label(info, textvariable=self.image_label).pack(side=tk.LEFT)

        self.workspace = ttk.Frame(self.root, padding=(12, 10))
        self.workspace.pack(fill=tk.BOTH, expand=True)

        self.empty_state = ttk.Frame(self.workspace)
        self.empty_state.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            self.empty_state,
            text="Изображение не выбрано",
            bg="#f3f5f7",
            fg="#1d2731",
            font=("Segoe UI", 18, "bold"),
        ).pack(expand=True, pady=(0, 8))
        tk.Label(
            self.empty_state,
            text="Нажмите «Открыть фото», чтобы построить маску дорожных полос.",
            bg="#f3f5f7",
            fg="#687582",
            font=("Segoe UI", 11),
        ).pack(pady=(0, 140))

        self.panels = ttk.Frame(self.workspace)

        self.image_labels = []
        for index, title in enumerate(("Исходное фото", "Предсказанная маска", "Полосы поверх фото")):
            frame = ttk.Labelframe(self.panels, text=title, style="Panel.TLabelframe", padding=8)
            frame.grid(row=0, column=index, sticky="nsew", padx=6)
            self.panels.columnconfigure(index, weight=1)
            self.panels.rowconfigure(0, weight=1)

            label = tk.Label(frame, bg="#20252b", width=390, height=500, bd=0)
            label.pack(fill=tk.BOTH, expand=True)
            self.image_labels.append(label)

        ttk.Label(
            self.root,
            textvariable=self.status,
            style="Status.TLabel",
            anchor="w",
            padding=(12, 7),
        ).pack(fill=tk.X, side=tk.BOTTOM)

    def _try_autoload_model(self):
        if DEFAULT_MODEL.exists():
            self._load_model_from_path(DEFAULT_MODEL, silent=True)

    def load_model(self):
        initial_dir = DEFAULT_MODEL.parent if DEFAULT_MODEL.parent.exists() else APP_DIR
        path = filedialog.askopenfilename(
            title="Выберите обученную модель Keras",
            initialdir=initial_dir,
            filetypes=[("Keras model", "*.keras"), ("Все файлы", "*.*")],
        )
        if path:
            self._load_model_from_path(Path(path))

    def _load_model_from_path(self, path, silent=False):
        try:
            self.status.set("Загрузка модели...")
            self.root.update_idletasks()
            self.model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "bce_dice_loss": bce_dice_loss,
                    "dice_coef": dice_coef,
                    "iou_coef": iou_coef,
                },
            )
            self.model_path = Path(path)
            self.model_label.set(f"Модель: {self.model_path.name}")
            self.status.set("Модель загружена. Теперь выберите изображение дороги.")
        except Exception as exc:
            self.model = None
            self.model_path = None
            self.model_label.set("Модель не загружена")
            self.status.set("Не удалось загрузить модель.")
            if not silent:
                messagebox.showerror("Ошибка модели", str(exc))

    def open_image(self):
        if self.model is None:
            messagebox.showinfo("Нужна модель", "Сначала загрузите файл модели .keras.")
            return

        path = filedialog.askopenfilename(
            title="Выберите изображение дороги",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp"), ("Все файлы", "*.*")],
        )
        if not path:
            return

        image_bgr = cv2.imread(path)
        if image_bgr is None:
            messagebox.showerror("Ошибка изображения", f"Не удалось открыть файл:\n{path}")
            return

        self.image_path = Path(path)
        self.original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.image_label.set(f"Фото: {self.image_path.name}")
        self.run_prediction()

    def run_prediction(self):
        resized = cv2.resize(self.original, (256, 256))
        model_input = resized.astype(np.float32)[np.newaxis, ...] / 255.0

        self.status.set("Модель анализирует изображение...")
        self.root.update_idletasks()

        self.prediction = self.model.predict(model_input, verbose=0)[0, ..., 0]
        self.refresh_overlay()
        self.status.set("Готово. Белые области на маске показывают найденные дорожные полосы.")

    def _on_slider_change(self, _value):
        self.threshold_value.configure(text=f"{self.threshold.get():.2f}")
        self.alpha_value.configure(text=f"{self.alpha.get():.2f}")
        self.refresh_overlay()

    def refresh_overlay(self):
        if self.original is None or self.prediction is None:
            return

        mask = cv2.resize(
            self.prediction,
            (self.original.shape[1], self.original.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        self.mask_gray = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        self.mask_binary = self.mask_gray >= int(self.threshold.get() * 255)

        colored = self.original.copy()
        colored[self.mask_binary] = [255, 45, 45]
        alpha = float(self.alpha.get())
        self.overlay = cv2.addWeighted(self.original, 1.0 - alpha, colored, alpha, 0)

        mask_rgb = np.stack([self.mask_gray] * 3, axis=-1)
        self.show_images([self.original, mask_rgb, self.overlay])

    def show_images(self, images):
        self._show_result_panels()
        self._panel_images = []

        for label, image in zip(self.image_labels, images):
            display = self._fit_image(image, 390, 500)
            photo = ImageTk.PhotoImage(display)
            label.configure(image=photo)
            label.image = photo
            self._panel_images.append(photo)

    def _show_result_panels(self):
        if self.empty_state.winfo_manager():
            self.empty_state.pack_forget()
        if not self.panels.winfo_manager():
            self.panels.pack(fill=tk.BOTH, expand=True)

    def _show_empty_state(self):
        if self.panels.winfo_manager():
            self.panels.pack_forget()
        if not self.empty_state.winfo_manager():
            self.empty_state.pack(fill=tk.BOTH, expand=True)
        self._panel_images = []
        for label in self.image_labels:
            label.configure(image="")
            label.image = None

    def _fit_image(self, image, max_width, max_height):
        pil = Image.fromarray(image)
        pil.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (max_width, max_height), (32, 37, 43))
        x = (max_width - pil.width) // 2
        y = (max_height - pil.height) // 2
        canvas.paste(pil, (x, y))
        return canvas

    def save_overlay(self):
        if self.overlay is None:
            messagebox.showinfo("Нет результата", "Сначала откройте изображение.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить изображение с полосами",
            defaultextension=".png",
            initialfile=self._default_output_name("overlay"),
            filetypes=[("PNG image", "*.png"), ("Все файлы", "*.*")],
        )
        if path:
            cv2.imwrite(path, cv2.cvtColor(self.overlay, cv2.COLOR_RGB2BGR))
            self.status.set(f"Результат сохранён: {path}")

    def save_mask(self):
        if self.mask_gray is None:
            messagebox.showinfo("Нет маски", "Сначала откройте изображение.")
            return

        path = filedialog.asksaveasfilename(
            title="Сохранить предсказанную маску",
            defaultextension=".png",
            initialfile=self._default_output_name("mask"),
            filetypes=[("PNG image", "*.png"), ("Все файлы", "*.*")],
        )
        if path:
            cv2.imwrite(path, self.mask_gray)
            self.status.set(f"Маска сохранена: {path}")

    def reset_image(self):
        self.image_path = None
        self.original = None
        self.prediction = None
        self.mask_gray = None
        self.mask_binary = None
        self.overlay = None
        self.image_label.set("Изображение не выбрано")
        self.status.set("Изображение сброшено. Можно открыть новое фото.")

        self._show_empty_state()

    def _default_output_name(self, suffix):
        if self.image_path is None:
            return f"lane_{suffix}.png"
        return f"{self.image_path.stem}_lane_{suffix}.png"


def main():
    root = tk.Tk()
    LaneSegmentationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
