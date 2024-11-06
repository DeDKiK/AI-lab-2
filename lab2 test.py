import tkinter as tk
from tkinter import Text, filedialog, Label, Button, Scale, Toplevel, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class FeatureExtractionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Feature Extraction Tool")

        # Зберігаємо зображення для кожного класу
        self.class_images = {'A': [], 'B': [], 'C': []}
        self.class_vectors = {'A': [], 'B': [], 'C': []}

        # Вікна для зображень кожного класу
        self.create_class_windows()

        # Основне вікно
        self.threshold_slider = Scale(master, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold")
        self.threshold_slider.pack()

        self.segments_slider = Scale(master, from_=2, to=10, orient=tk.HORIZONTAL, label="Number of Segments")
        self.segments_slider.pack()

        # Кнопки для завантаження зображень для кожного класу
        self.upload_class_a_button = Button(master, text="Upload Class A Images",
                                            command=lambda: self.upload_images('A'))
        self.upload_class_a_button.pack()

        self.upload_class_b_button = Button(master, text="Upload Class B Images",
                                            command=lambda: self.upload_images('B'))
        self.upload_class_b_button.pack()

        self.upload_class_c_button = Button(master, text="Upload Class C Images",
                                            command=lambda: self.upload_images('C'))
        self.upload_class_c_button.pack()

        self.upload_unknown_button = Button(master, text="Upload Unknown Image", command=self.upload_unknown_image)
        self.upload_unknown_button.pack()

        self.classify_button = Button(master, text="Classify Unknown Image", command=self.classify_image)
        self.classify_button.pack()

        # Текстові блоки для векторів кожного класу
        self.vector_text_a = Text(master, height=10)
        self.vector_text_a.pack(expand=True, fill='both')

        self.vector_text_b = Text(master, height=10)
        self.vector_text_b.pack(expand=True, fill='both')

        self.vector_text_c = Text(master, height=10)
        self.vector_text_c.pack(expand=True, fill='both')

        # Текстовий блок для вектора невідомого зображення та його шляху
        self.unknown_vector_text = Text(master, height=5)
        self.unknown_vector_text.pack(expand=True, fill='both')

        # Для зберігання невідомого зображення та його векторів
        self.unknown_image_path = None
        self.unknown_vector = []

        # Для зберігання невідомого зображення та його векторів
        self.unknown_image_path = None
        self.unknown_vector = []

    def create_class_windows(self):
        """ Створюємо вікна для відображення зображень кожного класу. """
        self.class_windows = {}
        for class_name in ['A', 'B', 'C']:
            self.class_windows[class_name] = Toplevel(self.master)
            self.class_windows[class_name].title(f"Class {class_name} Images")
            self.class_windows[class_name].geometry("400x400")

    def upload_images(self, class_name):
        """ Завантажуємо зображення для вказаного класу. """
        filepaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if filepaths:
            self.class_images[class_name] = list(filepaths)
            self.process_class_images(class_name)

    def process_class_images(self, class_name):
        """ Обробляємо завантажені зображення для класу. """
        vectors = []
        for filepath in self.class_images[class_name]:
            vector = self.process_image(filepath)
            if vector:
                vectors.append(vector)

        # Зберігаємо вектори для класу
        self.class_vectors[class_name] = vectors

        # Форматуємо вектори з заголовками для красивого виведення
        formatted_vectors = ""
        for i, vector in enumerate(vectors):
            formatted_vectors += f"Absolute Vector: [{', '.join([f'{val:.2f}' for val in vector])}]\n"
            formatted_vectors += f"Deresh S{i + 1}: [{', '.join([f'{val:.2f}' for val in vector])}]\n"
            formatted_vectors += f"Deresh M{i + 1}: [{', '.join([f'{max(val, 0.99):.2f}' for val in vector])}]\n\n"

        # Оновлюємо текстові блоки для векторів
        if class_name == 'A':
            self.vector_text_a.insert(tk.END, formatted_vectors)
        elif class_name == 'B':
            self.vector_text_b.insert(tk.END, formatted_vectors)
        elif class_name == 'C':
            self.vector_text_c.insert(tk.END, formatted_vectors)

        # Виводимо зображення у вікно класу
        self.display_class_images(class_name)

    def display_class_images(self, class_name):
        """ Виводимо зображення для певного класу у відповідному вікні. """
        window = self.class_windows[class_name]
        for widget in window.winfo_children():
            widget.destroy()

        for filepath in self.class_images[class_name]:
            img_with_segments = self.draw_segments(filepath)
            tk_image = ImageTk.PhotoImage(img_with_segments)

            image_label = Label(window, image=tk_image)
            image_label.image = tk_image  # Keep reference
            image_label.pack(side=tk.LEFT)

    def draw_segments(self, filepath):
        """ Малюємо сегменти на зображенні для візуалізації. """
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read the image {filepath}.")
            return None

        # Кількість сегментів
        segments = self.segments_slider.get()
        height, width = img.shape
        segment_width = width // segments

        # Малюємо вертикальні лінії для кожного сегменту
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(1, segments):
            cv2.line(img_color, (i * segment_width, 0), (i * segment_width, height), (255, 0, 0), 1)

        # Перетворюємо на зображення для tkinter
        img_pil = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.resize((100, 100))  # Resize for display
        return img_pil

    def process_image(self, filepath):
        """ Обробка зображення, створення векторів ознак із сегментацією. """
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read the image {filepath}.")
            return None

        # Порогове перетворення
        _, thresholded = cv2.threshold(img, self.threshold_slider.get(), 255, cv2.THRESH_BINARY)

        # Сегментація та формування векторів
        segments = self.segments_slider.get()
        height, width = thresholded.shape
        segment_width = width // segments

        absolute_vector = []
        for i in range(segments):
            segment = thresholded[:, i * segment_width:(i + 1) * segment_width]
            count_black_pixels = np.sum(segment == 0)
            absolute_vector.append(count_black_pixels)

        # Нормалізація вектора
        normalized_vector = [x / sum(absolute_vector) for x in absolute_vector] if sum(absolute_vector) > 0 else [0] * segments
        return normalized_vector

    def calculate_class_max_min(self, class_name):
        """ Обчислюємо максимальні та мінімальні вектори для кожного класу. """
        vectors = self.class_vectors[class_name]
        if not vectors:
            return None, None

        max_vector = np.max(vectors, axis=0)
        min_vector = np.min(vectors, axis=0)

        return max_vector, min_vector

    def upload_unknown_image(self):
        """ Завантажуємо невідоме зображення. """
        self.unknown_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if self.unknown_image_path:
            self.unknown_vector = self.process_image(self.unknown_image_path)
            if self.unknown_vector:
                # Виводимо шлях та вектор невідомого зображення
                self.unknown_vector_text.delete(1.0, tk.END)  # Очистити попередній текст
                self.unknown_vector_text.insert(tk.END, f"Path: {self.unknown_image_path}\n")
                self.unknown_vector_text.insert(tk.END, f"Vector: [{', '.join(map(lambda x: f'{x:.5f}', self.unknown_vector))}]\n")

    def classify_image(self):
        """ Класифікуємо невідоме зображення на основі максимальних та мінімальних векторів кожного класу. """
        if not self.unknown_vector:
            messagebox.showerror("Error", "Please upload an unknown image.")
            return

        tolerance = 0.05  # Допуск для порівняння векторів
        classification_result = None

        for class_name in ['A', 'B', 'C']:
            max_vector, min_vector = self.calculate_class_max_min(class_name)
            if max_vector is None or min_vector is None:
                continue

            # Перевірка, чи належить вектор невідомого зображення до діапазону класу
            in_range = all(min_val - tolerance <= val <= max_val + tolerance for val, min_val, max_val in
                           zip(self.unknown_vector, min_vector, max_vector))

            if in_range:
                classification_result = f"Unknown Image belongs to Class {class_name}"
                break

        if classification_result is None:
            classification_result = "Unknown Image does not belong to any class."

        # Виводимо результат класифікації у текстовому блоці для невідомого зображення
        self.unknown_vector_text.insert(tk.END, f"{classification_result}\n")

root = tk.Tk()
app = FeatureExtractionApp(root)
root.mainloop()
