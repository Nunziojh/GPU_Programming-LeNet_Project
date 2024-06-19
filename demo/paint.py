import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import subprocess  # for launching external processes
import os

def image_to_vector(image):
    image_array = np.array(image, dtype=np.float32)
    vector = np.pad(image_array, ((2, 2), (2, 2)), mode='constant', constant_values=0)
    new_vector = vector.flatten() / 255.0
    return new_vector

class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Number")
        
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white")
        self.canvas.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=0, column=0)

        self.save_button = tk.Button(self.button_frame, text="Go", command=self.save_and_process_image)
        self.save_button.grid(row=0, column=1)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        self.old_x = None
        self.old_y = None
        self.draw_color = "black"

    def start_draw(self, event):
        self.old_x = event.x
        self.old_y = event.y

    def draw(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=7, fill=self.draw_color, capstyle=tk.ROUND, smooth=tk.TRUE)
            self.old_x = event.x
            self.old_y = event.y

    def stop_draw(self, event):
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        self.canvas.delete("all")

    def save_and_process_image(self):
        # file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        file_path = "immagine.png"
        # if file_path:
        # Get the canvas content and save it to a file
        self.canvas.update()
        ps_file = "tmp_canvas.ps"
        self.canvas.postscript(file=ps_file)
        # img = Image.open(ps_file)
        img = Image.open(ps_file).convert("1") #---MODIFIED---

        # Create a new image with the same dimensions as the canvas
        new_img = Image.new("L", (self.canvas.winfo_width(), self.canvas.winfo_height()), "white")
        draw = ImageDraw.Draw(new_img)

        # Paste the canvas image onto the new image
        new_img.paste(img, (0, 0))

        new_img = ImageOps.invert(new_img)

        bbox = new_img.getbbox()
        new_img = new_img.crop(bbox)
        margin = 60
        new_img = ImageOps.expand(new_img, border=margin, fill=0)

        new_img = new_img.resize((28, 28), resample=Image.Resampling.LANCZOS)

        enhancer = ImageEnhance.Brightness(new_img)
        new_img = enhancer.enhance(2)

        # Save the new image as a PNG file
        new_img.save(file_path)

        # Create a vector from the image
        vector = image_to_vector(new_img)

        n_values = 0
        with open("input_img.txt", "w") as file:
            for val in vector:
                file.write("{0:0.2f}".format(val))
                n_values += 1
                if n_values % 32 == 0:
                    file.write("\n")
                else:
                    file.write(" ")

        # Call another process that uses the saved image (example: print file_path)
        self.call_external_process(file_path)

        self.root.quit()

    def call_external_process(self, image_path):
        # Example: Print the image path
        # print(f"Saved image path: {image_path}")
        subprocess.run([".\\a.exe"], check=True)
        # Here you can replace the print statement with your actual code to call another process

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()

    os.remove("tmp_canvas.ps")
    os.remove("input_img.txt")
