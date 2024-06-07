import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os

# Funzione per convertire l'immagine in un vettore
def image_to_vector(image):
    image_array = np.array(image, dtype=np.float32)
    vector = np.pad(image_array, ((2, 2), (2, 2)), mode='constant', constant_values=0)
    new_vector = vector.flatten() / 255.0
    return new_vector

def save_image(canvas):

    canvas.postscript(file="tmp.ps", colormode='color')
    img = Image.open("tmp.ps").convert('L')

    img = ImageOps.invert(img)
    img.save("image.png")

    # Crop the image
    bbox = img.getbbox()
    img = img.crop(bbox)
    # Add some margin
    margin = 40
    img = ImageOps.expand(img, border=margin, fill=0)

    img.save("cropped_img.png")

    # Risize image
    img = img.resize((28,28), resample=Image.LANCZOS)
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(2)
    
    img.save("resized_img.png")
    vector = image_to_vector(img)

    n_values = 0
    with open("input_img.txt", "w") as file:
        for val in vector:
            #file.write("{0:0.2f}".format(val))
            file.write(f"{val:.16f}") 
            n_values += 1
            if n_values % 32 == 0:
                file.write("\n")
            else:
                file.write(" ")

    global root
    root.quit()

def draw(event):
    global prev_x
    global prev_y
    x, y = event.x, event.y
    if(prev_x is not None and prev_y is not None):
        canvas.create_line(prev_x, prev_y, x, y, width=15, fill="black", smooth=True)
    else:
        canvas.create_line(x, y, x, y, width=15, fill="black", smooth=True)
    prev_x = x
    prev_y = y

def onrelease(event):
    global prev_x
    global prev_y
    prev_x = None
    prev_y = None

if __name__ == "__main__":

    root = tk.Tk()
    root.title("Draw and Save")

    prev_x = None
    prev_y = None

    canvas = tk.Canvas(root, width=400, height=400, bg="white")
    canvas.pack(expand=True, fill="both")
    canvas.bind("<B1-Motion>", draw)
    canvas.bind("<ButtonRelease-1>", onrelease)

    save_button = tk.Button(root, text="Launch", command=lambda: save_image(canvas))
    save_button.pack()

    root.mainloop()

    os.remove("tmp.ps")
    os.system("./backward_new_usage.out")
