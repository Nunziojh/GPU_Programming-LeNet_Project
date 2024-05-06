# import tkinter as tk
# import numpy as np
# from PIL import Image, ImageDraw
# from wand.image import Image as WandImage
# import io
# import os


# # Funzione chiamata quando il mouse viene premuto
# def on_click(event):
#     global prev_x, prev_y
#     prev_x, prev_y = event.x, event.y

# # Funzione chiamata quando il mouse viene rilasciato
# def on_release(event):
#     global prev_x, prev_y
#     prev_x, prev_y = None, None
#     save_image()

# # Funzione chiamata quando il mouse si muove
# def on_motion(event):
#     global prev_x, prev_y
#     if prev_x is not None and prev_y is not None:
#         x, y = event.x, event.y
#         c.create_line(prev_x, prev_y, x, y, width=5, fill='white')
#         prev_x, prev_y = x, y

# # Funzione per convertire l'immagine in un vettore
# def image_to_vector(image):
#     resized_image = image.resize(image_size).convert('L')
#     image_array = np.array(resized_image)
#     vector = np.pad(image_array, ((2, 2), (2, 2)), mode='constant', constant_values=0)
#     new_vector = vector.flatten() / 255.0
#     return new_vector

# def wrapper():
#     save_image()
#     global root
#     root.quit()


# def save_image():
#     global vector

#     # Salva il canvas come file PostScript
#     canvas_post = c.postscript(colormode='mono')
#     fd = open("debug.txt", "w")
#     print(canvas_post, file=fd)
#     img = Image.open(io.BytesIO(canvas_post.encode('utf-8')))

#     img.save("image.png")
#     # Taglia l'immagine
#     img_cropped = img.resize((28,28))

#     vector = image_to_vector(img_cropped)

#     n_values = 0
#     with open("input_img.txt", "w") as file:
#         for val in vector:
#             file.write(str(val))
#             n_values += 1
#             if n_values % 32 == 0:
#                 file.write("\n")
#             else:
#                 file.write("\t")


# if __name__ == "__main__":

#     # Dimensioni del canvas
#     canvas_width = 300
#     canvas_height = 300

#     # Dimensioni dell'immagine
#     image_size = (28, 28)

#     # Variabile per memorizzare il punto iniziale del tratto
#     prev_x, prev_y = None, None

#     # Inizializzazione della finestra tkinter
#     root = tk.Tk()
#     root.title("Disegna e converte in vettore")

#     # Creazione del canvas
#     c = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='black', background='black')
#     c.pack()

#     saveButton = tk.Button(root, text="Save", command=wrapper)
#     saveButton.pack()

#     # Bind degli eventi del mouse
#     c.bind("<Button-1>", on_click)
#     c.bind("<B1-Motion>", on_motion)
#     c.bind("<ButtonRelease-1>", on_release)

#     root.mainloop()

#     os.system("./backward.out")


# import tkinter as tk
# from tkinter import filedialog
# import io
# import os
# from PIL import Image, ImageDraw
# import numpy as np

# # Funzione per convertire l'immagine in un vettore
# def image_to_vector(image):
#     image_array = np.array(image)
#     # vector = np.pad(image_array, ((2, 2), (2, 2)), mode='constant', constant_values=0)
#     new_vector = image_array.flatten() / 255.0
#     return new_vector

# def save_image(canvas):
#     file_path = filedialog.asksaveasfilename(defaultextension=".png")
#     if file_path:
#         canvas.postscript(file=file_path + ".eps", colormode='color')
#         canvas.postscript(file=file_path + ".ps", colormode='color')
#         canvas.postscript(file=file_path + ".pdf", colormode='color')
#         canvas.postscript(file=file_path + ".svg", colormode='color')
#         canvas.postscript(file=file_path + ".png", colormode='color')

#     # tmp = canvas.postscript(file="tmp.eps", colormode='color')
#     # fp = open("tmp.eps", "r")
#     # img = Image.open(fp)
#     # close(fp)

#     # img.save("image.png")
#     # # Taglia l'immagine
#     # img = img.resize((28,28))
#     # vector = image_to_vector(img)    

#     # n_values = 0
#     # with open("input_img.txt", "w") as file:
#     #     for val in vector:
#     #         file.write(str(val))
#     #         n_values += 1
#     #         if n_values % 32 == 0:
#     #             file.write("\n")
#     #         else:
#     #             file.write("\t")

# def draw(event):
#     global prev_x
#     global prev_y
#     x, y = event.x, event.y
#     canvas.create_line(prev_x, prev_y, x, y, width=4, fill="white")
#     prev_x = x
#     prev_y = y
#     # canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="black")

# if __name__ == "__main__":

#     prev_x = 0
#     prev_y = 0

#     root = tk.Tk()
#     root.title("Draw and Save")

#     canvas = tk.Canvas(root, width=400, height=400, bg="black")
#     canvas.pack(expand=True, fill="both")
#     canvas.bind("<B1-Motion>", draw)

#     save_button = tk.Button(root, text="Save", command=lambda: save_image(canvas))
#     save_button.pack()

#     root.mainloop()


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps
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

    # Taglia l'immagine
    img = img.resize((28,28), resample=Image.BICUBIC)

    img.save("cropped_img.png")

    vector = image_to_vector(img)

    n_values = 0
    with open("input_img.txt", "w") as file:
        for val in vector:
            file.write("{0:0.2f}".format(val))
            n_values += 1
            if n_values % 28 == 0:
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
        canvas.create_line(prev_x, prev_y, x, y, width=5, fill="black")
    else:
        canvas.create_line(x, y, x, y, width=5, fill="black")
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
    os.system("./backward.out")
