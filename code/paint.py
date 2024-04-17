import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from wand.image import Image as WandImage

# Dimensioni del canvas
canvas_width = 200
canvas_height = 200

# Dimensioni dell'immagine
image_size = (28, 28)

# Inizializzazione della finestra tkinter
root = tk.Tk()
root.title("Disegna e converte in vettore")

# Funzione per convertire l'immagine in un vettore
def image_to_vector(image):
    resized_image = image.resize(image_size).convert('L')
    image_array = np.array(resized_image)
    vector = np.pad(image_array, ((2, 2), (2, 2)), mode='constant', constant_values=0)
    new_vector = vector.flatten() / 255.0
    return new_vector

# Variabile per memorizzare il punto iniziale del tratto
prev_x, prev_y = None, None

# Funzione chiamata quando il mouse viene premuto
def on_click(event):
    global prev_x, prev_y
    prev_x, prev_y = event.x, event.y

# Funzione chiamata quando il mouse viene rilasciato
def on_release(event):
    global prev_x, prev_y
    prev_x, prev_y = None, None
    save_image()

# Funzione chiamata quando il mouse si muove
def on_motion(event):
    global prev_x, prev_y
    if prev_x is not None and prev_y is not None:
        x, y = event.x, event.y
        c.create_line(prev_x, prev_y, x, y, width=5, fill='white')
        prev_x, prev_y = x, y

# Funzione per salvare l'immagine disegnata come vettore
def save_image():
    global vector
    # Salva il canvas come file PostScript
    c.postscript(file="temp.ps", colormode='mono')
    # Converti il file PostScript in un'immagine PNG usando wand
    with WandImage(filename="temp.ps", resolution=300) as img:
        img.format = 'png'
        img.background_color = 'black'
        img.alpha_channel = 'remove'
        img.save(filename="temp.png")
    # Apri l'immagine convertita
    img = Image.open("temp.png")
    # Converte l'immagine in un vettore
    vector = image_to_vector(img)
    print("Vettore salvato:", vector)
    # Scrivi il vettore su file
    n_values = 0
    with open("draw_0.txt", "w") as file:
        for val in vector:
            file.write(str(val))
            n_values += 1
            if n_values % 32 == 0:
                file.write("\n")
            else:
                file.write("\t")

    # Taglia l'immagine
    img_cropped = img.resize((28,28))
    img_cropped.save("cropped_image.png")

# Creazione del canvas
c = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='black')
c.pack()

# Bind degli eventi del mouse
c.bind("<Button-1>", on_click)
c.bind("<B1-Motion>", on_motion)
c.bind("<ButtonRelease-1>", on_release)

# def main():
#     c = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')

# if __name__ == "__main__" :
#     main()

root.mainloop()