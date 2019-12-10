import PIL.Image
import os

for (dirpath, dirnames, filenames) in os.walk("resources\\lucy"):
    for file in filenames:
        pre, ext = os.path.splitext(file)
        if(ext == ".png"):
            im = PIL.Image.open(dirpath + "\\" + file)
            rgb_im = im.convert('RGB')
            rgb_im.save(dirpath + "\\" + pre + '.jpg')
    break

# # pre, ext = os.path.splitext(renamee)
# # os.rename(renamee, pre + new_extension)

# im = Image.open("Ba_b_do8mag_c6_big.png")
# rgb_im = im.convert('RGB')
# rgb_im.save('colors.jpg')