import os
import imghdr
import shutil

directory = "../data/images/neg_person_map"

i = 900000

for name in os.listdir(directory):
    # combining file+its directory:
    file = directory+"/"+name
    file = directory+"/"+name
    # fetch the file extension from the file:
    ftype = imghdr.what(file)
    # if it is impossible to get the extension (if the file is damaged for example), the file(s) will be listed in the terminal window:
    if ftype != None:
        print(i)
        shutil.move(file, directory + "/" + str(i) + "." + ftype)
        i += 1
    else:
        print("could not determine: " + file)
