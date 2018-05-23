import os
import imghdr
import shutil

directory = "../data/images/neg_person_map/"

with open(directory + 'person_train_neg.txt', 'a') as the_file:
    for name in os.listdir(directory):
        the_file.write(name.split(".")[0] + " -1\n")
