
# # download the mit fivek dataset for expert C from this link and put it in dataset/
# # https://www.dropbox.com/sh/web5of2dswd55b3/AABs5xY3V1CXEzfGWzBw9OUQa?dl=0&preview=C.zip

# # extract the zip file
# unzip dataset/C.zip -d dataset/

# # remove the zip file
# rm dataset/C.zip

# create train val test splits
python list_images.py --datasetdir dataset/C 


