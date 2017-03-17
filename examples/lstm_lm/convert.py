import os
import text_to_hdf5

FILE_DIR='./../../reddit_data/'
if __name__=="__main__":
	dg=text_to_hdf5.DataGenerator()

for file in os.listdir(FILE_DIR):
	if file.endswith(".txt"):
		dg.data_to_hdf5(FILE_DIR+file)
