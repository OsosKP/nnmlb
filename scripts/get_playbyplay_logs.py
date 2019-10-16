from urllib.request import urlopen
import os
import zipfile

DATA_DIRECTORY = 'data/retrosheet/playbyplay/'

if not os.path.exists(DATA_DIRECTORY):
    os.mkdir(DATA_DIRECTORY)

for year in range(1919, 2021):
    request = urlopen('https://www.retrosheet.org/events/' +
                      str(year) + 'eve.zip')

    zip_file_pointer = DATA_DIRECTORY + str(year) + '.zip'
    output_folder_pointer = DATA_DIRECTORY + str(year) + 'DD' + '/'

    output = open(zip_file_pointer, 'wb')

    output.write(request.read())
    output.close()

    with zipfile.ZipFile(zip_file_pointer, "r") as zip_ref:
        zip_ref.extractall(output_folder_pointer)

    os.remove(zip_file_pointer)
