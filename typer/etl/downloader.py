"""Download Utils
"""

import os
import os.path
import tarfile
import urllib.request
import zipfile

def reporthook(count, block_size, total_size):
    if count == 0:
        return
    if count * block_size >= total_size:
        print("")
        return
    if count % 100 != 0:
        return
    print(".", end="")


def maybe_download(url, destination_path):
    if os.path.exists(destination_path):
        return

    dirpath, _ = os.path.split(destination_path)
    os.makedirs(dirpath, exist_ok=True)
    urllib.request.urlretrieve(url, destination_path, reporthook)


def reduce_folder_depth(folder_path):
    _, folder_name = os.path.split(folder_path)
    files = os.listdir(folder_path)
    if len(files) == 1 and files[0] == folder_name:
        tmp_dir_path = os.path.join(folder_path, ".__tmp_dir__")
        os.rename(os.path.join(folder_path, files[0]), tmp_dir_path)
        innerfiles = os.listdir(tmp_dir_path)
        for f in innerfiles:
            os.rename(os.path.join(tmp_dir_path, f), os.path.join(folder_path, f))
        os.rmdir(tmp_dir_path)


def maybe_unzip(source_zip_file_path):
    dirpath, filename = os.path.split(source_zip_file_path)
    name, _ = os.path.splitext(filename)
    destination_folder = os.path.join(dirpath, name)
    if os.path.exists(destination_folder):
        return destination_folder
    if destination_folder and not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    with zipfile.ZipFile(source_zip_file_path) as z:
        z.extractall(destination_folder)
    reduce_folder_depth(destination_folder)
    return destination_folder


def maybe_extract_tar(source_tar_file_path):
    dirpath, filename = os.path.split(source_tar_file_path)
    name, ext = os.path.splitext(filename)
    destination_folder = os.path.join(dirpath, name)

    if os.path.exists(destination_folder):
        return destination_folder
    if destination_folder and not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    with tarfile.open(source_tar_file_path) as t:
        t.extractall(destination_folder)
    reduce_folder_depth(destination_folder)
    return destination_folder
