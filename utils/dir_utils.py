"""
 @author   Maksim Penkin
"""


import os
import shutil


def delete_file_folder(f):
    try:
        if os.path.isfile(f) or os.path.islink(f):
            os.unlink(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (f, e))


def delete_contents_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        delete_file_folder(file_path)


def create_folder(folder, force=False, raise_except_if_exists=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        if force:
            delete_contents_folder(folder)
        else:
            if raise_except_if_exists:
                raise Exception('utils/dir_utils.py: '
                                'def create_folder(...): '
                                'error: directory {} exists. In order to overwrite it set force=True'.format(folder))
