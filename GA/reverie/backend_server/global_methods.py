"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: global_methods.py
Description: Contains functions used throughout my projects.
"""
import random
import string
import csv
import time
import datetime as dt
import pathlib
import os
import sys
import numpy
import math
import shutil, errno

from os import listdir

def create_folder_if_not_there(curr_path):
  """
  Checks if a folder in the curr_path exists. If it does not exist, creates
  the folder.
  Note that if the curr_path designates a file location, it will operate on
  the folder that contains the file. But the function also works even if the
  path designates to just a folder.
  Args:
    curr_list: list to write. The list comes in the following form:
              [['key1', 'val1-1', 'val1-2'...],
                ['key2', 'val2-1', 'val2-2'...],]
    outfile: name of the csv file to write
  RETURNS:
    True: if a new folder is created
    False: if a new folder is not created
  """
  outfolder_name = curr_path.split("/")
  if len(outfolder_name) != 1:
    # This checks if the curr path is a file or a folder.
    if "." in outfolder_name[-1]:
      outfolder_name = outfolder_name[:-1]

    outfolder_name = "/".join(outfolder_name)
    if not os.path.exists(outfolder_name):
      os.makedirs(outfolder_name)
      return True

  return False


def write_list_of_list_to_csv(curr_list_of_list, outfile):
  """
  Writes a list of list to csv.
  Unlike write_list_to_csv_line, it writes the entire csv in one shot.
  ARGS:
    curr_list_of_list: list to write. The list comes in the following form:
              [['key1', 'val1-1', 'val1-2'...],
                ['key2', 'val2-1', 'val2-2'...],]
    outfile: name of the csv file to write
  RETURNS:
    None
  """
  create_folder_if_not_there(outfile)
  with open(outfile, "w") as f:
    writer = csv.writer(f)
    writer.writerows(curr_list_of_list)


def write_list_to_csv_line(line_list, outfile):
  """
  Writes one line to a csv file.
  Unlike write_list_of_list_to_csv, this opens an existing outfile and then
  appends a line to that file.
  This also works if the file does not exist already.
  ARGS:
    curr_list: list to write. The list comes in the following form:
              ['key1', 'val1-1', 'val1-2'...]
              Importantly, this is NOT a list of list.
    outfile: name of the csv file to write
  RETURNS:
    None
  """
  create_folder_if_not_there(outfile)

  # Opening the file first so we can write incrementally as we progress
  curr_file = open(outfile, 'a',)
  csvfile_1 = csv.writer(curr_file)
  csvfile_1.writerow(line_list)
  curr_file.close()


def read_file_to_list(curr_file, header=False, strip_trail=True):
  """
  Reads in a csv file to a list of list. If header is True, it returns a
  tuple with (header row, all rows)
  ARGS:
    curr_file: path to the current csv file.
  RETURNS:
    List of list where the component lists are the rows of the file.
  """
  if not header:
    analysis_list = []
    with open(curr_file) as f_analysis_file:
      data_reader = csv.reader(f_analysis_file, delimiter=",")
      for count, row in enumerate(data_reader):
        if strip_trail:
          row = [i.strip() for i in row]
        analysis_list += [row]
    return analysis_list
  else:
    analysis_list = []
    with open(curr_file) as f_analysis_file:
      data_reader = csv.reader(f_analysis_file, delimiter=",")
      for count, row in enumerate(data_reader):
        if strip_trail:
          row = [i.strip() for i in row]
        analysis_list += [row]
    # Ensure file is not empty before trying to access header/rows
    if not analysis_list:
        return [], [] # Return empty header and rows if file is empty
    return analysis_list[0], analysis_list[1:]


def read_file_to_set(curr_file, col=0):
  """
  Reads in a "single column" of a csv file to a set.
  ARGS:
    curr_file: path to the current csv file.
  RETURNS:
    Set with all items in a single column of a csv file.
  """
  analysis_set = set()
  with open(curr_file) as f_analysis_file:
    data_reader = csv.reader(f_analysis_file, delimiter=",")
    for count, row in enumerate(data_reader):
      if row: # Check if row is not empty
          try:
              analysis_set.add(row[col])
          except IndexError:
              print(f"Warning: Row {count+1} in {curr_file} has fewer than {col+1} columns. Skipping.")
  return analysis_set


def get_row_len(curr_file):
  """
  Get the number of rows in a csv file
  ARGS:
    curr_file: path to the current csv file.
  RETURNS:
    The number of rows
    False if the file does not exist
  """
  try:
    row_count = 0
    with open(curr_file) as f_analysis_file:
      data_reader = csv.reader(f_analysis_file, delimiter=",")
      for row in data_reader:
        row_count += 1
    return row_count # Return count instead of set length
  except:
    return False


def check_if_file_exists(curr_file):
  """
  Checks if a file exists
  ARGS:
    curr_file: path to the current csv file.
  RETURNS:
    True if the file exists
    False if the file does not exist
  """
  return os.path.exists(curr_file) # Use os.path.exists for clarity


def find_filenames(path_to_dir, suffix=".csv"):
  """
  Given a directory, find all files that ends with the provided suffix and
  returns their paths.
  ARGS:
    path_to_dir: Path to the current directory
    suffix: The target suffix.
  RETURNS:
    A list of paths to all files in the directory.
  """
  try:
      filenames = listdir(path_to_dir)
      return [ os.path.join(path_to_dir, filename) # Use os.path.join for cross-platform compatibility
              for filename in filenames if filename.endswith( suffix ) ]
  except FileNotFoundError:
      print(f"Warning: Directory not found: {path_to_dir}")
      return []


def average(list_of_val):
  """
  Finds the average of the numbers in a list.
  ARGS:
    list_of_val: a list of numeric values
  RETURNS:
    The average of the values, or 0 if list is empty.
  """
  if not list_of_val: return 0 # Handle empty list
  return sum(list_of_val)/float(len(list_of_val))


def std(list_of_val):
  """
  Finds the std of the numbers in a list.
  ARGS:
    list_of_val: a list of numeric values
  RETURNS:
    The std of the values, or 0 if list has fewer than 2 elements.
  """
  if len(list_of_val) < 2: return 0 # Std dev requires at least 2 points
  std_dev = numpy.std(list_of_val)
  return std_dev


def copyanything(src, dst):
  """
  Copy over everything in the src folder to dst folder.
  Allows the destination directory to exist.
  ARGS:
    src: address of the source folder
    dst: address of the destination folder
  RETURNS:
    None
  """
  try:
    # Use dirs_exist_ok=True (Python 3.8+) to prevent error if dst exists
    shutil.copytree(src, dst, dirs_exist_ok=True)
  except OSError as exc: # python >2.5
    # This fallback might still be needed for specific file copy errors
    # within copytree, though dirs_exist_ok handles the main directory error.
    if exc.errno in (errno.ENOTDIR, errno.EINVAL):
      shutil.copy(src, dst)
    else: raise


if __name__ == '__main__':
  pass
    