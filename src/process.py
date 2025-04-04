import os
import pandas as pd
import gzip, shutil

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

def parquet_to_csv(data, path, index=False):
  assert os.path.exists(path), "path doesn't exist!"
  df = pd.read_parquet(data)
  df.to_csv(path, sep=',', index=index)
  print(f"Parquet to CSV conversion success!!")
  print(f"Saved the file to the path: {path}")

def parquet_to_text(data, path, index=False):
  assert os.path.exists(path), "path doesn't exist!"
  df = pd.read_parquet(data)
  df.to_csv(path, sep='\t', index=False)
  print(f"Parquet to CSV conversion success!!")
  print(f"Saved the file to the path: {path}")

def split_file(input_path, output_dir, num_files):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
  with open(input_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)
    lines_per_file = total_lines // num_files
    f.seek(0)
        
    for i in range(num_files):
      output_file = os.path.join(output_dir, f'chunk_0{i+1}.txt')
      with open(output_file, 'w', encoding='utf-8') as fw:
        lines_written = 0
        while lines_written < lines_per_file:
          line = f.readline()
          if not line:
            break
          fw.write(line)
          lines_written += 1

def unzip(input_directory, output_directory):
  os.makedirs(output_directory, exist_ok=True)
  files = os.listdir(input_directory)

  for file_name in files:
    input_path = os.path.join(input_directory, file_name)
    output_path = os.path.join(output_directory, os.path.splitext(file_name)[0])
    if file_name.endswith('.gz'):
      print(f"Unzipping: {file_name}")
      with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)
          print(f"Unzipping complete: {output_path}")
    else:
      print(f"Skipping non-GZip file: {file_name}")

def consolidate(input_dir, output_file):
  files = os.listdir(input_dir)

  with open(output_file, 'a', encoding='utf-8') as output_file:
    for file_name in files:
      input_path = os.path.join(input_dir, file_name)

      if file_name.endswith('.txt') and os.path.isfile(input_path):
        print(f"Reading: {file_name}")
        with open(input_path, 'r', encoding='utf-8') as input_file:
          output_file.write(input_file.read())
          output_file.write('\n')

        print(f"Reading complete: {file_name}")
      else:
        print(f"Skipping non-text file: {file_name}")
