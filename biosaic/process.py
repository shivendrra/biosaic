import os
import pandas as pd
import gzip, shutil
import regex as re
from Bio import SeqIO

def parquet_to_csv(data, path, index=False):
  assert os.path.exists(path), "path doesn't exist!"
  df = pd.read_parquet(data)
  df.to_csv(path, sep=",", index=index)
  print(f"Parquet to CSV conversion success!!")
  print(f"Saved the file to the path: {path}")

def parquet_to_text(data, path, index=False):
  assert os.path.exists(path), "path doesn't exist!"
  df = pd.read_parquet(data)
  df.to_csv(path, sep="\t", index=False)
  print(f"Parquet to CSV conversion success!!")
  print(f"Saved the file to the path: {path}")

def split_file(input_path, output_dir, num_files):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
  with open(input_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
    lines_per_file = total_lines // num_files
    f.seek(0)
        
    for i in range(num_files):
      output_file = os.path.join(output_dir, f"chunk_0{i+1}.txt")
      with open(output_file, "w", encoding="utf-8") as fw:
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
    if file_name.endswith(".gz"):
      print(f"Unzipping: {file_name}")
      with gzip.open(input_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
          shutil.copyfileobj(f_in, f_out)
          print(f"Unzipping complete: {output_path}")
    else:
      print(f"Skipping non-GZip file: {file_name}")

def consolidate(input_dir, output_file):
  files = os.listdir(input_dir)

  with open(output_file, "a", encoding="utf-8") as output_file:
    for file_name in files:
      input_path = os.path.join(input_dir, file_name)

      if file_name.endswith(".txt") and os.path.isfile(input_path):
        print(f"Reading: {file_name}")
        with open(input_path, "r", encoding="utf-8") as input_file:
          output_file.write(input_file.read())
          output_file.write("\n")

        print(f"Reading complete: {file_name}")
      else:
        print(f"Skipping non-text file: {file_name}")

def sanitize_filename(s):
  # keep alphanumeric, dash, underscore; replace others with underscore
  return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")

def merge_sequences(input_fasta, output_file):
  with open(output_file, "w", encoding="utf-8") as out_handle:
    for record in SeqIO.parse(input_fasta, "fasta"):
      out_handle.write(str(record.seq) + "\n")
  print(f"Merged {input_fasta} → {output_file} (raw DNA only)")

def split_sequences(input_fasta, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  for record in SeqIO.parse(input_fasta, "fasta"):
    name = sanitize_filename(record.description)
    path = os.path.join(out_dir, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(str(record.seq) + "\n")
    print(f"Wrote to {path}")

def cleanse_db(input_fasta, action, output_dir="ouptut", merged_file="merged.txt"):
  """ calling function for the whole logic
    Args:
      input_fasta (str, path): Path to input FASTA file
      action (merge, split): merge into one file or split into files
      output_dir (str, path): Directory for split files (default: output)
      merged_file (str): Filename for merged output (default: merged.txt)
  """

  if action == "merge":
    merge_sequences(input_fasta, merged_file)
  else:
    split_sequences(input_fasta, output_dir)