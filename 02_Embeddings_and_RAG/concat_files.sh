#!/bin/bash

# Check if at least one directory is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/dir1 [/path/to/dir2 ...]"
  exit 1
fi

# Output file
output_file="prompt.txt"
> "$output_file"  # Clear or create

# Loop through all provided directories
for folder in "$@"; do
  if [ ! -d "$folder" ]; then
    echo "Warning: '$folder' is not a valid directory. Skipping..."
    continue
  fi

  echo "Processing directory: $folder"

  # Find and sort .py files (excluding __init__.py)
  py_files=$(find "$folder" -type f -name "*.py" ! -name "__init__.py" | sort)

  for file in $py_files; do
    relative_path="${file#"$PWD"}"
    echo -e "File: $relative_path" >> "$output_file"
    cat "$file" >> "$output_file"
    echo -e "\n\n" >> "$output_file"
  done
done

echo "Concatenated Python files saved to '$output_file'"
