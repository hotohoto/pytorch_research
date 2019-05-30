# PyTorch research

## Folder Structure

- data
  - names
    - names.txt
  - dictionary
    - words1.txt
    - words2.txt
  - sentences
    - sentences_1.txt
    - sentences_2.txt
  - speech
    - meta.csv
    - raws
      - 1.mp3
      - 2.mp3
- docs
- LICENSE.md
- README.md
- test
  - etl
  - textgen
  - stt
- typer
  - etl
  - textgen
  - stt

## Modules

### ETL

- maybe_download(url, output_name, overwrite=False)
  - url
  - output file or folder
  - overwrite

- DataSourceParser
- MultiDataSource

### Character level text generator

###
