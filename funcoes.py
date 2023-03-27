import os
import time
import pandas as pd
#----------#----------#----------#----------#----------##----------#----------#----------#----------#----------#
import importlib
import subprocess
def try_import(module_name,how='',alias=''):
  if how == '':
    try:
      if alias == '':
        print(f'import {module_name}')
        globals()[module_name] = importlib.import_module(module_name)
      else:
        print(f'import {module_name} as {alias}')
        globals()[alias] = importlib.import_module(module_name)
    except Exception as e:
      print(e)
      try:
        subprocess.check_call(["pip", "install", module_name]) #!pip install {module_name}
        print(f'{module_name} installed successfully')
        if alias == '':
          print(f'import {module_name}')
          globals()[module_name] = importlib.import_module(module_name)
        else:
          print(f'import {module_name} as {alias}')
          globals()[alias] = importlib.import_module(module_name)
      except Exception as e:
        print(e)

  elif how == '*':
    try:
      print(f'from {module_name} import *')
      modules = importlib.import_module(module_name)
      globals().update(vars(modules))
    except Exception as e:
      print(e)
      subprocess.check_call(["pip", "install", module_name]) #!pip install {module_name}
      print(f'{module_name} installed successfully')
    finally:
      print(f'from {module_name} import *')
      modules = importlib.import_module(module_name)
      globals().update(vars(modules))
  else:
    print('Nothing to import!')
#----------#----------#----------#----------#----------##----------#----------#----------#----------#----------#
from IPython.display import Audio, display
try:
  try_import('chime')
  chime.theme('pokemon')
except Exception as e:
  print(f'Não foi possivel carregar modulo chime: {e}')
  
def success():
  try:
    chime.success(sync=True,raise_error=True)
  except Exception as e:
    display(Audio('files/themes_pokemon_success.wav', autoplay=True))
#----------#----------#----------#----------#----------##----------#----------#----------#----------#----------#
def read_and_create_feather(path_file,sep=','):
  ini = time.perf_counter()
  path = '/'.join(path_file.split('/')[:-1])+'/'
  if path == '/':
    path = './'
  filename = path_file.split('/')[-1]
  file_feather = f'{filename.split(".")[0]}.feather'
  extension = filename.split('.')[-1]
  try:
    base = pd.read_feather(file_feather)
    print(f'Reading {path}{file_feather} instead')
  except Exception as e:
    if extension == 'csv':
      base = pd.read_csv(path_file,sep=sep)
    elif extension == 'xlsx':
      base = pd.read_excel(path_file)
    else:
      print(f'Extension: {extension}. (.csv) or (.xlsx) only.')
      return
    base.to_feather(f'{path}{file_feather}')
    print(f'{path}{file_feather} created.')
  return base
#----------#----------#----------#----------#----------##----------#----------#----------#----------#----------#