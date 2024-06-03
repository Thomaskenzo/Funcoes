import importlib
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
        !pip install {module_name}
        if alias == '':
          print(f'import {module_name}')
          globals()[module_name] = importlib.import_module(module_name)
        else:
          print(f'import {module_name} as {alias}')
          globals()[alias] = importlib.import_module(module_name)
      except Exception as e:
        print(e)

  elif how == '*':
    print(f'from {module_name} import *')
    modules = importlib.import_module(module_name)
    globals().update(vars(modules))
  else:
    print('Nothing to import!')