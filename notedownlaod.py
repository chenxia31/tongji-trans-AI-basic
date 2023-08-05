import requests
import os

pdf_urls = [
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note01.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note02.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note03.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note04.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note05.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note06.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note07.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note08.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note09.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note10.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note11.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note12.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note13.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note14.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note15.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note16.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note17.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note18.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note19.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note20.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note21.pdf',
  'https://inst.eecs.berkeley.edu/~cs188/fa22/assets/notes/cs188-fa22-note22.pdf'
]

for url in pdf_urls:
  filename = url.split('/')[-1]
  r = requests.get(url)
  with open(filename, 'wb') as f:
    f.write(r.content)
  print(f"Downloaded {filename}")