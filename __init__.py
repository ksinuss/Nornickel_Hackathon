### importing libraries
import pip._internal as pip
import importlib
import os
import re

package_lib = ['torch', 
              'cv2', 
              'pytesseract', 
              'pdfminer.high_level', 
              'transformers', 
              'sklearn.metrics.pairwise',
              'numpy']

for lib in package_lib:
    try:
        pip.main(['install', lib])
    except BaseException as e:
        print(f"Exeption found {e}")

try:
    import torch
    import cv2
    import pytesseract
    from pdfminer.high_level import extract_text
    from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except BaseException as e:
    print(f"Exeption found {e}")