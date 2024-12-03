### importing libraries
import subprocess
import importlib
import os
import re

package_lib = ['torch', 
              'cv2', 
              'pytesseract', 
              'pdfminer.six', 
              'transformers', 
              'scikit-learn',
              'numpy',
              'flask']

def install_in_error():
    for lib in package_lib:
        try:
            subprocess.check_call(['pip', 'install', lib])
        except subprocess.CalledProcessError as e:
            print(f"Exeption found {e}")

try:
    import torch
    import cv2
    import pytesseract
    from pdfminer.high_level import extract_text
    from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from flask import Flask, render_template, request, jsonify
except BaseException as e:
    print(f"Exeption found {e}")
