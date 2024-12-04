### importing libraries
import subprocess
import importlib
import os
import re
import pickle

package_lib = ['torch', 
              'cv2', 
              'pytesseract', 
              'pdfminer.six', 
              'transformers', 
              'scikit-learn',
              'numpy',
              'flask',
              'datasets',
              'faiss-cpu',
              'elasticsearch',
              'python-docx']

# for lib in package_lib:
#     try:
#         subprocess.check_call(['pip', 'install', lib])
#     except subprocess.CalledProcessError as e:
#         print(f"Exeption found {e}")

try:
    import torch
    import cv2
    import pytesseract
    from pdfminer.high_level import extract_text
    from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from flask import Flask, render_template, request, jsonify
    import datasets
    import faiss
    from elasticsearch import Elasticsearch
    from docx import Document

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    os.environ['TRUST_REMOTE_CODE'] = 'True'
except BaseException as e:
    print(f"Exeption found {e}")

### create a pcgs_w100.tsv.pkl file with the correct data
data = {
    "document1": "content1",
    "document2": "content2",
    "document3": "content3"
}

with open("local_dataset/psgs_w100.tsv.pkl", "wb") as f:
    pickle.dump(data, f)
