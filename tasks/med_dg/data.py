import json
import os

MED_DG_SET = [
    'Enteritis', 
    'Gastritis', 
    'Gastroenteritis', 
    'Esophagitis',
    'Cholecystitis', 
    'Appendicitis', 
    'Pancreatitis', 
    'Gastric ulcer',
    'Constipation', 
    'Cold', 
    'Irritable bowel syndrome', 
    'Diarrhea',
    'Allergic rhinitis', 
    'Upper respiratory tract infection',
    'Pneumonia'
]

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "MedDG.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data