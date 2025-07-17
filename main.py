import os
import re
import math
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from Metrics import compute_all_metrics
from langchain_community.llms import Ollama
from collections import Counter


def predict_diseases_from_text(text, llm):
    """
    Step 1: Predict diseases and medical conditions from clinical text.
    """
    # Generate likely symptoms based on code category
    symptom_map = {
        'A': 'fever, malaise, infectious symptoms',
        'B': 'fever, systemic symptoms, infectious manifestations',
        'C': 'mass, weight loss, fatigue, pain',
        'D': 'anemia, bleeding, immune dysfunction',
        'E': 'metabolic symptoms, hormonal changes',
        'F': 'behavioral changes, cognitive symptoms',
        'G': 'neurological symptoms, headache, weakness',
        'H': 'visual/hearing impairment, sensory loss',
        'I': 'chest pain, dyspnea, cardiovascular symptoms',
        'J': 'cough, dyspnea, respiratory symptoms',
        'K': 'abdominal pain, nausea, digestive symptoms',
        'L': 'rash, skin lesions, dermatological symptoms',
        'M': 'joint pain, muscle weakness, mobility issues',
        'N': 'urinary symptoms, reproductive issues',
        'O': 'pregnancy-related symptoms',
        'P': 'neonatal symptoms, developmental issues',
        'Q': 'congenital abnormalities, developmental defects',
        'R': 'general symptoms, vital sign abnormalities',
        'S': 'trauma, injury, external cause effects',
        'T': 'poisoning, adverse effects, complications',
        'Z': 'health maintenance, screening, follow-up'
    }
    
    # Create symptom guidance for the prompt
    symptom_guidance = "\n".join([f"- {category}: {symptoms}" for category, symptoms in symptom_map.items()])
    
    prompt = f"""
    You are an expert medical professional. Analyze the clinical text below and identify all diseases, medical conditions, and diagnoses mentioned or implied.

    INSTRUCTIONS:
    - List each disease/condition on a separate line
    - Use standard medical terminology
    - Include both primary and secondary conditions
    - Be specific (e.g., "Acute myocardial infarction" not just "heart problem")
    - If no clear diseases are identified, output: None

    SYMPTOM GUIDANCE by ICD-10 Category:
    {symptom_guidance}

    Use this symptom mapping to help identify potential diseases when symptoms are mentioned in the text. Consider which ICD-10 categories the symptoms might belong to and identify corresponding diseases.
    
    IMPORTANT: Do NOT write explanations, reasoning, or your thinking process. Output ONLY the disease names

    Clinical Text:
    {text}

    Diseases/Conditions Identified:
    """
    
    response = llm.invoke(prompt, temperature=0.1).strip()
    
    # Extract diseases from response
    diseases = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('Diseases/Conditions') and line != 'None':
            # Clean formatting (remove bullets, numbers, etc.)
            clean_line = re.sub(r'^[-•*\d+\.\)]\s*', '', line).strip()
            if clean_line and clean_line not in diseases:
                diseases.append(clean_line)
    
    return diseases if diseases else []