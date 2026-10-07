# AI-Based PPE Compliance Detection System

**Author:** Adithya S S

## Overview
This application provides a Streamlit interface for uploading an image, running PPE object detection, presenting annotated results, and supporting human PPE review. 

## Application Workflow
1. Upload a clear JPG, JPEG, or PNG image.
2. The YOLO model detects PPE-related objects.
3. Review the annotated image, PPE checklist, and detection confidence.
4. Verify the safety results manually.

## Technology Stack
- Python
- Streamlit
- Ultralytics YOLO
- Pillow, OpenCV, and NumPy

## Run the Application
From the project directory, start Streamlit with the existing environment:

  powershell
.\venv\Scripts\python.exe -m streamlit run app.py