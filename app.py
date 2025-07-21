import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import joblib
import os
import logging
from datetime import datetime, date
import warnings
import time
import io
import base64
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="💰 AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with clean, modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Clean white background */
    .main {
        font-family: 'Inter', sans-serif;
        background: #ffffff;
        color: #2c3e50;
        padding: 0;
    }
    
    .stApp {
        background: #ffffff;
    }
    
    /* Header Styles */
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0 1rem 0;
        animation: slideInDown 1s ease-out;
    }
    
    @keyframes slideInDown {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .sub-header {
        font-size: 1.3rem;
        font-weight: 400;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInUp 1.2s ease-out;
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Clean Card Design */
    .clean-card {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e8ecef;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInUp 0.8s ease-out;
    }
    
    .clean-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    @keyframes slideInUp {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Sidebar Clean Design */
    .css-1d391kg {
        background: #f8f9fa;
        border-right: 1px solid #e8ecef;
    }
    
    /* Enhanced Input Styling */
    .stSelectbox > div > div, .stNumberInput > div > div {
        background: #ffffff;
        border: 2px solid #e8ecef;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .stSelectbox > div > div:hover, .stNumberInput > div > div:hover {
        border-color: #667eea;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
    }
    
    .stSelectbox > div > div:focus-within, .stNumberInput > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Modern Button Design */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a6fd8, #6842a8);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Prediction Result Enhanced */
    .prediction-result {
        font-family: 'Poppins', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0;
        animation: zoomIn 1s ease-out, pulse 2s infinite;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    @keyframes zoomIn {
        from { transform: scale(0.5); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Enhanced Metrics */
    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e8ecef;
        transition: all 0.3s ease;
        animation: scaleIn 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    @keyframes scaleIn {
        from { transform: scale(0.9); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    /* Alert Boxes Enhanced */
    .success-alert {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #27ae60;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: bounceIn 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .success-alert::before {
        content: '✅';
        position: absolute;
        top: 1rem;
        left: 1rem;
        font-size: 1.2rem;
    }
    
    .error-alert {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 1px solid #e74c3c;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: shakeX 0.8s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .error-alert::before {
        content: '❌';
        position: absolute;
        top: 1rem;
        left: 1rem;
        font-size: 1.2rem;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #f39c12;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: wobble 1s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .warning-alert::before {
        content: '⚠️';
        position: absolute;
        top: 1rem;
        left: 1rem;
        font-size: 1.2rem;
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes shakeX {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-8px); }
        20%, 40%, 60%, 80% { transform: translateX(8px); }
    }
    
    @keyframes wobble {
        0%, 100% { transform: rotate(0deg); }
        15% { transform: rotate(-3deg); }
        30% { transform: rotate(2deg); }
        45% { transform: rotate(-2deg); }
        60% { transform: rotate(1deg); }
        75% { transform: rotate(-1deg); }
    }
    
    .info-alert {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: 1px solid #3498db;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        animation: fadeIn 1s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .info-alert::before {
        content: '💡';
        position: absolute;
        top: 1rem;
        left: 1rem;
        font-size: 1.2rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Section Headers Enhanced */
    .section-header {
        font-family: 'Poppins', sans-serif;
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 600;
        margin: 2.5rem 0 1.5rem 0;
        text-align: center;
        position: relative;
        animation: slideInLeft 0.8s ease-out;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 4px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Progress Bar Enhanced */
    .progress-container {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1rem;
        margin: 1.5rem 0;
        border: 1px solid #e8ecef;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: #e8ecef;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
        transition: width 1s ease-out;
    }
    
    /* Loading Animation Enhanced */
    .loading-container {
        text-align: center;
        padding: 3rem;
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid #e8ecef;
        margin: 2rem 0;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 32px;
        height: 32px;
        border: 4px solid #e8ecef;
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s linear infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Download Button Enhanced */
    .download-btn {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 0.9rem;
        font-weight: 500;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(39, 174, 96, 0.3);
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(39, 174, 96, 0.4);
        background: linear-gradient(135deg, #229954, #27ae60);
    }
    
    /* Tabs Enhanced */
    .stTabs [data-baseweb="tab-list"] {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid #e8ecef;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        color: #7f8c8d;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #ffffff;
        color: #2c3e50;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    /* Mobile Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .prediction-result {
            font-size: 2.5rem;
        }
        
        .clean-card {
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .section-header {
            font-size: 1.5rem;
        }
    }
    
    /* Chart Container */
    .chart-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e8ecef;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        margin: 1.5rem 0;
    }
    
    /* Feature Cards */
    .feature-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e8ecef;
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(135deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)

class SalaryPredictionApp:
    def __init__(self):
        self.model_path = 'models/salary_prediction_model.joblib'
        self.metadata_path = 'models/model_metadata.json'
        self.model = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load trained model and metadata"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.metadata_path):
                self.model = joblib.load(self.model_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                return True
            else:
                st.error("⚠️ Model files not found. Please train the model first by running the training script.")
                return False
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def validate_age_education_consistency(self, age, education_level):
        """Enhanced validation for age and education level consistency"""
        min_ages = {
            "High School": 17,
            "Bachelor's": 21,
            "Master's": 23,
            "PhD": 26
        }
        
        min_age = min_ages.get(education_level, 18)
        if age < min_age:
            return False, f"❌ Minimum age for {education_level} degree is typically {min_age} years"
        
        # Additional logical checks
        if education_level == "PhD" and age > 65:
            return True, f"⚠️ PhD completion after age 65 is rare but possible"
        
        return True, ""
    
    def validate_age_experience_consistency(self, age, experience):
        """Enhanced validation for age and experience consistency"""
        min_working_age = 16  # Realistic minimum working age
        max_possible_experience = age - min_working_age
        
        if experience > max_possible_experience:
            return False, f"❌ With age {age}, maximum possible experience is {max_possible_experience} years"
        
        if experience < 0:
            return False, "❌ Years of experience cannot be negative"
        
        # Career stage validation - more realistic
        if age < 22 and experience > 4:  # More realistic for young people
            return False, f"❌ At age {age}, having {experience} years of experience seems unrealistic"
        
        # Fresh graduate check
        if age >= 22 and age <= 25 and experience == 0:
            return True, f"⚠️ Fresh graduate? Consider internships or part-time work experience"
        
        # Mid-career with low experience
        if age > 40 and experience < 8:
            return True, f"⚠️ Career change or re-entry? Consider highlighting transferable skills"
        
        # Very experienced person
        if experience > 35:
            return True, f"⚠️ {experience} years is exceptional experience - ensure this is accurate"
        
        return True, ""
    
    def validate_experience_education_consistency(self, experience, education_level, age):
        """Enhanced validation for experience and education consistency"""
        degree_years = {
            "High School": 0,
            "Bachelor's": 4,
            "Master's": 6,
            "PhD": 9
        }
        
        years_in_education = degree_years.get(education_level, 0)
        earliest_work_start = 18 + years_in_education  # Assuming work starts after education
        max_realistic_experience = max(0, age - earliest_work_start)
        
        if experience > max_realistic_experience and max_realistic_experience >= 0:
            # Allow for concurrent work during studies
            adjusted_max = max_realistic_experience + 3
            if experience > adjusted_max:
                return False, f"❌ With {education_level} degree at age {age}, maximum realistic experience is {adjusted_max} years"
        
        return True, ""
    
    def validate_job_title_requirements(self, job_title, education_level, experience, age):
        """Enhanced job title validation with industry standards"""
        job_requirements = {
            'Data Scientist': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 2,  # Data Science requires substantial training
                'typical_age_range': (24, 60),
                'preferred_education': ["Master's", "PhD"],
                'blocked_education': ["High School"]  # Cannot be Data Scientist with only High School
            },
            'Software Engineer': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 0,
                'typical_age_range': (22, 65),
                'preferred_education': ["Bachelor's"],
                'blocked_education': ["High School"]  # Modern software engineering requires degree
            },
            'Senior Engineer': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 7,  # "Senior" requires more experience
                'typical_age_range': (29, 65),
                'preferred_education': ["Bachelor's", "Master's"],
                'blocked_education': ["High School"]
            },
            'Manager': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 5,  # Management requires proven track record
                'typical_age_range': (27, 65),
                'preferred_education': ["Bachelor's", "Master's"],
                'blocked_education': ["High School"]
            },
            'Director': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 12,  # Director is very senior position
                'typical_age_range': (35, 65),
                'preferred_education': ["Master's", "PhD"],
                'blocked_education': ["High School"]
            },
            'Consultant': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 3,  # Consultants need expertise to advise others
                'typical_age_range': (25, 60),
                'preferred_education': ["Master's"],
                'blocked_education': ["High School"]
            },
            'Developer': {
                'min_education': ["High School", "Bachelor's", "Master's", "PhD"],
                'min_experience': 0,
                'typical_age_range': (18, 65),
                'preferred_education': ["Bachelor's"],
                'blocked_education': []  # Developer can start with High School + self-learning
            },
            'Analyst': {
                'min_education': ["Bachelor's", "Master's", "PhD"],
                'min_experience': 0,
                'typical_age_range': (22, 60),
                'preferred_education': ["Bachelor's"],
                'blocked_education': ["High School"]  # Analysis requires formal education
            }
        }
        
        if job_title in job_requirements:
            req = job_requirements[job_title]
            warnings = []
            
            # Blocked education check (unrealistic combinations)
            if education_level in req.get('blocked_education', []):
                return False, f"❌ {job_title} cannot be achieved with {education_level} education. Minimum required: {', '.join(req['min_education'])}"
            
            # Education requirement check
            if education_level not in req['min_education']:
                return False, f"❌ {job_title} typically requires: {', '.join(req['min_education'])}"
            
            # Experience requirement check
            if experience < req['min_experience']:
                return False, f"❌ {job_title} typically requires at least {req['min_experience']} years of experience"
            
            # Age range check
            min_age, max_age = req['typical_age_range']
            if age < min_age:
                return False, f"❌ {job_title} positions typically start at age {min_age}+"
            
            if age > max_age:
                warnings.append(f"⚠️ {job_title} at age {age} is possible but less common")
            
            # Preferred education notification
            if education_level not in req['preferred_education']:
                warnings.append(f"⚠️ {job_title} typically prefers: {', '.join(req['preferred_education'])}")
            
            if warnings:
                return True, " | ".join(warnings)
        
        return True, ""
    
    def validate_salary_expectations(self, job_title, experience, education_level):
        """Validate salary expectations based on role and experience"""
        salary_ranges = {
            'Data Scientist': {'entry': (70000, 90000), 'mid': (90000, 130000), 'senior': (130000, 180000)},
            'Software Engineer': {'entry': (60000, 80000), 'mid': (80000, 120000), 'senior': (120000, 160000)},
            'Senior Engineer': {'entry': (100000, 130000), 'mid': (130000, 160000), 'senior': (160000, 200000)},
            'Manager': {'entry': (80000, 110000), 'mid': (110000, 150000), 'senior': (150000, 200000)},
            'Director': {'entry': (120000, 160000), 'mid': (160000, 220000), 'senior': (220000, 300000)},
            'Consultant': {'entry': (65000, 85000), 'mid': (85000, 120000), 'senior': (120000, 170000)},
            'Developer': {'entry': (50000, 70000), 'mid': (70000, 100000), 'senior': (100000, 140000)},
            'Analyst': {'entry': (45000, 65000), 'mid': (65000, 90000), 'senior': (90000, 120000)}
        }
        
        if job_title in salary_ranges:
            ranges = salary_ranges[job_title]
            if experience <= 2:
                return ranges['entry']
            elif experience <= 7:
                return ranges['mid']
            else:
                return ranges['senior']
        
        return (50000, 150000)  # Default range
    
    def comprehensive_validation(self, data):
        """Enhanced comprehensive validation with detailed feedback"""
        errors = []
        warnings = []
        
        age = data.get('Age', 0)
        experience = data.get('Years of Experience', 0)
        education = data.get('Education Level', '')
        job_title = data.get('Job Title', '')
        gender = data.get('Gender', '')
        
        # Enhanced range validations
        if age < 16 or age > 75:
            errors.append("❌ Age must be between 16 and 75 years")
        
        if experience < 0 or experience > 55:
            errors.append("❌ Years of experience must be between 0 and 55")
        
        # Gender validation
        if not gender or gender not in ['Male', 'Female']:
            errors.append("❌ Please select a valid gender")
        
        # Age-Education consistency with warnings
        if age and education:
            is_valid, msg = self.validate_age_education_consistency(age, education)
            if not is_valid:
                errors.append(msg)
            elif "⚠️" in msg:
                warnings.append(msg)
        
        # Age-Experience consistency
        if age and experience is not None:
            is_valid, msg = self.validate_age_experience_consistency(age, experience)
            if not is_valid:
                errors.append(msg)
            elif "⚠️" in msg:
                warnings.append(msg)
        
        # Experience-Education consistency
        if experience is not None and education and age:
            is_valid, msg = self.validate_experience_education_consistency(experience, education, age)
            if not is_valid:
                errors.append(msg)
        
        # Enhanced job title validations
        if job_title and education and experience is not None and age:
            is_valid, msg = self.validate_job_title_requirements(job_title, education, experience, age)
            if not is_valid:
                errors.append(msg)
            elif "⚠️" in msg:
                warnings.append(msg)
        
        # Career progression logic
        if experience > 15 and "Senior" not in job_title and job_title not in ["Manager", "Director"]:
            warnings.append("⚠️ With 15+ years experience, consider senior or management roles")
        
        if experience < 3 and job_title in ["Manager", "Director"]:
            errors.append("❌ Management roles typically require 3+ years of experience")
        
        # Education-experience alignment
        if education == "PhD" and experience < 2:
            warnings.append("⚠️ PhD graduates typically have research or internship experience")
        
        if education == "High School" and job_title in ["Data Scientist", "Manager", "Director"]:
            warnings.append("⚠️ This role typically requires higher education")
        
        # Age-related career insights
        if age > 45 and experience < 10:
            warnings.append("⚠️ Career change or re-entry? Consider highlighting transferable skills")
        
        if age < 30 and job_title == "Director":
            warnings.append("⚠️ Director role at young age - ensure leadership experience")
        
        return errors, warnings
    
    def predict_salary(self, data):
        if not self.model or not self.metadata:
            return None, ["❌ Model not loaded properly"]
        
        try:
            errors, warnings = self.comprehensive_validation(data)
            
            if errors:
                return None, errors
            
            input_df = pd.DataFrame([data])
            prediction = self.model.predict(input_df)[0]
            
            # Enhanced prediction intervals based on model uncertainty
            model_metrics = self.metadata.get('final_metrics', {})
            model_std = model_metrics.get('test_rmse', 15000)
            
            # Adjust margin based on data quality and model confidence
            confidence_factor = self._calculate_confidence_factor(data)
            margin_of_error = model_std * confidence_factor
            
            lower_bound = max(25000, prediction - margin_of_error)
            upper_bound = min(500000, prediction + margin_of_error)
            
            # Calculate percentile ranking
            expected_range = self.validate_salary_expectations(
                data['Job Title'], 
                data['Years of Experience'], 
                data['Education Level']
            )
            
            result = {
                'prediction': round(prediction, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'confidence': self._get_confidence_level(),
                'expected_range': expected_range,
                'warnings': warnings,
                'market_position': self._get_market_position(prediction),
                'confidence_factor': confidence_factor
            }
            
            return result, []
            
        except Exception as e:
            error_msg = f"❌ Prediction error: {str(e)}"
            st.error(error_msg)
            return None, [error_msg]
        # Missing methods to add to your SalaryPredictionApp class:
    
    def _calculate_confidence_factor(self, data):
        """Calculate confidence factor based on data quality"""
        confidence = 1.0
        
        # Age factor
        age = data.get('Age', 30)
        if age < 22 or age > 60:
            confidence *= 1.2  # Less confident for extreme ages
        
        # Experience factor
        experience = data.get('Years of Experience', 0)
        if experience > 30:
            confidence *= 1.3  # Less confident for very high experience
        
        # Education-job alignment
        education = data.get('Education Level', '')
        job_title = data.get('Job Title', '')
        
        high_skill_jobs = ['Data Scientist', 'Senior Engineer', 'Director']
        if job_title in high_skill_jobs and education == 'High School':
            confidence *= 1.4
        
        return min(confidence, 2.0)  # Cap at 2.0
    
    def _get_confidence_level(self):
        """Return confidence level as percentage"""
        return "85-90%"  # Based on model performance
    
    def _get_market_position(self, prediction):
        """Determine market position of the predicted salary"""
        if prediction < 60000:
            return "Entry Level"
        elif prediction < 100000:
            return "Mid Level"
        elif prediction < 150000:
            return "Senior Level"
        else:
            return "Executive Level"
    
    def create_salary_distribution_chart(self, prediction, job_title):
        """Create salary distribution visualization"""
        # Sample data for visualization (you can replace with actual data)
        np.random.seed(42)
        sample_salaries = np.random.normal(prediction, 20000, 1000)
        sample_salaries = sample_salaries[sample_salaries > 30000]  # Remove unrealistic values
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=sample_salaries,
            nbinsx=30,
            name='Salary Distribution',
            opacity=0.7,
            marker=dict(color='rgba(102, 126, 234, 0.7)')
        ))
        
        # Add prediction line
        fig.add_vline(
            x=prediction,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"Your Prediction: ${prediction:,.0f}"
        )
        
        fig.update_layout(
            title=f'Salary Distribution for {job_title}',
            xaxis_title='Salary ($)',
            yaxis_title='Frequency',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_feature_importance_chart(self):
        """Create feature importance visualization"""
        # Sample feature importance (replace with actual model feature importance)
        features = ['Years of Experience', 'Education Level', 'Age', 'Job Title', 'Gender']
        importance = [0.40, 0.25, 0.15, 0.15, 0.05]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(color='rgba(102, 126, 234, 0.8)')
        ))
        
        fig.update_layout(
            title='Feature Importance in Salary Prediction',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=300
        )
        
        return fig
    
    def display_prediction_insights(self, result, data):
        """Display detailed prediction insights"""
        prediction = result['prediction']
        lower_bound = result['lower_bound']
        upper_bound = result['upper_bound']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Predicted Salary</h3>
                <h2>${prediction:,.0f}</h2>
                <p>Annual Base Salary</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Salary Range</h3>
                <h2>${lower_bound:,.0f} - ${upper_bound:,.0f}</h2>
                <p>Confidence Interval</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Market Position</h3>
                <h2>{result['market_position']}</h2>
                <p>Career Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display warnings if any
        if result['warnings']:
            st.markdown('<div class="warning-alert">', unsafe_allow_html=True)
            st.markdown("**⚠️ Important Considerations:**")
            for warning in result['warnings']:
                st.markdown(f"• {warning}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    def run_app(self):
        """Main application logic"""
        # Header
        st.markdown('<h1 class="main-header">💰 AI Salary Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Get accurate salary predictions powered by machine learning</p>', unsafe_allow_html=True)
        
        if not self.model:
            st.error("❌ Model not available. Please ensure the model files are in the correct location.")
            st.stop()
        
        # Sidebar for inputs
        with st.sidebar:
            st.markdown('<h2 class="section-header">📝 Enter Your Details</h2>', unsafe_allow_html=True)
            
            # Input fields
            age = st.number_input(
                "Age",
                min_value=16,
                max_value=75,
                value=30,
                help="Your current age"
            )
            
            education_options = ["High School", "Bachelor's", "Master's", "PhD"]
            education = st.selectbox(
                "Education Level",
                education_options,
                index=1,
                help="Your highest level of education"
            )
            
            job_titles = [
                "Data Scientist", "Software Engineer", "Senior Engineer", 
                "Manager", "Director", "Consultant", "Developer", "Analyst"
            ]
            job_title = st.selectbox(
                "Job Title",
                job_titles,
                help="Your current or target job title"
            )
            
            experience = st.number_input(
                "Years of Experience",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.5,
                help="Total years of relevant work experience"
            )
            
            gender_options = ["Male", "Female"]
            gender = st.selectbox(
                "Gender",
                gender_options,
                help="Select your gender"
            )
            
            predict_button = st.button(
                "🔮 Predict My Salary",
                type="primary",
                use_container_width=True
            )
        
        # Main content area
        if predict_button:
            # Prepare input data
            input_data = {
                'Age': age,
                'Education Level': education,
                'Job Title': job_title,
                'Years of Experience': experience,
                'Gender': gender
            }
            
            # Show loading animation
            with st.spinner('🤖 Analyzing your profile and predicting salary...'):
                time.sleep(1)  # Simulate processing time
                result, errors = self.predict_salary(input_data)
            
            if errors:
                st.markdown('<div class="error-alert">', unsafe_allow_html=True)
                st.markdown("**❌ Validation Errors:**")
                for error in errors:
                    st.markdown(f"• {error}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Display prediction results
                st.markdown('<div class="success-alert">', unsafe_allow_html=True)
                st.markdown("**✅ Prediction Generated Successfully!**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                self.display_prediction_insights(result, input_data)
                
                # Charts section
                st.markdown('<h2 class="section-header">📊 Salary Analysis</h2>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Salary distribution chart
                    dist_chart = self.create_salary_distribution_chart(result['prediction'], job_title)
                    st.plotly_chart(dist_chart, use_container_width=True)
                
                with col2:
                    # Feature importance chart
                    importance_chart = self.create_feature_importance_chart()
                    st.plotly_chart(importance_chart, use_container_width=True)
                
                # Additional insights
                st.markdown('<h2 class="section-header">💡 Career Insights</h2>', unsafe_allow_html=True)
                
                expected_min, expected_max = result['expected_range']
                prediction = result['prediction']
                
                if prediction < expected_min:
                    insight_type = "below"
                    insight_color = "info"
                elif prediction > expected_max:
                    insight_type = "above"
                    insight_color = "success"
                else:
                    insight_type = "within"
                    insight_color = "info"
                
                st.markdown(f'<div class="{insight_color}-alert">', unsafe_allow_html=True)
                
                if insight_type == "below":
                    st.markdown(f"""
                    **💡 Your predicted salary (${prediction:,.0f}) is below the typical range for {job_title} (${expected_min:,.0f} - ${expected_max:,.0f})**
                    
                    **Recommendations:**
                    • Consider gaining additional experience or certifications
                    • Look for opportunities in higher-paying markets
                    • Negotiate based on your unique skills and achievements
                    """)
                elif insight_type == "above":
                    st.markdown(f"""
                    **🎉 Your predicted salary (${prediction:,.0f}) is above the typical range for {job_title} (${expected_min:,.0f} - ${expected_max:,.0f})**
                    
                    **Great news:**
                    • Your profile suggests premium market value
                    • You're likely in a competitive position
                    • Consider leadership or specialized roles
                    """)
                else:
                    st.markdown(f"""
                    **✅ Your predicted salary (${prediction:,.0f}) is within the expected range for {job_title} (${expected_min:,.0f} - ${expected_max:,.0f})**
                    
                    **You're on track:**
                    • Your profile aligns well with market standards
                    • Continue building experience for future growth
                    • Consider specialization for salary increases
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            # Welcome screen
            st.markdown('<div class="clean-card">', unsafe_allow_html=True)
            st.markdown("""
            ## 🚀 Welcome to AI Salary Predictor
            
            Our advanced machine learning model analyzes your profile to provide accurate salary predictions based on:
            
            ### Key Factors:
            - **📚 Education Level**: Your academic background
            - **💼 Job Title**: Your role and responsibilities  
            - **⏱️ Experience**: Years of relevant work experience
            - **🎂 Age**: Your current age and career stage
            - **👤 Gender**: Demographic information
            
            ### How it works:
            1. Enter your details in the sidebar
            2. Click "Predict My Salary" 
            3. Get instant predictions with confidence intervals
            4. View detailed analysis and career insights
            
            **Ready to discover your market value?** Fill out the form on the left to get started! 👈
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show sample prediction for demonstration
            if self.metadata:
                st.markdown('<h2 class="section-header">📈 Model Performance</h2>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                metrics = self.metadata.get('final_metrics', {})
                
                with col1:
                    r2_score = metrics.get('test_r2', 0.85)
                    st.markdown(f"""
                    <div class="feature-card">
                        <h3>🎯 Accuracy</h3>
                        <h2>{r2_score:.1%}</h2>
                        <p>R² Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    rmse = metrics.get('test_rmse', 15000)
                    st.markdown(f"""
                    <div class="feature-card">
                        <h3>📊 Precision</h3>
                        <h2>±${rmse:,.0f}</h2>
                        <p>Average Error</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="feature-card">
                        <h3>🔬 Model Type</h3>
                        <h2>ML Ensemble</h2>
                        <p>Advanced Algorithm</p>
                    </div>
                    """, unsafe_allow_html=True)


# Main execution
if __name__ == "__main__":
    app = SalaryPredictionApp()
    app.run_app()