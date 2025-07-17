import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="💼 Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: slideDown 0.8s ease-out;
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    
    .prediction-amount {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-label {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
        font-weight: 600;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    @keyframes slideDown {
        from {
            transform: translateY(-100px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeInUp {
        from {
            transform: translateY(50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stNumberInput > div > div {
        border-radius: 10px;
    }
    
    .stSlider > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("salary_prediction_pipeline.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'salary_prediction_pipeline.pkl' exists in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Data validation functions
def validate_age(age):
    if age < 18 or age > 100:
        return False, "Age must be between 18 and 100"
    return True, ""

def validate_experience(experience, age):
    if experience < 0:
        return False, "Experience cannot be negative"
    if experience > (age - 16):
        return False, "Experience cannot exceed (Age - 16) years"
    return True, ""

def get_salary_insights(age, experience, education, job_title):
    insights = []
    
    # Experience insights
    if experience < 2:
        insights.append("📈 Entry-level position - Consider gaining more experience")
    elif experience < 5:
        insights.append("🔄 Mid-junior level - Good growth potential")
    elif experience < 10:
        insights.append("⭐ Experienced professional - Strong market position")
    else:
        insights.append("🏆 Senior expert - Premium salary expectations")
    
    # Education insights
    if education == "PhD":
        insights.append("🎓 PhD qualification adds significant value")
    elif education == "Master":
        insights.append("📚 Master's degree provides competitive advantage")
    
    # Age-experience ratio
    exp_ratio = experience / (age - 18) if age > 18 else 0
    if exp_ratio > 0.8:
        insights.append("🚀 Excellent experience-to-age ratio")
    elif exp_ratio < 0.3:
        insights.append("💡 Potential for rapid career growth")
    
    return insights

# Main app function
def main():
    model = load_model()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💼 AI-Powered Salary Prediction</h1>
        <p>Get accurate salary predictions based on your professional profile</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📊 Prediction Settings</div>', unsafe_allow_html=True)
        
        show_insights = st.checkbox("Show Career Insights", value=True)
        show_charts = st.checkbox("Show Visualization", value=True)
        currency = st.selectbox("Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)"])
        
        st.markdown("""
        <div class="info-box">
            <h4>💡 Tips for Better Predictions</h4>
            <ul>
                <li>Ensure accurate experience data</li>
                <li>Select the most relevant job title</li>
                <li>Consider location factors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 👤 Personal Information")
        
        # Personal info in columns
        per_col1, per_col2 = st.columns(2)
        
        with per_col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                help="Your current age (18-100 years)"
            )
        
        with per_col2:
            gender = st.selectbox(
                "Gender",
                ("Male", "Female"),
                help="Select your gender"
            )
        
        # Professional info
        st.markdown("### 💼 Professional Details")
        
        prof_col1, prof_col2 = st.columns(2)
        
        with prof_col1:
            education_level = st.selectbox(
                "Education Level",
                ["High School", "Bachelor", "Master", "PhD"],
                index=1,
                help="Your highest education qualification"
            )
        
        with prof_col2:
            job_title = st.selectbox(
                "Job Title",
                ["Developer", "Data Scientist", "Manager", "Analyst", "Engineer"],
                help="Your current or desired job title"
            )
        
        # Experience with advanced slider
        st.markdown("### 📈 Experience")
        experience = st.slider(
            "Years of Experience",
            min_value=0,
            max_value=50,
            value=5,
            help="Your total years of professional experience"
        )
        
        # Progress bar for experience
        exp_progress = min(experience / 20, 1.0)
        st.progress(exp_progress)
        
        if experience <= 2:
            st.info("🌱 Entry Level")
        elif experience <= 5:
            st.info("🔄 Mid-Junior Level")
        elif experience <= 10:
            st.warning("⭐ Experienced")
        else:
            st.success("🏆 Senior Expert")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Validation
        age_valid, age_msg = validate_age(age)
        exp_valid, exp_msg = validate_experience(experience, age)
        
        if not age_valid:
            st.error(age_msg)
        if not exp_valid:
            st.error(exp_msg)
        
        # Show insights
        if show_insights and age_valid and exp_valid:
            st.markdown("### 🔍 Career Insights")
            insights = get_salary_insights(age, experience, education_level, job_title)
            for insight in insights:
                st.info(insight)
    
    # Prediction section
    st.markdown("---")
    
    if st.button("🔮 Predict My Salary", use_container_width=True):
        if not (age_valid and exp_valid):
            st.error("❌ Please fix the validation errors above before predicting.")
        else:
            with st.spinner("🤖 AI is analyzing your profile..."):
                # Simulate processing time for better UX
                time.sleep(1)
                
                try:
                    # Create input DataFrame (using original values, not encoded)
                    input_df = pd.DataFrame({
                        "Age": [age],
                        "Gender": [gender],  # Use original categorical values
                        "Education Level": [education_level],
                        "Job Title": [job_title],
                        "Years of Experience": [experience]
                    })
                    
                    # Make prediction
                    prediction = model.predict(input_df)[0]
                    
                    # Currency conversion (simplified)
                    currency_multipliers = {
                        "₹ (INR)": 1,
                        "$ (USD)": 0.012,
                        "€ (EUR)": 0.011,
                        "£ (GBP)": 0.0095
                    }
                    
                    symbol = currency.split()[0]
                    converted_salary = prediction * currency_multipliers[currency]
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-card">
                        <p class="prediction-label">💰 Predicted Annual Salary</p>
                        <div class="prediction-amount">{symbol} {converted_salary:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        monthly_salary = converted_salary / 12
                        st.metric("Monthly Salary", f"{symbol} {monthly_salary:,.2f}")
                    
                    with col2:
                        hourly_rate = converted_salary / (40 * 52)
                        st.metric("Hourly Rate", f"{symbol} {hourly_rate:.2f}")
                    
                    with col3:
                        daily_rate = converted_salary / 365
                        st.metric("Daily Earning", f"{symbol} {daily_rate:.2f}")
                    
                    # Visualization
                    if show_charts:
                        st.markdown("### 📊 Salary Breakdown")
                        
                        # Create a simple comparison chart
                        job_avg_salaries = {
                            "Developer": 800000,
                            "Data Scientist": 1200000,
                            "Manager": 1500000,
                            "Analyst": 700000,
                            "Engineer": 900000
                        }
                        
                        chart_data = pd.DataFrame({
                            "Role": list(job_avg_salaries.keys()),
                            "Average Salary": list(job_avg_salaries.values())
                        })
                        
                        fig = px.bar(
                            chart_data,
                            x="Role",
                            y="Average Salary",
                            title="Average Salary by Role (INR)",
                            color="Average Salary",
                            color_continuous_scale="viridis"
                        )
                        
                        # Highlight current prediction
                        fig.add_hline(
                            y=prediction,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Your Prediction: ₹{prediction:,.2f}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Success message
                    st.success("✅ Prediction completed successfully!")
                    
                    # Download option
                    result_data = {
                        "Prediction_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Age": age,
                        "Gender": gender,
                        "Education": education_level,
                        "Job_Title": job_title,
                        "Experience": experience,
                        "Predicted_Salary": prediction,
                        "Currency": currency,
                        "Converted_Salary": converted_salary
                    }
                    
                    result_df = pd.DataFrame([result_data])
                    csv = result_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Prediction Report",
                        data=csv,
                        file_name=f"salary_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("💡 Please check your input data and try again.")

if __name__ == "__main__":
    main()