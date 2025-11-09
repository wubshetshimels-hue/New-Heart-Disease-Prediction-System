# app.py - Heart Disease Prediction System
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Try to import optional dependencies
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2e86ab;
        margin: 0.5rem 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
            'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
            'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
        ]

        self.feature_info = {
            'Age': {
                'desc': 'Age in years',
                'type': 'number',
                'min': 20, 'max': 100, 'step': 1,
                'normal_range': (30, 70),
                'default': 45
            },
            'Sex': {
                'desc': 'Gender',
                'type': 'select',
                'options': {'0': 'Female', '1': 'Male'},
                'normal_range': None
            },
            'Chest pain type': {
                'desc': 'Type of chest pain (1: Typical angina, 2: Atypical angina, 3: Non-anginal pain, 4: Asymptomatic)',
                'type': 'select',
                'options': {
                    '1': 'Typical angina',
                    '2': 'Atypical angina',
                    '3': 'Non-anginal pain',
                    '4': 'Asymptomatic'
                },
                'normal_range': None
            },
            'BP': {
                'desc': 'Resting blood pressure (mm Hg) - Systolic',
                'type': 'number',
                'min': 80, 'max': 200, 'step': 1,
                'normal_range': (90, 120),
                'default': 120
            },
            'Cholesterol': {
                'desc': 'Serum cholesterol (mg/dl)',
                'type': 'number',
                'min': 100, 'max': 600, 'step': 1,
                'normal_range': (125, 200),
                'default': 200
            },
            'FBS over 120': {
                'desc': 'Fasting blood sugar > 120 mg/dl (0: Normal <100 mg/dL, 1: High >120 mg/dL)',
                'type': 'select',
                'options': {'0': 'Normal (<100 mg/dL)', '1': 'High (>120 mg/dL)'},
                'normal_range': None
            },
            'EKG results': {
                'desc': 'Resting electrocardiographic results',
                'type': 'select',
                'options': {
                    '0': 'Normal',
                    '1': 'ST-T wave abnormality',
                    '2': 'Left ventricular hypertrophy'
                },
                'normal_range': None
            },
            'Max HR': {
                'desc': 'Maximum heart rate achieved during exercise',
                'type': 'number',
                'min': 60, 'max': 220, 'step': 1,
                'normal_range': (150, 190),  # For adults during exercise
                'default': 150
            },
            'Exercise angina': {
                'desc': 'Exercise induced angina (chest pain)',
                'type': 'select',
                'options': {'0': 'No', '1': 'Yes'},
                'normal_range': None
            },
            'ST depression': {
                'desc': 'ST depression induced by exercise relative to rest (mm)',
                'type': 'number',
                'min': 0.0, 'max': 6.0, 'step': 0.1,
                'normal_range': (0.0, 0.5),  # Normal is 0, up to 0.5 may be borderline
                'default': 1.0
            },
            'Slope of ST': {
                'desc': 'Slope of peak exercise ST segment (1: Upsloping, 2: Flat, 3: Downsloping)',
                'type': 'select',
                'options': {
                    '1': 'Upsloping (Normal)',
                    '2': 'Flat',
                    '3': 'Downsloping (Abnormal)'
                },
                'normal_range': None
            },
            'Number of vessels fluro': {
                'desc': 'Number of major vessels colored by fluoroscopy (0-3)',
                'type': 'select',
                'options': {'0': '0 (Normal)', '1': '1', '2': '2', '3': '3'},
                'normal_range': None
            },
            'Thallium': {
                'desc': 'Thallium stress test result',
                'type': 'select',
                'options': {
                    '3': 'Normal',
                    '6': 'Fixed defect',
                    '7': 'Reversible defect'
                },
                'normal_range': None
            }
        }

        self.load_model()

    def load_model(self):
        """Load the trained model and feature names"""
        try:
            if JOBLIB_AVAILABLE:
                # Try different possible model paths
                possible_paths = [
                    'models/optimized_random_forest_model.pkl',
                    'optimized_random_forest_model.pkl',
                    'model.pkl'
                ]

                model_loaded = False
                for model_path in possible_paths:
                    if os.path.exists(model_path):
                        self.model = joblib.load(model_path)
                        feature_path = model_path.replace('_model.pkl', '_feature_names.pkl')
                        if os.path.exists(feature_path):
                            self.feature_names = joblib.load(feature_path)
                        st.sidebar.success(f"✅ AI Model Loaded Successfully from {model_path}!")
                        model_loaded = True
                        break

                if not model_loaded:
                    st.sidebar.info("🤖 Using Rule-Based System (AI model not found)")
            else:
                st.sidebar.info("🤖 Using Rule-Based System (joblib not available)")
        except Exception as e:
            st.sidebar.info(f"🤖 Using Rule-Based System (Model loading failed: {str(e)})")

    def get_risk_level(self, probability):
        """Determine risk level based on prediction probability"""
        if probability < 0.3:
            return 'Low', '🟢', 'risk-low', "Continue healthy lifestyle with regular checkups. Maintain balanced diet and exercise routine."
        elif probability < 0.6:
            return 'Medium', '🟡', 'risk-medium', "Consider consulting a healthcare provider for routine evaluation. Monitor your health parameters regularly."
        elif probability < 0.8:
            return 'High', '🟠', 'risk-high', "Recommended to consult a cardiologist for comprehensive assessment. Schedule appointment within 2 weeks."
        else:
            return 'Very High', '🔴', 'risk-high', "Immediate medical consultation strongly recommended. Please visit healthcare provider soon."

    def rule_based_prediction(self, input_data):
        """Rule-based prediction when ML model is not available"""
        risk_score = 0

        # Age factor (increased risk after 45)
        if input_data['Age'] > 65:
            risk_score += 3
        elif input_data['Age'] > 55:
            risk_score += 2
        elif input_data['Age'] > 45:
            risk_score += 1

        # Blood pressure factor
        if input_data['BP'] > 180:
            risk_score += 3
        elif input_data['BP'] > 160:
            risk_score += 2
        elif input_data['BP'] > 140:
            risk_score += 1

        # Cholesterol factor
        if input_data['Cholesterol'] > 300:
            risk_score += 3
        elif input_data['Cholesterol'] > 240:
            risk_score += 2
        elif input_data['Cholesterol'] > 200:
            risk_score += 1

        # Chest pain factor (asymptomatic highest risk)
        if input_data['Chest pain type'] == 4:  # Asymptomatic
            risk_score += 3
        elif input_data['Chest pain type'] == 3:  # Non-anginal
            risk_score += 2
        elif input_data['Chest pain type'] == 2:  # Atypical angina
            risk_score += 1

        # ST depression factor
        if input_data['ST depression'] > 2.0:
            risk_score += 3
        elif input_data['ST depression'] > 1.0:
            risk_score += 2
        elif input_data['ST depression'] > 0.5:
            risk_score += 1

        # Other factors
        if input_data['Exercise angina'] == 1:
            risk_score += 2
        if input_data['Thallium'] == 7:  # Reversible defect
            risk_score += 3
        elif input_data['Thallium'] == 6:  # Fixed defect
            risk_score += 2
        if input_data['Number of vessels fluro'] == 3:
            risk_score += 3
        elif input_data['Number of vessels fluro'] == 2:
            risk_score += 2
        elif input_data['Number of vessels fluro'] == 1:
            risk_score += 1
        if input_data['Slope of ST'] == 3:  # Downsloping
            risk_score += 2
        elif input_data['Slope of ST'] == 2:  # Flat
            risk_score += 1
        if input_data['FBS over 120'] == 1:  # High fasting blood sugar
            risk_score += 1
        if input_data['EKG results'] == 2:  # Left ventricular hypertrophy
            risk_score += 2
        elif input_data['EKG results'] == 1:  # ST-T wave abnormality
            risk_score += 1
        if input_data['Sex'] == 1:  # Male gender has higher risk
            risk_score += 1

        # Calculate probability (normalized to 0-1 scale)
        max_possible_score = 25  # Updated based on all risk factors
        probability = min(risk_score / max_possible_score, 0.95)

        return probability

    def predict(self, input_data):
        """Make prediction for heart disease"""
        try:
            if self.model and JOBLIB_AVAILABLE:
                # ML Model prediction
                input_df = pd.DataFrame([input_data])
                # Ensure all required features are present
                for feature in self.feature_names:
                    if feature not in input_df.columns:
                        st.error(f"Missing feature: {feature}")
                        return {'success': False, 'error': f'Missing feature: {feature}'}

                input_df = input_df[self.feature_names]
                probability = self.model.predict_proba(input_df)[0][1]
                method = "AI Machine Learning Model"
            else:
                # Rule-based prediction
                probability = self.rule_based_prediction(input_data)
                method = "Medical Rule-Based System"

            risk_level, risk_emoji, risk_class, recommendation = self.get_risk_level(probability)

            return {
                'prediction': 1 if probability > 0.5 else 0,
                'probability': probability,
                'risk_level': risk_level,
                'risk_emoji': risk_emoji,
                'risk_class': risk_class,
                'recommendation': recommendation,
                'method': method,
                'success': True
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def create_sidebar():
    """Create the sidebar with information and controls"""
    st.sidebar.title("❤️ Heart Disease Predictor")

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This AI-powered system predicts heart disease risk using machine learning and medical rules.

    **Medical Disclaimer:**
    This tool is for educational and screening purposes only. Always consult healthcare professionals for medical diagnosis.
    """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")

    # Dependency status
    if JOBLIB_AVAILABLE:
        st.sidebar.success("✅ ML Engine: Available")
    else:
        st.sidebar.warning("⚠️ ML Engine: Limited")

    if PLOTLY_AVAILABLE:
        st.sidebar.success("✅ Visualizations: Available")
    else:
        st.sidebar.warning("⚠️ Visualizations: Basic")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Actions")

    if st.sidebar.button("🔄 Reset Form"):
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        Developed with ❤️ for Healthcare<br>
        Version 1.0 • Medical AI
    </div>
    """, unsafe_allow_html=True)


def create_feature_input(feature, info):
    """Create input for a single feature"""
    with st.container():
        if info['type'] == 'number':
            value = st.number_input(
                label=f"**{feature}**",
                min_value=info['min'],
                max_value=info['max'],
                value=info.get('default', (info['min'] + info['max']) // 2),
                step=info['step'],
                help=info['desc']
            )
        elif info['type'] == 'select':
            option_key = st.selectbox(
                label=f"**{feature}**",
                options=list(info['options'].keys()),
                format_func=lambda x: f"{info['options'][x]}",
                help=info['desc']
            )
            value = int(option_key)

        # Show normal range if available
        if info.get('normal_range'):
            min_val, max_val = info['normal_range']
            if min_val <= value <= max_val:
                st.markdown(
                    f"<span style='color: green; font-size: 0.8rem;'>✓ Within normal range ({min_val}-{max_val})</span>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<span style='color: orange; font-size: 0.8rem;'>⚠️ Outside normal range ({min_val}-{max_val})</span>",
                    unsafe_allow_html=True)

        return value


def create_input_form(predictor):
    """Create the main input form"""
    st.markdown('<div class="main-header">Heart Disease Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center; color: #666; margin-bottom: 2rem;">AI-Powered Cardiac Risk Assessment</div>',
        unsafe_allow_html=True)

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Detailed Input", "⚡ Quick Assessment"])

    input_data = {}

    with tab1:
        st.subheader("Patient Clinical Data")

        # Create columns for better layout
        col1, col2 = st.columns(2)

        features_list = list(predictor.feature_names)
        mid_point = len(features_list) // 2

        with col1:
            for feature in features_list[:mid_point]:
                input_data[feature] = create_feature_input(feature, predictor.feature_info[feature])

        with col2:
            for feature in features_list[mid_point:]:
                input_data[feature] = create_feature_input(feature, predictor.feature_info[feature])

    with tab2:
        st.subheader("Quick Health Assessment")
        st.info("Provide basic health information for preliminary assessment")

        col1, col2 = st.columns(2)

        with col1:
            input_data['Age'] = st.slider("**Age**", 20, 100, 45)
            input_data['BP'] = st.slider("**Blood Pressure**", 80, 200, 120)
            input_data['Cholesterol'] = st.slider("**Cholesterol**", 100, 600, 200)
            input_data['Max HR'] = st.slider("**Max Heart Rate**", 60, 220, 150)

        with col2:
            input_data['Sex'] = st.selectbox("**Gender**", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            input_data['Exercise angina'] = st.selectbox("**Exercise Chest Pain**", [0, 1],
                                                         format_func=lambda x: "No" if x == 0 else "Yes")
            input_data['ST depression'] = st.slider("**ST Depression**", 0.0, 6.0, 1.0, 0.1)

            # Set default values for other features
            default_values = {
                'Chest pain type': 1,
                'FBS over 120': 0,
                'EKG results': 0,
                'Slope of ST': 1,
                'Number of vessels fluro': 0,
                'Thallium': 3
            }

            for feature, default in default_values.items():
                input_data[feature] = default

    return input_data


def display_metric_analysis(metric, value, info):
    """Display analysis for a single metric"""
    if not info.get('normal_range'):
        return

    min_val, max_val = info['normal_range']
    status = "Normal" if min_val <= value <= max_val else "Attention Needed"
    color = "green" if status == "Normal" else "orange"

    st.markdown(f"""
    <div class="feature-card">
        <h4>{metric}</h4>
        <h3 style="color: {color};">{value}</h3>
        <p>Status: <strong style="color: {color};">{status}</strong></p>
        <p style="font-size: 0.8rem;">Normal Range: {min_val}-{max_val}</p>
    </div>
    """, unsafe_allow_html=True)


def display_results(results, input_data, predictor):
    """Display comprehensive prediction results"""
    st.markdown("---")

    if not results['success']:
        st.error(f"❌ Prediction failed: {results['error']}")
        return

    # Main prediction card
    risk_class = results['risk_class']
    risk_emoji = results['risk_emoji']

    st.markdown(f"""
    <div class="prediction-card {risk_class}">
        <h2 style="color: white; margin: 0;">{risk_emoji} {results['risk_level']} RISK</h2>
        <h3 style="color: white; margin: 0.5rem 0;">Prediction: {'HEART DISEASE DETECTED' if results['prediction'] == 1 else 'NO HEART DISEASE DETECTED'}</h3>
        <h4 style="color: white; margin: 0;">Confidence: {results['probability']:.1%}</h4>
        <p style="color: white; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Method: {results['method']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Results in columns
    col1, col2 = st.columns(2)

    with col1:
        if PLOTLY_AVAILABLE:
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=results['probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Heart Disease Probability"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple metric display
            st.metric("Risk Probability", f"{results['probability']:.1%}")
            st.metric("Risk Level", f"{results['risk_level']} {results['risk_emoji']}")

    with col2:
        st.subheader("🎯 Medical Recommendations")
        st.info(results['recommendation'])

        # Next steps based on risk
        if results['risk_level'] in ['High', 'Very High']:
            st.warning("""
            **Recommended Actions:**
            - Consult cardiologist within 2 weeks
            - Consider stress echocardiogram
            - Lipid profile and cardiac markers test
            - Lifestyle modification program
            - Regular blood pressure monitoring
            """)
        elif results['risk_level'] == 'Medium':
            st.warning("""
            **Suggested Actions:**
            - Schedule routine health checkup
            - Monitor blood pressure weekly
            - Check cholesterol levels quarterly
            - Maintain healthy diet and exercise
            """)
        else:
            st.success("""
            **Maintenance Actions:**
            - Annual cardiac checkup
            - Maintain healthy BMI
            - Regular physical activity
            - Balanced diet low in saturated fats
            - No smoking, limited alcohol
            """)

    # Detailed analysis
    st.markdown("---")
    st.subheader("📊 Detailed Health Analysis")

    # Key parameters analysis
    col1, col2, col3 = st.columns(3)

    key_metrics = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

    for i, metric in enumerate(key_metrics):
        if metric in input_data and metric in predictor.feature_info:
            with [col1, col2, col3][i % 3]:
                display_metric_analysis(metric, input_data[metric], predictor.feature_info[metric])


def save_prediction(input_data, results):
    """Save prediction to history"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_data = {
            'timestamp': timestamp,
            **input_data,
            'prediction': 'Heart Disease' if results['prediction'] == 1 else 'No Heart Disease',
            'probability': results['probability'],
            'risk_level': results['risk_level'],
            'method': results['method']
        }

        df = pd.DataFrame([save_data])

        # Ensure directory exists
        os.makedirs('data', exist_ok=True)

        file_path = 'data/heart_predictions.csv'
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

        st.success("✅ Prediction saved to medical records!")
    except Exception as e:
        st.error(f"❌ Error saving prediction: {e}")


def main():
    """Main application function"""

    # Initialize predictor
    predictor = HeartDiseasePredictor()

    # Create sidebar
    create_sidebar()

    # Main content area
    if not JOBLIB_AVAILABLE:
        st.warning("""
        ⚠️ **ML Engine Notice:** 
        Advanced machine learning features are limited. The system is using medical rule-based prediction.
        For full AI capabilities, install: `pip install joblib scikit-learn`
        """)

    # Create input form and get data
    input_data = create_input_form(predictor)

    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button(
            "🔍 ANALYZE HEART DISEASE RISK",
            type="primary",
            use_container_width=True,
            help="Click to analyze the patient's heart disease risk"
        )

    if predict_clicked:
        with st.spinner("🔄 Analyzing clinical data..."):
            # Add small delay for better UX
            import time
            time.sleep(1)

            results = predictor.predict(input_data)
            display_results(results, input_data, predictor)

            # Save prediction
            if st.button("💾 Save to Medical Records", use_container_width=True):
                save_prediction(input_data, results)


# Run the application
if __name__ == "__main__":
    main()