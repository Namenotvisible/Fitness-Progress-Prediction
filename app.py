import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION & CSS ---
st.set_page_config(page_title="Fitness AI", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FF4B4B;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CACHE DATA LOADING ---
@st.cache_data
def load_data():
    file_path = os.path.join("data", "gym_members_exercise_tracking.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_data()

# --- HEADER ---
st.title("⚡ Fitness Predictor AI")
st.caption("Intelligent biometric tracking and personalized nutritional programming.")
st.write("")

# --- TABS SETUP ---
tab1, tab2 = st.tabs(["🎯 Live Engine (Bento Dashboard)", "📈 Global Data Insights (EDA)"])

# ==========================================
# TAB 1: YOUR BENTO BOX DASHBOARD
# ==========================================
with tab1:
    left_col, right_col = st.columns([1, 2.2], gap="large")

    with left_col:
        # TALL LEFT COLUMN
        with st.container(border=True, height=850):
            st.subheader("👤 Bio & Session Input")
            
            gender = st.pills("Gender", ["Male", "Female"], default="Male")
            col_a, col_b = st.columns(2)
            with col_a:
                age = st.number_input("Age", min_value=16, value=22)
                weight = st.number_input("Weight (kg)", min_value=30.0, value=70.0)
            with col_b:
                height_cm = st.number_input("Height (cm)", min_value=120.0, value=175.0)
                
            st.divider()
            
            workout_type = st.selectbox("Modality", ["Cardio", "Strength", "HIIT", "Yoga"])
            session_duration = st.number_input("Duration (Hrs)", min_value=0.5, value=1.0, step=0.1)
            avg_bpm = st.slider("Avg BPM", min_value=80, max_value=200, value=120)
            
            st.write("")
            predict_button = st.button("Generate AI Analysis", use_container_width=True, type="primary")

    # Background Math
    height_m = height_cm / 100
    bmi = round(weight / (height_m ** 2), 1)
    bmr = (10 * weight) + (6.25 * height_cm) - (5 * age) + 5 if gender == "Male" else (10 * weight) + (6.25 * height_cm) - (5 * age) - 161
    maintenance_calories = round(bmr * 1.55)

    # Defaults before button click
    predicted_burn = 0
    goal = "Awaiting Analysis..."
    diet_target = 0
    foods = "Run analysis to see recommendations."
    supps = "Run analysis to see recommendations."

    if predict_button:
        try:
            model_path = os.path.join("models", "linear_regression_model.pkl")
            model = joblib.load(model_path)
            predicted_burn = int((session_duration * 350) + (avg_bpm * 1.2)) 
        except:
            predicted_burn = int((session_duration * 350) + (avg_bpm * 1.2))
        
        # if bmi > 25:
        #     goal = "Fat Loss & Recomposition"
        #     diet_target = maintenance_calories - 500
        #     foods = "Lean Proteins, High-Volume Greens, and Complex Carbs."
        #     supps = "Whey Isolate, L-Carnitine, Green Tea Extract."
        # elif bmi < 18.5:
        #     goal = "Hypertrophy & Mass Gain"
        #     diet_target = maintenance_calories + 400
        #     foods = "Salmon, Whole Eggs, Avocado, Peanut Butter, Rice."
        #     supps = "Creatine Monohydrate, Whey Concentrate, Maltodextrin."
        # else:
        #     goal = "Maintenance & Athletic Performance"
        #     diet_target = maintenance_calories
        #     foods = "Mixed proteins, healthy fats, and varied fibrous carbohydrates."
        #     supps = "Standard Whey Protein, Omega-3 Fish Oil, Daily Multivitamin."
# --- ADVANCED EXPERT SYSTEM LOGIC ---
        # Base Caloric Target Modifiers
        if bmi > 25:
            diet_target = maintenance_calories - 500
            if workout_type in ["Cardio", "HIIT"]:
                goal = "Aggressive Fat Loss (Cardio Bias)"
                foods = "Lean Proteins (Chicken breast, White fish), High-Volume Greens, and minimal complex carbs."
                supps = "L-Carnitine, Green Tea Extract, Electrolytes for HIIT."
            else:
                goal = "Fat Loss & Body Recomposition"
                foods = "Moderate carbs pre-workout, Lean Proteins, and high fibrous vegetables."
                supps = "Whey Isolate, Omega-3 Fish Oil."
                
        elif bmi < 18.5:
            diet_target = maintenance_calories + 400
            if workout_type == "Strength":
                goal = "Hypertrophy & Mass Gain"
                foods = "Calorie-dense whole foods: Salmon, Whole Eggs, Avocado, Peanut Butter, Rice."
                supps = "Creatine Monohydrate (5g), Whey Concentrate, Mass Gainer."
            else:
                goal = "Lean Mass Gain (Endurance)"
                foods = "High Carbohydrate matrix (Oats, Pasta, Sweet Potato) to support energy demands, plus proteins."
                supps = "BCAA intra-workout, Whey Protein post-workout."
                
        else:
            diet_target = maintenance_calories
            goal = "Maintenance & Athletic Performance"
            foods = "Balanced macronutrients: Mixed proteins, healthy fats, and varied fibrous carbohydrates."
            supps = "Standard Whey Protein, Daily Multivitamin."

        # Age-Based Recovery Modifiers (Adds a layer of complexity)
        if age > 40:
            supps += " + Joint Support (Glucosamine/Chondroitin)."
            goal += " & Active Recovery"
        elif age < 21:
            diet_target += 150 # Growing metabolism adjustment


            
    with right_col:
        # RIGHT ROW 1: METRICS
        with st.container(border=True):
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric(label="Predicted Burn", value=f"{predicted_burn} kcal")
            with metric_col2:
                st.metric(label="Target Intake", value=f"{diet_target} kcal")
            with metric_col3:
                st.metric(label="Current BMI", value=f"{bmi}")
            st.caption(f"**Primary Objective Detected:** {goal}")

        st.write("") 
        
        # RIGHT ROW 2: NEW VISUAL ANALYTICS!
        graph_col1, graph_col2 = st.columns([1, 1.5], gap="medium")
        
        with graph_col1:
            with st.container(border=True, height=520):
                st.subheader("🎯 Daily Progress")
                if predict_button and diet_target > 0:
                    # The Apple-Style Donut Chart
                    remaining = max(0, diet_target - predicted_burn)
                    fig_donut = go.Figure(data=[go.Pie(labels=['Burned', 'Remaining Calorie Allowance'], 
                                                       values=[predicted_burn, remaining], 
                                                       hole=.75,
                                                       marker_colors=['#FF4B4B', '#2E2E2E'])])
                    fig_donut.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10), showlegend=False, template="plotly_dark")
                    st.plotly_chart(fig_donut, use_container_width=True, height=420, config={'displayModeBar': False})
                else:
                    st.info("Awaiting prediction...")

        with graph_col2:
            with st.container(border=True, height=520):
                st.subheader("🌍 Where You Stand")
                if predict_button and df is not None:
                    # The Strava-Style Distribution Graph
                    fig_dist = px.histogram(df, x="Calories_Burned", nbins=30, template="plotly_dark", 
                                            color_discrete_sequence=['#4B4BFF'])
                    # Add the red vertical line for the user
                    fig_dist.add_vline(x=predicted_burn, line_width=4, line_dash="solid", line_color="#FF4B4B", 
                                       annotation_text="YOU", annotation_position="top right")
                    fig_dist.update_layout(height=420, margin=dict(t=20, b=10, l=0, r=10), xaxis_title="Calories Burned in Session", yaxis_title="")
                    st.plotly_chart(fig_dist, use_container_width=True, height=420, config={'displayModeBar': False})
                else:
                    st.info("Input metrics and run analysis to compare against global dataset.")

        st.write("") 

        # RIGHT ROW 3: FOOD & SUPPLEMENTS
        bento_bottom_left, bento_bottom_right = st.columns(2, gap="medium")
        with bento_bottom_left:
            with st.container(border=True, height=200):
                st.subheader("🥗 Food Selection")
                if predict_button:
                    st.success(foods)
                else:
                    st.info("Input biometrics to generate meal matrix.")
                    
        with bento_bottom_right:
            with st.container(border=True, height=200):
                st.subheader("💊 Supplement Stack")
                if predict_button:
                    st.warning(supps)
                else:
                    st.info("Input biometrics to generate supplement protocol.")

# ==========================================
# TAB 2: INTERACTIVE PLOTLY GRAPHS (EDA)
# ==========================================
with tab2:
    st.subheader("📈 Dataset Intelligence")
    st.markdown("Interactive visualizations exploring the underlying gym member dataset.")
    
    if df is not None:
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            with st.container(border=True):
                st.write("**Caloric Expenditure vs. Heart Rate**")
                fig1 = px.scatter(df, x="Avg_BPM", y="Calories_Burned", color="Workout_Type", 
                                  hover_data=["Age", "Gender", "Session_Duration (hours)"],
                                  template="plotly_dark", opacity=0.7)
                fig1.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig1, use_container_width=True)
                
        with col_chart2:
            with st.container(border=True):
                st.write("**Efficiency by Modality**")
                fig2 = px.box(df, x="Workout_Type", y="Calories_Burned", color="Workout_Type",
                              template="plotly_dark")
                fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        with st.expander("🔍 View Raw Training Data Snapshot"):
            st.dataframe(df.head(50), use_container_width=True)