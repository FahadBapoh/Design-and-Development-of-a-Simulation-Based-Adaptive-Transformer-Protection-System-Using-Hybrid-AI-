
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
import base64
import io
import uuid
import os

# Streamlit configuration
st.set_page_config(
    layout="wide", 
    page_title="Power System Protection Suite", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged from your code)
st.markdown("""
<style>
    .logo-container {
        position:absolute;
        top: 60px;
        right: 10px;
        z-index: 1000;
    }
    .logo-img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .logo-img:hover {
        transform: scale(1.05);
    }
    :root {
        --primary: #1e3c72;
        --secondary: #2a5298;
        --danger: #f44336;
        --warning: #ff9800;
        --success: #4caf50;
    }
    .main-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid var(--secondary);
        margin: 0.75rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .alarm-critical {
        background: #ffebee;
        border-left-color: var(--danger);
        animation: pulse 2s infinite;
    }
    .alarm-warning {
        background: #fff3e0;
        border-left-color: var(--warning);
    }
    .status-normal {
        background: #e8f5e8;
        border-left-color: var(--success);
    }
    .nav-link {
        padding: 0.75rem 1.5rem;
        border-radius: 4px;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .nav-link:hover {
        background-color: #e3eaf3;
    }
    .nav-link.active {
        background-color: var(--secondary);
        color: white !important;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        border-radius: 8px !important;
        background-color: #f0f2f6 !important;
        transition: all 0.3s !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary) !important;
        color: white !important;
    }
    .power-component {
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
        text-align: center;
        font-weight: bold;
    }
    .component-normal {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .component-warning {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
    }
    .component-fault {
        background-color: #ffebee;
        border: 2px solid #f44336;
        animation: pulse 2s infinite;
    }
    .component-tripped {
        background-color: #f5f5f5;
        border: 2px solid #9e9e9e;
    }
    .transformer-selection {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .transformer-button {
        background-color: var(--secondary);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
        margin: 0.5rem 0;
    }
    .transformer-button:hover {
        background-color: var(--primary);
        transform: scale(1.05);
    }
    .api-key-container {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .api-key {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to convert image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error("Logo file 'logo.jpeg' not found. Please ensure the file is in the correct directory.")
        return ""

# Display the logo
logo = "logo.jpeg"
st.markdown(f'<div class="logo-container"><img src="data:image/jpeg;base64,{get_base64_image(logo)}" class="logo-img"></div>', unsafe_allow_html=True)

# Initialize session state
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = {
        'time': [],
        'voltage': [],
        'current': [],
        'temperature': [],
        'oil_level': [],
        'vibration': [],
        'power_factor': [],
        'frequency': [],
        'harmonics': [],
        'load_factor': [],
        'fault_type': [],
        'severity': [],
        'alarm': [],
        'fault_detected': [],
        'relay_tripped': [],
        'power_flow': [],
        'bus_voltages': [],
        'line_loadings': []
    }

if 'historical_data' not in st.session_state:
    st.session_state.historical_data = []

if 'system_events' not in st.session_state:
    st.session_state.system_events = []

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Real-time Monitoring"

if 'running' not in st.session_state:
    st.session_state.running = False

if 'transformer_rating' not in st.session_state:
    st.session_state.transformer_rating = None

if 'models_dict' not in st.session_state:
    st.session_state.models_dict = {}

if 'previous_page' not in st.session_state:
    st.session_state.previous_page = "Real-time Monitoring"

if 'api_key' not in st.session_state:
    st.session_state.api_key = None

# Power System Components Class
class PowerSystemComponent:
    def __init__(self, name, component_type, x_pos, y_pos, maintenance_mode=False, normal_value=None):
        self.name = name
        self.type = component_type
        self.x = x_pos
        self.y = y_pos
        self.status = "normal"
        self.maintenance_mode = maintenance_mode
        self.value = normal_value
        self.current_value = normal_value
        self.breaker_closed = True if component_type == "breaker" else None
        
    def update_status(self, new_value=None, fault_detected=False, severity=0, breaker_status=None):
        if new_value is not None:
            self.current_value = new_value
            
        if breaker_status is not None and self.type == "breaker":
            self.breaker_closed = breaker_status
            self.status = "tripped" if not breaker_status else "normal"
        elif fault_detected:
            if severity > 80:
                self.status = "fault"
            elif severity > 40:
                self.status = "warning"
            else:
                self.status = "normal"
        else:
            self.status = "normal"

# Initialize Power System Components
def initialize_power_system():
    components = {
        'generator': PowerSystemComponent("Generator", "generator", 1, 4, normal_value=200),
        'transmission_line': PowerSystemComponent("Transmission Line (161kV)", "line", 4, 4, normal_value=100),
        'stepdown_transformer_1': PowerSystemComponent("Step-down Transformer (33kV)", "transformer", 6, 4, normal_value=33),
        'stepdown_transformer_2': PowerSystemComponent("Step-down Transformer (433V)", "transformer", 8, 4, normal_value=433),
        'load': PowerSystemComponent("Distribution Load", "load", 10, 4, normal_value=50),
        'relay': PowerSystemComponent("Relay", "relay", 8.5, 3.4, normal_value=0),
        'breaker': PowerSystemComponent("CB", "breaker", 8.5, 4.5, normal_value=1),
    }
    return components

if 'power_components' not in st.session_state:
    st.session_state.power_components = initialize_power_system()

# Enhanced Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">⚡ Power System Protection Suite</h1>
    <p style="color: #e3f2fd; margin: 0.5rem 0 0 0;">Advanced Transformer Protection & Power System Analysis</p>
</div>
""", unsafe_allow_html=True)

# Navigation options
nav_options = {
    "Real-time Monitoring": "📊",
    "Transformer Protection": "🛡️",
    "Power System Analysis": "🔌",
    "Manual Prediction": "🔍",
    "Historical Data": "📈",
    "System Configuration": "⚙️",
    "Transformer Selection": "🔧",
    "API Key Management": "🔐"
}

# Navigation Sidebar
with st.sidebar:
    st.markdown("## Navigation")
    
    for page, icon in nav_options.items():
        if st.button(f"{icon} {page}", use_container_width=True, 
                    type="primary" if st.session_state.current_page == page else "secondary"):
            st.session_state.current_page = page
    
    st.markdown("---")
    st.markdown("## System Status")
    
    if st.session_state.sim_data['time']:
        latest_severity = st.session_state.sim_data['severity'][-1] if st.session_state.sim_data['severity'] else 0
        latest_fault = st.session_state.sim_data['fault_type'][-1] if st.session_state.sim_data['fault_type'] else 0
        
        if latest_severity > 80:
            status_class = "alarm-critical"
            status_text = "🚨 CRITICAL ALERT"
        elif latest_severity > 40:
            status_class = "alarm-warning"
            status_text = "⚠️ WARNING"
        else:
            status_class = "status-normal"
            status_text = "✅ NORMAL"
        
        st.markdown(f'<div class="metric-card {status_class}">{status_text}</div>', unsafe_allow_html=True)
        st.metric("Severity", f"{latest_severity:.1f}%")
        st.metric("Fault Type", fault_labels.get(latest_fault, "Normal") if 'fault_labels' in globals() else "Normal")
    
    st.markdown("---")
    st.markdown("## Simulation Control")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start" if not st.session_state.running else "⏸️ Stop", use_container_width=True):
            st.session_state.running = not st.session_state.running
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.sim_data = {key: [] for key in st.session_state.sim_data}
            st.session_state.system_events = []
            st.session_state.power_components = initialize_power_system()
            st.session_state.transformer_rating = None
            st.session_state.models_dict = {}
            st.session_state.api_key = None
            st.rerun()
    
    if st.button("📥 Export Data", use_container_width=True):
        df_export = pd.DataFrame(st.session_state.sim_data)
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"power_system_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Fault labels
fault_labels = {
    0: "Normal",
    1: "Overvoltage",
    2: "Undervoltage", 
    3: "Overcurrent",
    4: "Overheating",
    5: "Mechanical fault",
    6: "Power factor issue",
    7: "Harmonic distortion",
    8: "Load imbalance",
    9: "Undercurrent"
}

# Load pre-trained models and scalers
def load_models(transformer_rating=None):
    try:
        if transformer_rating is None:
            transformer_rating = 100  # Default rating
        if transformer_rating not in st.session_state.models_dict:
            rf_model = joblib.load(f"models/rf_{transformer_rating}kva.pkl")
            nn_model = joblib.load(f"models/nn_{transformer_rating}kva.pkl")
            scaler = joblib.load(f"models/scaler_{transformer_rating}kva.pkl")
            features = ['voltage', 'current', 'temperature', 'oil_level', 'vibration', 
                        'power_factor', 'frequency', 'harmonics', 'load_factor']
            st.session_state.models_dict[transformer_rating] = {
                'Random Forest': rf_model,
                'Neural Network': nn_model,
                'scaler': scaler,
                'features': features
            }
        return st.session_state.models_dict[transformer_rating]
    except Exception as e:
        st.error(f"Error loading models for {transformer_rating} kVA: {str(e)}")
        return None

# Fuzzy logic system
def create_fuzzy_system(transformer_rating=None):
    try:
        voltage = ctrl.Antecedent(np.arange(300, 551, 1), 'voltage')
        if transformer_rating:
            power_va = transformer_rating * 1000
            current_nom = power_va / (np.sqrt(3) * 433 * 1)
            current_range = np.arange(max(current_nom * 0.5, 1), current_nom * 2.0, 1)
        else:
            current_nom = 100
            current_range = np.arange(50, 201, 1)
        current = ctrl.Antecedent(current_range, 'current')
        temperature = ctrl.Antecedent(np.arange(50, 121, 1), 'temperature')
        oil_level = ctrl.Antecedent(np.arange(20, 101, 1), 'oil_level')
        vibration = ctrl.Antecedent(np.arange(0, 11, 0.1), 'vibration')
        power_factor = ctrl.Antecedent(np.arange(0.6, 1.01, 0.01), 'power_factor')
        frequency = ctrl.Antecedent(np.arange(48, 52, 0.1), 'frequency')
        harmonics = ctrl.Antecedent(np.arange(0, 25, 0.1), 'harmonics')
        severity = ctrl.Consequent(np.arange(0, 101, 1), 'severity')
        
        for var in [voltage, current, temperature, oil_level, vibration, 
                    power_factor, frequency, harmonics]:
            var.automf(3, names=['low', 'normal', 'high'])
        
        voltage_nom = 433
        voltage['low'] = fuzz.trapmf(voltage.universe, [voltage.universe.min(), voltage.universe.min(), voltage_nom * 0.95, voltage_nom * 1.0])
        voltage['normal'] = fuzz.trimf(voltage.universe, [voltage_nom * 0.95, voltage_nom, voltage_nom * 1.05])
        voltage['high'] = fuzz.trapmf(voltage.universe, [voltage_nom * 1.0, voltage_nom * 1.05, voltage.universe.max(), voltage.universe.max()])

        current['low'] = fuzz.trapmf(current.universe, [current.universe.min(), current.universe.min(), current_nom * 0.95, current_nom * 1.0])
        current['normal'] = fuzz.trimf(current.universe, [current_nom * 0.95, current_nom, current_nom * 1.05])
        current['high'] = fuzz.trapmf(current.universe, [current_nom * 1.0, current_nom * 1.05, current.universe.max(), current.universe.max()])

        severity['low'] = fuzz.trimf(severity.universe, [0, 0, 50])
        severity['medium'] = fuzz.trimf(severity.universe, [0, 50, 100])
        severity['high'] = fuzz.trimf(severity.universe, [50, 100, 100])
        
        rules = [
            ctrl.Rule(voltage['high'] | current['high'] | temperature['high'], severity['high']),
            ctrl.Rule(voltage['normal'] & current['normal'] & temperature['normal'], severity['low']),
            ctrl.Rule(oil_level['low'] | vibration['high'], severity['medium']),
            ctrl.Rule(power_factor['low'] | frequency['low'] | frequency['high'], severity['medium']),
            ctrl.Rule(harmonics['high'], severity['high']),
            ctrl.Rule(voltage['low'] | current['low'], severity['high'])
        ]
        
        severity_ctrl = ctrl.ControlSystem(rules)
        fuzzy_system = ctrl.ControlSystemSimulation(severity_ctrl)
        
        return fuzzy_system, {
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'oil_level': oil_level,
            'vibration': vibration,
            'power_factor': power_factor,
            'frequency': frequency,
            'harmonics': harmonics
        }
    except Exception as e:
        st.error(f"Error creating fuzzy system: {str(e)}")
        return None, None

# Real-time data generation
def generate_realtime_data(fault_prob, transformer_rating=None):
    try:
        fault_detected = random.random() < fault_prob / 100
        relay_tripped = fault_detected and random.random() < 0.9
        
        voltage = 433
        power_factor = 1
        if transformer_rating:
            power_va = transformer_rating * 1000
            current_nom = power_va / (np.sqrt(3) * voltage * power_factor)
        else:
            current_nom = 100
        
        if fault_detected:
            fault = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            if fault == 1:  # Overvoltage
                return (voltage * 1.1 + random.uniform(-5, 5), current_nom, 65, 80, 2, 0.95, 50, 2, 0.75, fault_detected, relay_tripped, 180, [161, 33, 433], [80], fault)
            elif fault == 2:  # Undervoltage
                return (voltage * 0.9 + random.uniform(-5, 5), current_nom, 65, 80, 2, 0.95, 50, 2, 0.75, fault_detected, relay_tripped, 160, [150, 30, 350], [70], fault)
            elif fault == 3:  # Overcurrent
                return (voltage, current_nom * 1.2 + random.uniform(-5, 5), 75 + random.uniform(-2, 2), 80, 2, 0.9, 50, 4, 0.9, fault_detected, relay_tripped, 200, [161, 33, 433], [100], fault)
            elif fault == 4:  # Overheating
                return (voltage, current_nom, 100 + random.uniform(-5, 5), 70 + random.uniform(-2, 2), 3, 0.92, 50, 5, 0.8, fault_detected, relay_tripped, 180, [161, 33, 433], [80], fault)
            elif fault == 5:  # Mechanical fault
                return (voltage, current_nom, 65, 80, 7 + random.uniform(-1, 1), 0.95, 50, 6, 0.75, fault_detected, relay_tripped, 180, [161, 33, 433], [80], fault)
            elif fault == 6:  # Power factor issue
                return (voltage, current_nom, 65, 80, 2, 0.75 + random.uniform(-0.05, 0.05), 50, 8, 0.75, fault_detected, relay_tripped, 170, [161, 33, 433], [75], fault)
            elif fault == 7:  # Harmonic distortion
                return (voltage, current_nom, 65, 80, 2, 0.85, 50, 15 + random.uniform(-2, 2), 0.75, fault_detected, relay_tripped, 170, [161, 33, 433], [75], fault)
            elif fault == 8:  # Load imbalance
                return (voltage, current_nom, 65, 80, 2, 0.95, 50, 3, 0.4 + random.uniform(-0.05, 0.05), fault_detected, relay_tripped, 190, [161, 33, 433], [85], fault)
            elif fault == 9:  # Undercurrent
                return (voltage, current_nom * 0.8 + random.uniform(-5, 5), 65, 80, 2, 0.95, 50, 2, 0.5 + random.uniform(-0.05, 0.05), fault_detected, relay_tripped, 140, [161, 33, 433], [60], fault)
        else:
            return (
                voltage + random.uniform(-10, 10), 
                current_nom + random.uniform(-current_nom*0.05, current_nom*0.05), 
                65 + random.uniform(-2, 2), 
                80 + random.uniform(-2, 2), 
                2 + random.uniform(-0.2, 0.2), 
                0.95 + random.uniform(-0.01, 0.01), 
                50 + random.uniform(-0.05, 0.05), 
                2 + random.uniform(-0.2, 0.2), 
                0.75 + random.uniform(-0.02, 0.02),
                False, False, 180, [161, 33, 433], [80], 0
            )
    except Exception as e:
        st.error(f"Error generating real-time data: {str(e)}")
        return (433, 100, 65, 80, 2, 0.95, 50, 2, 0.75, False, False, 180, [161, 33, 433], [80], 0)

# Enhanced Single Line Diagram (unchanged from your code)
def create_interactive_single_line_diagram(fault_detected, relay_tripped, severity, voltage=433, current=100, load=50):
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        
        components = st.session_state.power_components
        
        components['stepdown_transformer_1'].update_status(
            new_value=33 if not fault_detected else (33 * (1 - severity/200)),
            fault_detected=fault_detected,
            severity=severity
        )
        components['stepdown_transformer_2'].update_status(
            new_value=voltage,
            fault_detected=fault_detected,
            severity=severity
        )
        components['breaker'].update_status(breaker_status=not relay_tripped)
        components['relay'].update_status(fault_detected=severity > 40, severity=severity)
        components['transmission_line'].update_status(new_value=current, fault_detected=fault_detected, severity=severity)
        components['load'].update_status(new_value=load, fault_detected=fault_detected, severity=severity)
        
        status_colors = {
            'normal': '#4CAF50',
            'warning': '#FF9800', 
            'fault': '#F44336',
            'tripped': '#9E9E9E'
        }
        
        ax.add_patch(Rectangle((0, 0), 11, 6, facecolor='#f8f9fa', edgecolor='none'))
        
        ax.text(5.5, 5.5, 'Power System Single Line Diagram', 
                fontsize=14, fontweight='bold', ha='center', va='center')
        
        if fault_detected and severity > 40:
            ax.add_patch(FancyBboxPatch((3.5, 5.0), 4, 0.4, boxstyle="round,pad=0.3", 
                                       facecolor='#ffebee', edgecolor='#f44336', linewidth=2))
            ax.text(5.5, 5.2, '⚠️ FAULT DETECTED!', fontsize=12, fontweight='bold', 
                    ha='center', va='center', color='#f44336')
        
        gen_color = status_colors[components['generator'].status]
        ax.add_patch(Circle((1, 4), 0.3, facecolor=gen_color, edgecolor='black', linewidth=2))
        ax.text(1, 4, 'G', fontsize=12, fontweight='bold', ha='center', va='center')
        ax.text(1, 3.5, f'{components["generator"].current_value:.0f}MW', 
                fontsize=9, ha='center', va='center')
        ax.text(1, 3.2, 'Generator', fontsize=9, ha='center', va='center', fontweight='bold')
        
        line_color = status_colors[components['transmission_line'].status]
        ax.plot([1.5, 5.5], [4, 4], color=line_color, linewidth=4)
        for tower_x in [2.5, 3.5, 4.5]:
            ax.plot([tower_x, tower_x], [3.8, 4.2], color='black', linewidth=2)
            ax.plot([tower_x-0.1, tower_x+0.1], [4.2, 4.2], color='black', linewidth=2)
        ax.text(3.5, 4.3, 'Transmission Line\n161kV', fontsize=9, ha='center', va='center', fontweight='bold')
        ax.text(3.5, 3.7, f'{components["transmission_line"].current_value:.0f}A', 
                fontsize=8, ha='center', va='center')
        
        trans1_color = status_colors[components['stepdown_transformer_1'].status]
        ax.add_patch(Circle((6.2, 4.15), 0.15, facecolor=trans1_color, edgecolor='black', linewidth=2))
        ax.add_patch(Circle((6.4, 4.15), 0.15, facecolor=trans1_color, edgecolor='black', linewidth=2))
        ax.text(6.3, 4.4, 'T1', fontsize=9, ha='center', va='center', fontweight='bold')
        ax.text(6.3, 3.7, f'{components["stepdown_transformer_1"].current_value:.1f}kV', 
                fontsize=8, ha='center', va='center')
        
        trans2_color = status_colors[components['stepdown_transformer_2'].status]
        ax.add_patch(Circle((7.9, 4.15), 0.15, facecolor=trans2_color, edgecolor='black', linewidth=2))
        ax.add_patch(Circle((8.1, 4.15), 0.15, facecolor=trans2_color, edgecolor='black', linewidth=2))
        ax.text(8, 4.4, 'T2', fontsize=9, ha='center', va='center', fontweight='bold')
        ax.text(8, 3.7, f'{components["stepdown_transformer_2"].current_value:.0f}V', 
                fontsize=8, ha='center', va='center')
        
        cb_color = status_colors[components['breaker'].status]
        if components['breaker'].breaker_closed:
            ax.add_patch(Rectangle((8.7, 3.95), 0.3, 0.1, facecolor=cb_color, edgecolor='black', linewidth=2))
            ax.text(8.85, 4.2, 'CB', fontsize=8, ha='center', va='center', fontweight='bold')
            ax.text(8.85, 3.7, 'CLOSED', fontsize=7, ha='center', va='center', color='green')
        else:
            ax.add_patch(Rectangle((8.7, 3.95), 0.15, 0.1, facecolor='white', edgecolor='black', linewidth=2))
            ax.add_patch(Rectangle((8.85, 3.95), 0.15, 0.1, facecolor='white', edgecolor='black', linewidth=2))
            ax.text(8.85, 4.2, 'CB', fontsize=8, ha='center', va='center', fontweight='bold')
            ax.text(8.85, 3.7, 'OPEN', fontsize=7, ha='center', va='center', color='red')
        
        relay_color = status_colors[components['relay'].status]
        ax.add_patch(Circle((8.5, 3.4), 0.1, facecolor=relay_color, edgecolor='black', linewidth=2))
        ax.text(8.5, 3.4, 'R', fontsize=8, ha='center', va='center', fontweight='bold')
        ax.text(8.5, 3.1, 'Relay', fontsize=8, ha='center', va='center')
        
        load_color = status_colors[components['load'].status]
        triangle = Polygon([(9.7, 4.3), (10, 4), (10.3, 4.3)], 
                          facecolor=load_color, edgecolor='black', linewidth=2)
        ax.add_patch(triangle)
        ax.text(10, 3.7, 'Load', fontsize=9, ha='center', va='center', fontweight='bold')
        ax.text(10, 3.4, f'{components["load"].current_value:.0f}MW', 
                fontsize=8, ha='center', va='center')
        
        ax.plot([1.3, 5.5], [4, 4], color='black', linewidth=2)
        ax.plot([5.5, 6.15], [4, 4.15], color='black', linewidth=2)
        ax.plot([6.45, 7.75], [4.15, 4.15], color='black', linewidth=2)
        ax.plot([8.25, 8.7], [4.15, 4], color='black', linewidth=2)
        ax.plot([9.0, 9.7], [4, 4.3], color='black', linewidth=2)
        ax.plot([8.5, 8.5], [3.4, 4.0], color='black', linewidth=1, linestyle='--')
        
        if st.session_state.sim_data['power_flow']:
            power_flow = st.session_state.sim_data['power_flow'][-1]
            if power_flow > 0:
                ax.arrow(2.5, 4, 0.5, 0, head_width=0.1, head_length=0.2, fc='blue', ec='blue')
                ax.text(2.75, 4.2, f'{power_flow:.1f}MW', fontsize=8, ha='center', va='center', color='blue')
        
        legend_elements = [
            patches.Patch(facecolor='#4CAF50', edgecolor='black', label='Normal'),
            patches.Patch(facecolor='#FF9800', edgecolor='black', label='Warning'),
            patches.Patch(facecolor='#F44336', edgecolor='black', label='Fault'),
            patches.Patch(facecolor='#9E9E9E', edgecolor='black', label='Tripped')
        ]
        ax.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=8)
        
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error rendering single line diagram: {str(e)}")
        return None

# Page: Real-time Monitoring
if st.session_state.current_page == "Real-time Monitoring":
    st.header("📊 Real-time Monitoring Dashboard")
    
    st.button("Select Transformer Rating", key="transformer_selection_button", 
              on_click=lambda: st.session_state.__setitem__('previous_page', 'Real-time Monitoring') or st.session_state.__setitem__('current_page', 'Transformer Selection'))
    
    # Model selection
    model_choice = st.selectbox("Prediction Model", ["Random Forest", "Neural Network"], key="rt_model_choice")
    
    # Load models
    models_dict = load_models(st.session_state.transformer_rating)
    
    if models_dict is None:
        st.error("Model initialization failed. Please select a transformer rating and ensure model files exist.")
    elif st.session_state.running:
        # Generate new data
        voltage, current, temp, oil, vib, pf, freq, harm, load, fault_detected, relay_tripped, power_flow, bus_voltages, line_loadings, fault_type = generate_realtime_data(12, st.session_state.transformer_rating)
        
        # Make prediction
        try:
            input_data = [[voltage, current, temp, oil, vib, pf, freq, harm, load]]
            scaler = models_dict['scaler']
            model = models_dict[model_choice]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            fault_detected = (prediction != 0)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            prediction = fault_type
        
        # Calculate severity
        fuzzy_system, fuzzy_inputs = create_fuzzy_system(st.session_state.transformer_rating)
        if fuzzy_system:
            try:
                fuzzy_system.input['voltage'] = voltage
                fuzzy_system.input['current'] = min(max(current, fuzzy_inputs['current'].universe[0]), fuzzy_inputs['current'].universe[-1])
                fuzzy_system.input['temperature'] = temp
                fuzzy_system.input['oil_level'] = oil
                fuzzy_system.input['vibration'] = vib
                fuzzy_system.input['power_factor'] = pf
                fuzzy_system.input['frequency'] = freq
                fuzzy_system.input['harmonics'] = harm
                fuzzy_system.compute()
                severity = fuzzy_system.output['severity']
            except Exception as e:
                st.error(f"Fuzzy system error: {str(e)}")
                severity = 0
        else:
            severity = 0
        
        relay_tripped = fault_detected and (severity > 80)
        
        # Store data
        st.session_state.sim_data['time'].append(len(st.session_state.sim_data['time']))
        st.session_state.sim_data['voltage'].append(voltage)
        st.session_state.sim_data['current'].append(current)
        st.session_state.sim_data['temperature'].append(temp)
        st.session_state.sim_data['oil_level'].append(oil)
        st.session_state.sim_data['vibration'].append(vib)
        st.session_state.sim_data['power_factor'].append(pf)
        st.session_state.sim_data['frequency'].append(freq)
        st.session_state.sim_data['harmonics'].append(harm)
        st.session_state.sim_data['load_factor'].append(load)
        st.session_state.sim_data['fault_type'].append(prediction)
        st.session_state.sim_data['severity'].append(severity)
        st.session_state.sim_data['alarm'].append(severity > 40 or fault_detected)
        st.session_state.sim_data['fault_detected'].append(fault_detected)
        st.session_state.sim_data['relay_tripped'].append(relay_tripped)
        st.session_state.sim_data['power_flow'].append(power_flow)
        st.session_state.sim_data['bus_voltages'].append(bus_voltages)
        st.session_state.sim_data['line_loadings'].append(line_loadings)
    
    # Display metrics and charts
    if st.session_state.sim_data['time']:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Voltage", f"{st.session_state.sim_data['voltage'][-1]:.1f} V")
        with col2:
            st.metric("Current", f"{st.session_state.sim_data['current'][-1]:.1f} A")
        with col3:
            st.metric("Temperature", f"{st.session_state.sim_data['temperature'][-1]:.1f} °C")
        with col4:
            st.metric("Fault Severity", f"{st.session_state.sim_data['severity'][-1]:.1f}%")
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                           subplot_titles=("Voltage Profile", "Current Flow", "Temperature Trend"),
                           vertical_spacing=0.1)
        
        time_data = st.session_state.sim_data['time']
        
        fig.add_trace(
            go.Scatter(x=time_data, y=st.session_state.sim_data['voltage'], 
                      name="Voltage", line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_data, y=st.session_state.sim_data['current'], 
                      name="Current", line=dict(color='green')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_data, y=st.session_state.sim_data['temperature'], 
                      name="Temperature", line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced Single Line Diagram
        st.subheader("Power System Single Line Diagram")
        fig_sld = create_interactive_single_line_diagram(
            fault_detected=st.session_state.sim_data['fault_detected'][-1],
            relay_tripped=st.session_state.sim_data['relay_tripped'][-1],
            severity=st.session_state.sim_data['severity'][-1],
            voltage=st.session_state.sim_data['voltage'][-1],
            current=st.session_state.sim_data['current'][-1],
            load=st.session_state.sim_data['load_factor'][-1]*50
        )
        if fig_sld:
            st.pyplot(fig_sld)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.sim_data['power_factor'][-1],
                title={'text': "Power Factor"},
                gauge={'axis': {'range': [0.6, 1]},
                       'steps': [{'range': [0.6, 0.9], 'color': "lightgray"},
                                 {'range': [0.9, 1], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.85}}
            )), use_container_width=True)
        
        with col2:
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.sim_data['harmonics'][-1],
                title={'text': "Harmonics (THD %)"},
                gauge={'axis': {'range': [0, 20]},
                       'steps': [{'range': [0, 5], 'color': "lightgreen"},
                                 {'range': [5, 10], 'color': "lightyellow"},
                                 {'range': [10, 20], 'color': "lightpink"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 8}}
            )), use_container_width=True)
        
        with col3:
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=st.session_state.sim_data['load_factor'][-1] * 100,
                title={'text': "Load Factor (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'steps': [{'range': [0, 40], 'color': "lightpink"},
                                 {'range': [40, 70], 'color': "lightyellow"},
                                 {'range': [70, 100], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}}
            )), use_container_width=True)
        
        if st.session_state.running:
            time.sleep(0.5)
            st.rerun()
    else:
        st.info("No data available. Start the simulation to begin monitoring.")
    
    if not st.session_state.running:
        st.warning("Simulation is paused. Click 'Start' in the sidebar to begin.")

# Page: Transformer Protection (unchanged from your code)
elif st.session_state.current_page == "Transformer Protection":
    st.header("🛡️ Transformer Protection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Protection Status")
        
        if st.session_state.sim_data['time']:
            severity = st.session_state.sim_data['severity'][-1]
            fault_type = fault_labels.get(st.session_state.sim_data['fault_type'][-1], "Normal")
            
            if severity > 80:
                status_class = "alarm-critical"
                status_text = "🚨 CRITICAL ALERT"
                action = "Immediate shutdown required"
            elif severity > 40:
                status_class = "alarm-warning"
                status_text = "⚠️ WARNING"
                action = "Investigate and monitor closely"
            else:
                status_class = "status-normal"
                status_text = "✅ NORMAL OPERATION"
                action = "No action required"
            
            st.markdown(f'<div class="metric-card {status_class}">{status_text}</div>', unsafe_allow_html=True)
            st.metric("Current Fault Type", fault_type)
            st.metric("Recommended Action", action)
            
            st.subheader("Protection Relay")
            if st.session_state.sim_data['relay_tripped'][-1]:
                st.error("🔴 Relay Tripped - Circuit Breaker Open")
            else:
                st.success("🟢 Relay Normal - Circuit Closed")
            
            st.subheader("Transformer Metrics")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Oil Level", f"{st.session_state.sim_data['oil_level'][-1]:.1f}%")
                st.metric("Vibration", f"{st.session_state.sim_data['vibration'][-1]:.2f} mm/s")
            with col_b:
                st.metric("Temperature", f"{st.session_state.sim_data['temperature'][-1]:.1f} °C")
                st.metric("Load Factor", f"{st.session_state.sim_data['load_factor'][-1]*100:.1f}%")
    
    with col2:
        st.subheader("Fault Severity Analysis")
        
        if st.session_state.sim_data['time']:
            fig_severity = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=st.session_state.sim_data['severity'][-1],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fault Severity (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': st.session_state.sim_data['severity'][-1]}}
            ))
            fig_severity.update_layout(height=300)
            st.plotly_chart(fig_severity, use_container_width=True)
        
        if len(st.session_state.sim_data['severity']) > 1:
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                x=st.session_state.sim_data['time'],
                y=st.session_state.sim_data['severity'],
                mode='lines+markers',
                name='Severity',
                line=dict(color='royalblue', width=2))
            )
            fig_history.add_hrect(y0=50, y1=80, line_width=0, fillcolor="yellow", opacity=0.2)
            fig_history.add_hrect(y0=80, y1=100, line_width=0, fillcolor="red", opacity=0.2)
            fig_history.update_layout(
                title="Historical Severity Trend",
                xaxis_title="Time",
                yaxis_title="Severity (%)",
                height=300
            )
            st.plotly_chart(fig_history, use_container_width=True)
        
        if st.session_state.sim_data['fault_type']:
            fault_counts = {}
            for fault in st.session_state.sim_data['fault_type']:
                label = fault_labels.get(fault, "Normal")
                fault_counts[label] = fault_counts.get(label, 0) + 1
            
            fig_dist = px.pie(
                values=list(fault_counts.values()),
                names=list(fault_counts.keys()),
                title="Fault Type Distribution",
                height=300
            )
            st.plotly_chart(fig_dist, use_container_width=True)

# Page: Power System Analysis (unchanged from your code)
elif st.session_state.current_page == "Power System Analysis":
    st.header("🔌 Power System Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Load Flow Analysis", "Stability Assessment", "Fault Analysis"])
    
    with tab1:
        st.subheader("Load Flow Analysis")
        
        bus_data = {
            "Bus": [1, 2, 3],
            "Voltage (kV)": [161.0, 33.0, 0.433],
            "Angle (deg)": [0.0, -1.2, -2.5],
            "Load (MW)": [0, 0, 50],
            "Generation (MW)": [200, 0, 0]
        }
        
        st.dataframe(pd.DataFrame(bus_data), hide_index=True)
        
        fig_voltage = px.bar(
            pd.DataFrame(bus_data), 
            x="Bus", 
            y="Voltage (kV)",
            title="Bus Voltage Profile",
            color="Voltage (kV)",
            color_continuous_scale='RdYlGn',
            range_color=[0, 170]
        )
        fig_voltage.update_layout(height=400)
        st.plotly_chart(fig_voltage, use_container_width=True)
    
    with tab2:
        st.subheader("Stability Assessment")
        
        time_points = np.arange(0, 10, 0.1)
        rotor_angle = 30 + 5 * np.sin(time_points * 0.5)
        frequency = 50 + 0.5 * np.sin(time_points * 0.3)
        voltage_stability = 1.0 - 0.1 * np.sin(time_points * 0.4)
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        
        fig.add_trace(
            go.Scatter(x=time_points, y=rotor_angle, name="Rotor Angle (deg)", line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time_points, y=voltage_stability, name="Voltage Stability (p.u.)", line=dict(color='red')),
            row=3, col=1
        )
        
        fig.update_layout(
            title="Power System Stability Metrics",
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rotor Angle Deviation", f"{np.max(rotor_angle)-30:.2f} deg")
        col2.metric("Frequency Deviation", f"{np.max(frequency)-50:.2f} Hz")
        col3.metric("Voltage Deviation", f"{(1-np.min(voltage_stability))*100:.1f}%")
    
    with tab3:
        st.subheader("Fault Analysis")
        
        st.write("**Fault Current Calculation**")
        col1, col2 = st.columns(2)
        base_mva = col1.number_input("Base MVA", value=100.0)
        base_kv = col1.number_input("Base kV", value=161.0)
        impedance = col2.number_input("Impedance (p.u.)", value=0.1, min_value=0.001)
        
        try:
            base_current = base_mva * 1e6 / (np.sqrt(3) * base_kv * 1e3)
            fault_current = base_current / impedance
            st.metric("Fault Current", f"{fault_current/1e3:.1f} kA")
        except ZeroDivisionError:
            st.error("Impedance cannot be zero. Please enter a valid value.")
        
        st.subheader("Fault Location Probability")
        
        locations = ["Generator", "Transmission Line", "Transformer 1", "Transformer 2", "Load"]
        probabilities = [0.05, 0.40, 0.25, 0.20, 0.10]
        
        fig_fault_loc = px.bar(
            x=locations, 
            y=probabilities,
            title="Fault Location Probability",
            labels={"x": "Location", "y": "Probability"},
            color=probabilities,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_fault_loc, use_container_width=True)

# Page: Manual Prediction
elif st.session_state.current_page == "Manual Prediction":
    st.header("🔍 Manual Fault Prediction")
    
    st.button("Select Transformer Rating", key="transformer_selection_manual", 
              on_click=lambda: st.session_state.__setitem__('previous_page', 'Manual Prediction') or st.session_state.__setitem__('current_page', 'Transformer Selection'))
    
    model_choice = st.selectbox("Prediction Model", ["Random Forest", "Neural Network"], key="manual_model_choice")
    
    models_dict = load_models(st.session_state.transformer_rating)
    
    if models_dict is None:
        st.error("Model initialization failed. Please select a transformer rating and ensure model files exist.")
    else:
        voltage_nom = 433.0
        power_factor = 1
        if st.session_state.transformer_rating:
            power_va = st.session_state.transformer_rating * 1000
            current_nom = power_va / (np.sqrt(3) * voltage_nom * power_factor)
            current_min = current_nom * 0.5
            current_max = current_nom * 2.0
        else:
            current_nom = 100.0
            current_min = 50.0
            current_max = 200.0
        temperature_nom = 65.0
        temperature_max = 120.0
        oil_level_nom = 80.0
        oil_level_min = 20.0
        vibration_nom = 2.0
        vibration_max = 10.0
        power_factor_nom = 0.95
        power_factor_min = 0.6
        harmonics_nom = 2.0
        harmonics_max = 20.0
        load_factor_nom = 0.75
        load_factor_max = 1.0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            voltage_input = st.slider("Voltage (V)", min_value=200.0, max_value=600.0, value=voltage_nom, step=1.0)
            current_input = st.slider("Current (A)", min_value=float(current_min), max_value=float(current_max), value=float(current_nom), step=0.1)
            temperature_input = st.slider("Temperature (°C)", min_value=30.0, max_value=float(temperature_max), value=float(temperature_nom), step=1.0)
            oil_level_input = st.slider("Oil Level (%)", min_value=float(oil_level_min), max_value=100.0, value=float(oil_level_nom), step=1.0)
            vibration_input = st.slider("Vibration (mm/s)", min_value=0.0, max_value=float(vibration_max), value=float(vibration_nom), step=0.1)
        
        with col2:
            st.subheader("Additional Parameters")
            power_factor_input = st.slider("Power Factor", min_value=float(power_factor_min), max_value=1.0, value=float(power_factor_nom), step=0.01)
            frequency_input = st.slider("Frequency (Hz)", min_value=48.0, max_value=52.0, value=50.0, step=0.1, disabled=True)
            harmonics_input = st.slider("Harmonics (THD %)", min_value=0.0, max_value=float(harmonics_max), value=float(harmonics_nom), step=0.1)
            load_factor_input = st.slider("Load Factor", min_value=0.0, max_value=float(load_factor_max), value=float(load_factor_nom), step=0.01)
        
        if st.button("Predict Fault", use_container_width=True):
            input_data = [[voltage_input, current_input, temperature_input, oil_level_input, vibration_input, 
                          power_factor_input, frequency_input, harmonics_input, load_factor_input]]
            try:
                scaler = models_dict['scaler']
                model = models_dict[model_choice]
                scaled_input = scaler.transform(input_data)
                prediction = model.predict(scaled_input)[0]
                fault_detected = (prediction != 0)
                severity = 0
                
                if fault_detected:
                    fuzzy_system, fuzzy_inputs = create_fuzzy_system(st.session_state.transformer_rating)
                    if fuzzy_system:
                        fuzzy_system.input['voltage'] = voltage_input
                        fuzzy_system.input['current'] = min(max(current_input, fuzzy_inputs['current'].universe[0]), fuzzy_inputs['current'].universe[-1])
                        fuzzy_system.input['temperature'] = temperature_input
                        fuzzy_system.input['oil_level'] = oil_level_input
                        fuzzy_system.input['vibration'] = vibration_input
                        fuzzy_system.input['power_factor'] = power_factor_input
                        fuzzy_system.input['frequency'] = frequency_input
                        fuzzy_system.input['harmonics'] = harmonics_input
                        fuzzy_system.compute()
                        severity = fuzzy_system.output['severity']
                
                relay_tripped = fault_detected and (severity > 80)
                
                st.subheader("Prediction Results")
                
                if not fault_detected:
                    st.success(f"✅ **Status:** Normal Operation")
                    st.metric("Fault Severity", "N/A (No Fault)")
                else:
                    st.error(f"🚨 **Fault Detected:** {fault_labels[prediction]}")
                    st.metric("Fault Severity", f"{severity:.1f}%")
                    
                    fig_severity = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=severity,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Fault Severity"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': severity}}
                    ))
                    fig_severity.update_layout(height=300)
                    st.plotly_chart(fig_severity, use_container_width=True)
                
                st.subheader("Power System Single Line Diagram")
                fig_sld = create_interactive_single_line_diagram(
                    fault_detected=fault_detected,
                    relay_tripped=relay_tripped,
                    severity=severity,
                    voltage=voltage_input,
                    current=current_input,
                    load=load_factor_input*50
                )
                if fig_sld:
                    st.pyplot(fig_sld)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# Page: Historical Data (unchanged from your code)
elif st.session_state.current_page == "Historical Data":
    st.header("📈 Historical Data Analysis")
    
    if st.session_state.sim_data['time']:
        df = pd.DataFrame(st.session_state.sim_data)
        
        col1, col2 = st.columns(2)
        min_time = min(df['time']) if not df.empty else 0
        max_time = max(df['time']) if not df.empty else 100
        
        start_time = col1.number_input("Start Time", min_value=float(min_time), max_value=float(max_time), value=float(min_time))
        end_time = col2.number_input("End Time", min_value=float(min_time), max_value=float(max_time), value=float(max_time))
        
        filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        
        selected_params = st.multiselect(
            "Select Parameters to Visualize", 
            ['voltage', 'current', 'temperature', 'oil_level', 'vibration', 
             'power_factor', 'frequency', 'harmonics', 'load_factor', 'severity'],
            default=['voltage', 'current', 'temperature']
        )
        
        if selected_params:
            fig = go.Figure()
            for param in selected_params:
                fig.add_trace(go.Scatter(
                    x=filtered_df['time'],
                    y=filtered_df[param],
                    mode='lines',
                    name=param.capitalize()
                ))
            
            fig.update_layout(
                title="Parameter Trends Over Time",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500,
                legend_title="Parameters"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Analysis")
        corr_matrix = filtered_df[['voltage', 'current', 'temperature', 'oil_level', 'vibration', 
                                  'power_factor', 'frequency', 'harmonics', 'load_factor', 'severity']].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            title="Parameter Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(filtered_df.describe(), use_container_width=True)
        
        if st.button("Export Current View", use_container_width=True):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="power_system_historical_data.csv",
                mime="text/csv"
            )
    else:
        st.info("No historical data available. Run the simulation to generate data.")

# Page: System Configuration (modified to remove model retraining)
elif st.session_state.current_page == "System Configuration":
    st.header("⚙️ System Configuration")
    
    tab1, tab2 = st.tabs(["Simulation Parameters", "Protection Settings"])
    
    with tab1:
        st.subheader("Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Fault Simulation**")
            fault_prob = st.slider("Fault Probability (%)", 0, 50, 12)
            fault_duration = st.slider("Average Fault Duration (cycles)", 1, 100, 5)
        
        with col2:
            st.write("**Protection Settings**")
            voltage_threshold = st.number_input("Voltage Threshold (V)", value=475.0)
            current_threshold = st.number_input("Current Threshold (A)", value=150.0)
            temp_threshold = st.number_input("Temperature Threshold (°C)", value=80.0)
        
        st.write("**Relay Settings**")
        relay_time = st.slider("Relay Response Time (ms)", 1, 100, 20)
        relay_curve = st.selectbox("Relay Time-Current Curve", ["Standard Inverse", "Very Inverse", "Extremely Inverse"])
        
        if st.button("Save Configuration", use_container_width=True):
            st.success("Configuration saved successfully!")
    
    with tab2:
        st.subheader("Model Information")
        st.info("Using pre-trained Random Forest and Neural Network models. To update models, run the training script separately.")

# Page: Transformer Selection
elif st.session_state.current_page == "Transformer Selection":
    st.header("🔧 Transformer Rating Selection")
    
    st.markdown("""
    <div class="transformer-selection">
        <h3>Select Transformer Rating for Simulation</h3>
        <p>Choose the transformer capacity to adjust the simulation parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("50 kVA", key="50kva", help="Select 50 kVA transformer rating", 
                     type="primary", use_container_width=True):
            st.session_state.transformer_rating = 50
            st.session_state.current_page = st.session_state.previous_page
            load_models(50)
            st.rerun()
    
    with col2:
        if st.button("100 kVA", key="100kva", help="Select 100 kVA transformer rating", 
                     type="primary", use_container_width=True):
            st.session_state.transformer_rating = 100
            st.session_state.current_page = st.session_state.previous_page
            load_models(100)
            st.rerun()
    
    with col3:
        if st.button("200 kVA", key="200kva", help="Select 200 kVA transformer rating", 
                     type="primary", use_container_width=True):
            st.session_state.transformer_rating = 200
            st.session_state.current_page = st.session_state.previous_page
            load_models(200)
            st.rerun()
    
    if st.session_state.transformer_rating:
        st.success(f"Selected Transformer Rating: {st.session_state.transformer_rating} kVA")
        try:
            current = st.session_state.transformer_rating * 1000 / (np.sqrt(3) * 433)
            st.metric("Nominal Current", f"{current:.1f} A")
        except Exception as e:
            st.error(f"Error calculating nominal current: {str(e)}")
    
    st.markdown("""
    <div style='margin-top: 2rem;'>
        <p><strong>Note:</strong> The selected transformer rating will affect the current calculations using the formula P = √3 × V × I × cosφ, where:</p>
        <ul>
            <li>P is the power rating (kVA)</li>
            <li>V is the line voltage (433V)</li>
            <li>I is the line current</li>
            <li>cosφ is the power factor (assumed 1)</li>
        </ul>
        <p>After selecting a transformer rating, the simulation will use the corresponding current values.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Back to Previous Page", use_container_width=True):
        st.session_state.current_page = st.session_state.previous_page
        st.rerun()

# Page: API Key Management (unchanged from your code)
elif st.session_state.current_page == "API Key Management":
    st.header("🔐 API Key Management")
    
    st.markdown("""
    <div class="api-key-container">
        <h3>Manage Your API Key</h3>
        <p>Generate or view your API key for accessing the Power System Protection Suite API.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.api_key:
        st.subheader("Your Current API Key")
        st.markdown(f'<div class="api-key">{st.session_state.api_key}</div>', unsafe_allow_html=True)
        st.warning("Keep your API key secure. Do not share it publicly.")
        
        if st.button("Regenerate API Key", use_container_width=True):
            st.session_state.api_key = str(uuid.uuid4())
            st.success("New API key generated successfully!")
            st.rerun()
    else:
        if st.button("Generate API Key", use_container_width=True):
            st.session_state.api_key = str(uuid.uuid4())
            st.success("API key generated successfully!")
            st.rerun()
    
    st.markdown("""
    <div style='margin-top: 2rem;'>
        <p><strong>Note:</strong> The API key allows access to the Power System Protection Suite API for external integrations. For more information on using the API, visit <a href='https://x.ai/api' target='_blank'>xAI API Documentation</a>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Back to Previous Page", use_container_width=True):
        st.session_state.current_page = st.session_state.previous_page
        st.rerun()
