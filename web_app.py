import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from collections import Counter
import neurokit2 as nk
import numpy as np

#to run app: 'streamlit run web_app.py'
#Preprocess the file into 5 second segments       
def preprocess_segment(df, sample_rate=360):    
    label_mapping = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",  # Grouping into "N"
    "a": "S", "S": "S", "A": "S", "J": "S",  # Grouping into "S"
    "V": "V", "E": "V",  # Grouping into "V"
    "F": "F",  # Grouping into "F"
    "U": "Q",  "P": "Q", "f": "Q"  # Grouping into "Q"
    }
    df["Grouped_Label"] = df["Label"].map(label_mapping)
    r_peaks = df[df['Grouped_Label'].notna()].index.values
    
    segments = []
    labels = []
    r_locations = []
    
    for i in range(1, len(r_peaks)-1):
        prev_r = r_peaks[i-1]
        curr_r = r_peaks[i]
        next_r = r_peaks[i+1]
        
        start = prev_r + int(0.36*(curr_r - prev_r))
        end = curr_r + int(0.64*(next_r - curr_r))
        
        segment = df.iloc[start:end, 0].values
        label = df['Grouped_Label'].iloc[curr_r]
        r_locations.append(curr_r)
        
        segment = np.interp(np.linspace(0,1,360), np.linspace(0,1,len(segment)), segment)
        segments.append(segment)
        labels.append(label)
        
    X = np.array(segments)
    y = np.array(labels)
    
    return X, y, r_locations
                           
def calculate_heart_rate(df, sample_rate=360):
    r_peaks = df[df['Label'].notna()].index
    num_beats = len(r_peaks)                        
    duration_seconds = len(df) / sample_rate
    heart_rate = (num_beats / duration_seconds) * 60
    return round(heart_rate, 2)

def extract_features(ecg_signal, sample_rate=360): 
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sample_rate)
        ecg_cleaned = nk.signal_detrend(ecg_cleaned)
        
        # 2. Get R-peaks
        _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sample_rate)
        rpeaks = rpeaks_info["ECG_R_Peaks"]

        # 3. Delineate QRS, P, T waves
        signals, waves = nk.ecg_delineate(ecg_cleaned, rpeaks=rpeaks,sampling_rate=sample_rate, method="dwt")

        # Extract wave boundaries
        p_onsets_idx = np.where(signals["ECG_P_Onsets"] == 1)[0]
        q_peaks_idx = np.where(signals["ECG_Q_Peaks"] == 1)[0]
        s_peaks_idx = np.where(signals["ECG_S_Peaks"] == 1)[0]
        t_offsets_idx = np.where(signals["ECG_T_Offsets"] == 1)[0]

        # 4. Calculate intervals PER BEAT (aligned with R-peaks)
        pq_intervals = []
        qt_intervals = []
        qrs_durations = []
        
        if len(p_onsets_idx) > 0 and len(q_peaks_idx) > 0 and len(t_offsets_idx) > 0:
            p = p_onsets_idx[0]
            q = q_peaks_idx[0]
            s = s_peaks_idx[0] if len(s_peaks_idx) > 0 else None
            t = t_offsets_idx[0]
            r = rpeaks[0]

            # Intervals in milliseconds
            pq_interval = (q - p) / sample_rate * 1000  # P to Q
            pq_intervals.append(pq_interval)
            qrs_duration = (r - q) / sample_rate * 1000 * 2  # Q to S
            qrs_durations.append(qrs_duration)
            qt_interval = (t - q) / sample_rate * 1000  # Q to T
            qt_intervals.append(qt_interval)
        
        return {
            "PQ Interval (ms)": round(np.nanmedian(pq_intervals), 2) if pq_intervals else None,
            "QRS Duration (ms)": round(np.nanmedian(qrs_durations), 2) if qrs_durations else None,
            "QT Interval (ms)": round(np.nanmedian(qt_intervals), 2) if qt_intervals else None,
        }
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None

#Load the model
class HybridModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(HybridModel, self).__init__()            
            # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, stride=1, padding=1),  # Conv1D
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # MaxPooling1D
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # Conv1D
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # MaxPooling1D
        )            
        # LSTM layers
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)           
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64, 128),  # Dense layer
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout
            nn.Linear(128, num_classes),  # Output layer
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape for CNN: (batch_size, features, sequence_length)
        x = self.cnn(x)  # Output shape: (batch_size, 64, 90)
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, 90, 64) for LSTM
        x, _ = self.lstm(x)  # Output shape: (batch_size, 90, 64)  
        x = x[:, -1, :]  # Output shape: (batch_size, 64) to fully connected layer
        x = self.fc(x)  # Output shape: (batch_size, num_classes)
        
        return x

st.set_page_config(page_title="ECG Beat Classification", layout="wide")    
st.title("ECG Beat Classification Application")
st.markdown("""
            <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffc107;text-align: center;'>
            <strong>Disclaimer:</strong> This application is for demonstration and research purposes only and should not be used as a medical diagnosis. 
            The results are not clinically validated and you should consult a healthcare professional for medical advice.
            </div>
            """, unsafe_allow_html=True)
#start of the web app
col1, col2,col3 = st.columns([1, 2, 1])
with col2:    
    file = st.write('Please choose a file:')
    file = st.file_uploader('File uploader')

with st.sidebar:
        st.header('Guideline')
        
        st.markdown('ECG Beat Types')
        st.markdown("""
                    - N (Normal beat): Normal sinus rhythm, no irregularities.
                    - V (Ventricular): Ventricular ectopic beat, originating from the ventricles. A wide QRS complex is usually present.
                    - S (Supraventricular): Supraventricular ectopic beat, originating from above the ventricles. 
                    - F (Fusion): Fusion of Normal and Ventricular beats. A narrow QRS complex is usually present.
                    - Q (Unknown): Unclassifiable beat.
                    """)
        
        df_guide = pd.DataFrame({
            'Metric': ['Heart Rate','PQ Interval', 'QRS Duration', 'QT Interval'],
            'Typical Range': ['60-100 bpm','120-200 ms', '< 120 ms', '350-450 ms']
        })
        st.markdown('Typical Clinical Ranges')
        st.table(df_guide)
        # df_beats = pd.DataFrame({
        # 'Beat Type': ['N', 'V', 'S', 'F', 'Q'],
        # 'Colour': ['Green', 'Yellow', 'Orange', 'Red', 'Purple']
        # })

        # # Apply background color style to 'Colour' column
        # def highlight_colors(val):
        #     return f'background-color: {val.lower()}; color: black;'

        # styled_df = df_beats.style.applymap(highlight_colors, subset=['Colour'])

        # # Display
        # st.markdown("### ECG Beat Labels")
        # st.dataframe(styled_df, use_container_width=True)
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 15px !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > input,
    .stTextArea > div > textarea,
    .stSelectbox > div > div,
    .stButton > button {
        font-size: 13px !important;
        padding: 5px 8px !important;
    }
    .stDataFrame { font-size: 13px !important; }
    .stButton > button { padding: 5px 8px; font-size: 13px !important; }
    </style>
""", unsafe_allow_html=True)

#i want to visualise the uploaded file with a plot 
if file is not None:
    df = pd.read_csv(file)
    #st.write(df)
    #st.write('Here is a plot of the data:')   
    #st.plotly_chart(fig, use_container_width=True)
        
    #Test the function and display a table of the segments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y_true, r_locations = preprocess_segment(df)
    X_tensor = torch.tensor(X.reshape(-1,360,1), dtype=torch.float32).to(device)
    
    #start the model
    model = HybridModel(input_size=1, num_classes=5)
    model.load_state_dict(torch.load('hybrid_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    label_map = {0: 'F', 1: 'N', 2: 'Q', 3: 'S', 4: 'V'}
    predicted_classes = [label_map[p] for p in predictions]
    
    label_names = {
        'N': 'Normal',
        'S': 'Supraventricular',
        'V': 'Ventricular',
        'F': 'Fusion',
        'Q': 'Unknown'
    }
    label_counts = Counter(predicted_classes)
    label_counts = {label_names[label]: count for label, count in label_counts.items()}
    label_df = pd.DataFrame(list(label_counts.items()), columns=['Label', 'Count'])
    
    df_predictions = pd.DataFrame({   
        'Predicted Label': predicted_classes,
        'R-peak Location': r_locations,
    })
    # st.subheader('Predicted Beat Locations:')
    # st.dataframe(df_predictions)
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader('Predicted Beat Counts:')
        st.dataframe(label_df)
        
        #If you would like to view the actual segment labels, uncomment these 2 lines below
        st.write('True Label Counts (Segment-Level):')
        st.write(collections.Counter(y_true)) 
        
        heart_rate = calculate_heart_rate(df)
        #st.sidebar.metric("Heart Rate (bpm):", heart_rate)
        st.metric("Heart Rate (bpm)", calculate_heart_rate(df))
        
    with col_right:
        unique_labels = df_predictions['Predicted Label'].unique()
        default_selection = [label for label in unique_labels if label in ['N', 'V', 'S', 'F', 'Q']]
        selected_labels = st.multiselect("Select Beat Type", unique_labels, default=default_selection)
        filtered_df = df_predictions[df_predictions['Predicted Label'].isin(selected_labels)]
        st.subheader(f' Predicted Beat Locations: ({len(filtered_df)} beats)')
        st.dataframe(filtered_df)
    
    
    lead_name = df.columns[0]
    signal = df[lead_name].values
    _, signals = nk.ecg_peaks(signal, sampling_rate=360)

    r_peaks = signals['ECG_R_Peaks']
    
    window_size = 3600
    total_samples = len(df)

    
    if 'start_idx' not in st.session_state:
        st.session_state.start_idx = 0
    if 'search_r_peak' not in st.session_state:
        st.session_state.search_r_peak = ""
        
    col_left, col_centre,col_right = st.columns([1, 2, 1])
    with col_centre:
        
        col1, col2 = st.columns([6,1])    
        with col1:
            search_input = st.text_input("Search for R-peak location", value=st.session_state.search_r_peak)
            st.session_state.search_r_peak = search_input
        with col2:
            if st.button("Reset"):
                st.session_state.search_r_peak = ""
                st.session_state.start_idx = 0
                
        try:    
            if st.session_state.search_r_peak:
                val = int(st.session_state.search_r_peak)
                if val in r_locations:
                    st.session_state.start_idx = max(0, val - window_size // 2)
        except ValueError:
            st.error("Please enter a valid integer for R-peak location.")
            
    start_idx = st.slider(
        "Select starting sample index",
        min_value=0, 
        max_value=max(0, total_samples - window_size),
        step=window_size,
        value=st.session_state.start_idx,
        key="start_idx"
    )                
    
    segment = df.iloc[st.session_state.start_idx:st.session_state.start_idx+window_size]

    # Create the base plot
    fig = go.Figure(go.Scatter(
        x=segment.index,
        y=segment[lead_name],
        mode='lines',
        name='Lead'
    ))

    fig.update_layout(
        title='ECG Signal',
        xaxis_title='Time (samples)',
        yaxis_title='ECG Signal (mV)',
        height=500,
        width=600,
    )

    class_colours = {
        'N': 'green',
        'V': 'pink',
        'S': 'orange',
        'F': 'red',
        'Q': 'purple'
    }
    
    # Filter only R-peaks within the visible window
    for idx, label in zip(r_locations, predicted_classes):
        if start_idx <= idx < start_idx + window_size:
            fig.add_trace(go.Scatter(
                x=[idx], y=[df[lead_name][idx]],
                mode='markers+text',
                marker=dict(color=class_colours[label], size=10),
                text=[label],
                textposition='top center',
                name='Predicted Beat',
                showlegend=False
            ))
            
    for label, color in class_colours.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=color, size=14),
            name=label,
            showlegend=True
        ))        

    st.subheader('Annotated ECG Signal')
    st.plotly_chart(fig, use_container_width=True)            
    
                    

    
    




    


