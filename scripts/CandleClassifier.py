import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from io import BytesIO
from PIL import Image
import pickle

def normalize_candles(ohlc_data):
    """
    Function to normalize and scale OHLC data.
    
    Steps:
    1. Find the smallest value in the OHLC data.
    2. Divide all values by the smallest value and subtract 1.
    3. Scale the resulting values to the range 0 to 1.
    4. Round the values to 5 decimal places.

    Parameters:
    - ohlc_data: numpy array containing the OHLC data.

    Returns:
    - scaled_values_rounded: array with normalized, scaled, and rounded values.
    """
    # Step 1: Find the smallest value
    min_value = ohlc_data.min()

    # Step 2: Divide all data by the smallest value and subtract 1
    normalized_values = ohlc_data / min_value - 1

    # Step 3: Scale the resulting values to the range 0 to 1
    min_normalized = normalized_values.min()
    max_normalized = normalized_values.max()

    scaled_values = (normalized_values - min_normalized) / (max_normalized - min_normalized)

    # Step 4: Round the scaled values to 5 decimal places
    scaled_values_rounded = np.round(scaled_values, 5)

    return scaled_values_rounded

def plot_normalized_candlestick(candle_vectors):
    fig, ax = plt.subplots(figsize=(len(candle_vectors), len(candle_vectors)))
    width = 0.2 
    spacing = 0.3 

    for i in range(len(candle_vectors)):
        candle_vector = candle_vectors[i]
        adjusted_open, adjusted_high, adjusted_low, adjusted_close = candle_vector
        
        x = i * spacing
        
        lower_shadow_min = min(adjusted_open, adjusted_close)
        ax.plot([x, x], [adjusted_low, lower_shadow_min], color='black', linewidth=1)
        
        upper_shadow_max = max(adjusted_open, adjusted_close)
        ax.plot([x, x], [upper_shadow_max, adjusted_high], color='black', linewidth=1)
        
        color = 'green' if adjusted_close - adjusted_open >= 0 else 'red'
        
        if np.round(adjusted_close - adjusted_open, 5) == 0:
            ax.plot(
                [x - width / 2, x + width / 2], 
                [adjusted_close, adjusted_close], 
                color='black', 
                linewidth=1
            )
        else:
            rect = patches.Rectangle(
                (x - width / 2, min(adjusted_open, adjusted_close)), 
                width, 
                abs(adjusted_open - adjusted_close), 
                linewidth=0, 
                edgecolor='white', 
                facecolor=color
            )
            ax.add_patch(rect)
    
    margin = 0.1 
    
    adjusted_max_high = max([candle[1] for candle in candle_vectors])
    adjusted_min_low = min([candle[2] for candle in candle_vectors])
    
    ax.set_xlim(-width, len(candle_vectors) * spacing)
    ax.set_ylim(adjusted_min_low - margin, adjusted_max_high + margin)
    ax.grid(False)
    ax.set_xticks([])  
    ax.set_yticks([]) 
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_tick_params(labelleft=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def load_csv(uploaded_file, encoding='utf-8', delimiter=','):
    df = pd.read_csv(uploaded_file, encoding=encoding, delimiter=delimiter)
    df.columns = [col.lower() for col in df.columns]
    
    if 'date' in df.columns and 'time' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y.%m.%d %H:%M')
        df.drop(columns=['date', 'time'], inplace=True)
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'], format='%Y.%m.%d')
        df.drop(columns=['date'], inplace=True)
    
    df.set_index('datetime', inplace=True)
    
    return df


def load_patterns(pattern_type):
    pkl_file = f"{pattern_type.lower()}_patterns.pkl"
    pkl_path = os.path.join("pkl", pkl_file)  
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            patterns = pickle.load(f)
        return patterns
    return {}

def load_pkl(file_path):
    """loads .pkl file"""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_classified_candles(base_path_target, pattern_name, class_label, candle_size, vectors):
    """Save classified vectors to a PKL file in the target directory."""
    pattern_path = os.path.join(base_path_target, pattern_name, class_label)
    os.makedirs(pattern_path, exist_ok=True)
    pkl_file = f'{candle_size}_candles.pkl'
    
    pkl_path = os.path.join(pattern_path, pkl_file)
    
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            all_classified_vectors = pickle.load(f)
    else:
        all_classified_vectors = []

    all_classified_vectors.extend(vectors)
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_classified_vectors, f)
    
    st.success(f'Candle vectors saved at {pkl_path}')

def save_classified_image(buf, base_path_figures, pattern_name, class_label, candle_size):
    """Save the classified image to the figures directory with a unique index."""
    target_path = os.path.join(base_path_figures, 'target', pattern_name, class_label)
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    existing_files = [f for f in os.listdir(target_path) if f.endswith('.png')]
    indices = []
    for f in existing_files:
        try:
            prefix = f"{candle_size}_candles_"
            if f.startswith(prefix):
                index = int(f[len(prefix):-4])
                indices.append(index)
        except (IndexError, ValueError):
            continue
    
    next_index = max(indices, default=0) + 1
    image_file = f'{candle_size}_candles_{next_index}.png'
    image_path = os.path.join(target_path, image_file)
    with open(image_path, 'wb') as f:
        f.write(buf.getbuffer())

    st.success(f"Image saved at {image_path}")

def main():
    candle_info_patterns = load_pkl('C:\\Users\\guitz\\OneDrive\\Área de Trabalho\\pyquant\\pyquant\\pkl\\candles\\candle_patterns_info.pkl')
    base_path_classifications = 'C:\\Users\\guitz\\OneDrive\\Área de Trabalho\\pyquant\\pyquant\\classifications'
    base_path_target = 'C:\\Users\\guitz\\OneDrive\\Área de Trabalho\\pyquant\\pyquant\\pkl\\target'
    
    st.title('Candle Classifier')
    st.session_state.data = None

    if 'candle_vectors' not in st.session_state:
        st.session_state.candle_vectors = []
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'classification' not in st.session_state:
        st.session_state.classification = []
    if 'date' not in st.session_state:
        st.session_state.date = []

    st.markdown('### Upload CSV File')
    encoding = st.text_input("Enter CSV encoding", value='utf-16')
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        st.session_state.data = load_csv(uploaded_file, encoding=encoding, delimiter=',')
        df = st.session_state.data.copy()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            candle_size = st.number_input("Select number of candles per image", min_value=1, max_value=100, value=5, step=1)
        with col2:
            options = ["Random", "End to Start", "Start to End"]
            order_option = st.selectbox("Select data order", options)

        default_sample_size = 100
        sample_size = st.slider("Select sample size", min_value=1, max_value=len(df) - 1, value=default_sample_size)
        if st.button("Process"):
            max_start_index = len(df) - candle_size
            if order_option == "Random":
                sampled_indices = np.random.choice(np.arange(max_start_index + 1), size=sample_size, replace=False)
                df_index = sampled_indices
            elif order_option == "End to Start":
                sampled_indices = np.arange(len(df) - sample_size, len(df))
                df_index = sampled_indices
            else:
                sampled_indices = np.arange(sample_size)
                df_index = sampled_indices
                
            st.session_state.candle_vectors = []
            st.session_state.date = []
            st.session_state.current_index = 0

            for start in df_index:
                if start + candle_size <= len(df):
                    
                    segment = df.iloc[start:start + candle_size].sort_index(ascending=True)
                    ohlc_data = segment[['open', 'high', 'low', 'close']].values

                    candle_vector = normalize_candles(ohlc_data)
                  
                    if candle_vector is not None and not any(np.isnan(np.concatenate(candle_vector))):
                        st.session_state.candle_vectors.append(candle_vector)
                        st.session_state.date.append(df.index[start])
                        st.session_state.current_index = 0  

        if st.session_state.data is not None:
            st.divider()
            st.markdown('### Classification Tool')

            pattern_type = st.selectbox("Select pattern type", ["Bullish", "Bearish"])
            pattern_type = pattern_type.lower()  # Convert to lowercase
            
            filtered_patterns = {name: info for name, info in candle_info_patterns.items() if info['Type'] == pattern_type}
            
            if filtered_patterns:
                pattern_name = st.selectbox("Select pattern name", list(filtered_patterns.keys()))
                
                if pattern_name:
                    with st.expander(pattern_name, expanded=True):
                        pattern_info = filtered_patterns[pattern_name]
                        st.write(f"**Definition:** {pattern_info['Definition']}")
                        st.write(f"**Significance:** {pattern_info['Significance']}")
                        st.write(f"**Confidence:** {pattern_info['Confidence']}")

                    image_filename = pattern_info.get('Image', None)

                    if image_filename:
                        image_path = os.path.join(base_path_classifications, f"patterns/{pattern_type}/{image_filename}")
                        if os.path.exists(image_path):
                            image = Image.open(image_path)
                            with st.container():
                                st.image(image, caption=f"{pattern_name}", use_column_width=False, width=150)
                        else:
                            st.write("Image file not found.")
                    else:
                        st.write("Pattern name not found in images dictionary.")
            else:
                st.write(f"No patterns found for type '{pattern_type}'.")
                
            st.divider()
            col1, col2 = st.columns([1, 1])

            idx = st.session_state.current_index
            with col1:
                if st.button("Previous"):
                    if st.session_state.current_index > 0:
                        st.session_state.current_index -= 1

            with col2:
                if st.button("Next"):
                    if st.session_state.current_index < len(st.session_state.candle_vectors) - 1:
                        st.session_state.current_index += 1
            
            if 0 <= idx < len(st.session_state.candle_vectors):
                buf = plot_normalized_candlestick(st.session_state.candle_vectors[idx])
                col1, col2 = st.columns([1,1])
                with col2:
                    st.markdown('Current scaled vector')
                    st.write(st.session_state.date[idx])
                    st.dataframe(pd.DataFrame(st.session_state.candle_vectors[idx], columns=['open', 'high', 'low', 'close']))
                with col1:
                    with st.container():
                        if candle_size <= 2:
                            st.image(buf, caption=f"Candle Plot {idx + 1}", use_column_width=False, width=100)
                        else:
                            st.image(buf, caption=f"Candle Plot {idx + 1}", use_column_width=False, width=300)
            else:       
                st.write("Invalid index. Please check the data.")

            classify_pattern = st.selectbox("Select classification pattern", list(filtered_patterns.keys()))
            class_labels = st.radio("Select target", options=['0', '1'])
            
            if st.button("Save"):
                target_folder = 'target'
                save_path = os.path.join(base_path_classifications, target_folder, classify_pattern, class_labels)
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                save_classified_image(buf, base_path_classifications, classify_pattern, class_labels, candle_size)
                save_classified_candles(base_path_target, classify_pattern, class_labels, candle_size, [st.session_state.candle_vectors[idx]])

if __name__ == "__main__":
    main()