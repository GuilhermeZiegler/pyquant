import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import streamlit as st

class CandleTensor:
    def __init__(self, df: pd.DataFrame, tensor_size: int = 1):
        self.df = df
        self.tensor_size = tensor_size

    def candle_to_tensor(self):
        data = []
        labels = []

        self.df.columns = [col.lower() for col in self.df.columns]

        for i in range(len(self.df) - self.tensor_size + 1):
            sequence = []
            data_range = self.df.iloc[i:i + self.tensor_size]
            
            for j in range(self.tensor_size):
                open_price = data_range['open'].iloc[j]
                high = data_range['high'].iloc[j]
                low = data_range['low'].iloc[j]
                close = data_range['close'].iloc[j]

                min_val = min(open_price, high, low, close)
                max_val = max(open_price, high, low, close)

                # Normalize each value within the candle
                if max_val != min_val:
                    normalized_open = (open_price - min_val) / (max_val - min_val)
                    normalized_high = (high - min_val) / (max_val - min_val)
                    normalized_low = (low - min_val) / (max_val - min_val)
                    normalized_close = (close - min_val) / (max_val - min_val)
                else:
                    normalized_open = normalized_high = normalized_low = normalized_close = 0.0

                sequence.append([normalized_open, normalized_high, normalized_low, normalized_close])

            data.append(sequence)
            labels.append(self.df.index[i + self.tensor_size - 1])

        tensor_data = tf.convert_to_tensor(data, dtype=tf.float32)
        return tensor_data, labels

    def plot_candle_from_tensor(self, tensor_data):
        fig, ax = plt.subplots(figsize=(self.tensor_size, self.tensor_size))
        sequence = tensor_data[0].numpy()  # Extract the tensor data
        
        if sequence.size == 0:
            return None
        
        for j in range(sequence.shape[0]):
            open_price, high, low, close = sequence[j]

            color = 'green' if close >= open_price else 'red'
            line_color = 'black'

            ax.plot([j + 1, j + 1], [low, high], color=line_color, linewidth=1)
            ax.plot([j + 1, j + 1], [open_price, close], color=color, linewidth=8)

        ax.set_xlim(0.5, sequence.shape[0] + 0.5)
        ax.set_ylim(min(sequence[:, 2]), max(sequence[:, 1]))
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)

def main():
    st.title('Candle Classifier')
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-16', delimiter=',')
        df.columns = [col.lower() for col in df.columns]
        if df.columns[0] == 'date':
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), format='%Y.%m.%d %H:%M')
            df.drop(columns=['date', 'time'], inplace=True)
        df.set_index('datetime', inplace=True, drop=True)
        
        default_sample_size = 1000
        sample_size = st.slider("Select sample size", min_value=1, max_value=len(df), value=default_sample_size)
        
        options = ["Random", "End to Start", "Start to End"]
        order_option = st.selectbox("Select data order", options)
        
        tensor_size = st.number_input("Select tensor size", min_value=1, max_value=10, value=1, step=1)
        
        pattern = st.text_input("Enter pattern name", value='hammer')
        base_path = f"C:\\Users\\guitz\\OneDrive\\Área de Trabalho\\pyquant\\pyquant\\classifications\\{pattern}"
        os.makedirs(base_path, exist_ok=True)

        if st.button("Process Data"):
            if order_option == "Random":
                sampled_indices = np.random.choice(len(df), size=sample_size, replace=False)
                df_sample = df.iloc[sampled_indices]
            elif order_option == "End to Start":
                sampled_indices = np.arange(len(df) - sample_size, len(df))
                df_sample = df.iloc[sampled_indices]
            else:
                sampled_indices = np.arange(sample_size)
                df_sample = df.iloc[sampled_indices]
            
            candle_tensor = CandleTensor(df_sample, tensor_size)
            tensor_data, labels = candle_tensor.candle_to_tensor()
            
            # Save tensor_data, labels, and candle_tensor instance to session state
            st.session_state.tensor_data = tensor_data
            st.session_state.labels = labels
            st.session_state.current_index = 0
            st.session_state.classification = []
            st.session_state.candle_tensor = candle_tensor

            st.write("Data processing complete. Click 'Classify Image' to start.")
            st.write(st.session_state.candle_tensor )       
        if 'tensor_data' in st.session_state and st.session_state.tensor_data is not None:
            if st.button("Classify Image"):
                tensor_data = st.session_state.tensor_data
                current_index = st.session_state.current_index
                
                # Get the current tensor and plot it
                img = st.session_state.candle_tensor.plot_candle_from_tensor(tensor_data[current_index:current_index + 1])
                if img is not None:
                    st.image(img, caption=f"Image {current_index}")

                    # Classification buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Classify as 1"):
                            st.session_state.classification.append(1)
                            st.session_state.current_index += 1
                    with col2:
                        if st.button("Classify as 0"):
                            st.session_state.classification.append(0)
                            st.session_state.current_index += 1

                    # Check if there are more images to classify
                    if st.session_state.current_index >= len(tensor_data):
                        st.write("All images classified.")
                        # Save classified images
                        for idx, classification in enumerate(st.session_state.classification):
                            img = st.session_state.candle_tensor.plot_candle_from_tensor(tensor_data[idx:idx + 1])
                            if img is not None:
                                img.save(os.path.join(base_path, f"{idx}_classified_{classification}.png"))
                        st.write(f"Classifications saved to: {base_path}")

if __name__ == "__main__":
    main()
