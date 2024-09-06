import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import io

class CandleTensor:
    def __init__(self, df: pd.DataFrame, tensor_size: int = 1):
        self.df = df
        self.tensor_size = tensor_size

    def candle_to_tensor(self, df_sample):
        data = []
        labels = []
        df_sample.columns = [col.lower() for col in df_sample.columns]

        for i in range(len(df_sample) - self.tensor_size + 1):
            sequence = []
            data_range = df_sample.iloc[i:i + self.tensor_size]

            # Calculate min and max values for the current window
            min_vals = data_range[['open', 'high', 'low', 'close']].min()
            max_vals = data_range[['open', 'high', 'low', 'close']].max()

            # Scale data for the current window
            scaled_data = (data_range[['open', 'high', 'low', 'close']] - min_vals) / (max_vals - min_vals)
            
            for j in range(self.tensor_size):
                sequence.append(scaled_data.iloc[j].values)
            
            data.append(sequence)
            labels.append(df_sample.index[i + self.tensor_size - 1])

        tensor_data = tf.convert_to_tensor(data, dtype=tf.float32)
        return tensor_data, labels

    def plot_candle_from_tensor(self, tensor_data):
        fig, ax = plt.subplots(figsize=(self.tensor_size, self.tensor_size))
        sequence = tensor_data.numpy()
        
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
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img

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
        base_path = f"target/{pattern}"
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
            tensor_data, labels = candle_tensor.candle_to_tensor(df_sample)
            
            # Display one image at a time
            image_index = st.slider("Select image index", min_value=0, max_value=len(tensor_data) - 1, value=0)
            img = candle_tensor.plot_candle_from_tensor(tensor_data[image_index:image_index + 1])
            st.image(img, caption=f"Image {image_index}")

            # Save images
            for index in sampled_indices:
                img = candle_tensor.plot_candle_from_tensor(tensor_data[index:index + 1])
                img.save(os.path.join(base_path, f"{index}.png"))

            st.write("Processing complete. Images saved to:", base_path)

if __name__ == "__main__":
    main()
