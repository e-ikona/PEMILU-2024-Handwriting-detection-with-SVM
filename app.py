import streamlit as st
import os
import cv2
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from skimage.feature import hog
import matplotlib.pyplot as plt

def load_data(folder_path, img_size):
    data, labels = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            label = filename.split('_')[0] 
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (img_size, img_size))
            data.append(img_resized.flatten())
            labels.append(label)
    return np.array(data), np.array(labels)

def extract_hog_features(images, img_size):
    hog_features = []
    for img in images:
        feature = hog(img.reshape(img_size, img_size),pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(feature)
    return np.array(hog_features)

def split_data(X, y, test_size):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in splitter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test

def visualize_results(y_train, y_test):
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    labels = sorted(set(unique_train) | set(unique_test))
    train_counts = [counts_train[unique_train.tolist().index(label)] if label in unique_train else 0 for label in labels]
    test_counts = [counts_test[unique_test.tolist().index(label)] if label in unique_test else 0 for label in labels]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    bar_train = plt.bar(x - 0.2, train_counts, width=0.4, label='Train', color='blue')
    bar_test = plt.bar(x + 0.2, test_counts, width=0.4, label='Test', color='orange')
    
    for bar in bar_train:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(int(bar.get_height())), 
                 ha='center', va='bottom', fontsize=10, color='blue')
    for bar in bar_test:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(int(bar.get_height())), 
                 ha='center', va='bottom', fontsize=10, color='orange')
    
    plt.xticks(x, labels)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Distribusi Data Latih dan Data Uji')
    plt.legend()
    st.pyplot(plt)


def preview_predictions(model, X_test, y_test, X_original, img_size, num_samples=20):
    num_columns = 5
    num_rows = (num_samples // num_columns) + (num_samples % num_columns > 0)
    plt.figure(figsize=(15, 3 * num_rows))
    
    for i in range(num_samples):
        index = np.random.randint(0, len(X_test))
        
        # Mengambil gambar asli dan label asli
        img = X_original[index].reshape(img_size, img_size)
        true_label = y_test[index]
        
        # Menghitung prediksi untuk gambar yang terpilih
        predicted_label = model.predict([X_test[index]])[0]
        
        # Menampilkan gambar dengan label asli dan prediksi
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Asli: {true_label}\nPred: {predicted_label}", color='green' if true_label == predicted_label else 'red')
        plt.axis('off')
    
    plt.tight_layout()
    st.pyplot(plt)




def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred) *100
    f1 = f1_score(y_test, y_pred, average='weighted') *100
    
    st.text(f"Akurasi: {acc:.2f}%")  
    st.text(f"F1-Score: {f1:.2f}%")  
    
    return model

def process_uploaded_image(uploaded_file, img_size=40):
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (img_size, img_size))
    return img_resized.flatten()

def predict_image(model, uploaded_image, img_size=40):
    feature = hog(uploaded_image.reshape(img_size, img_size), pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
    prediction = model.predict([feature])
    return prediction[0]

def prediction_page(model, img_size=40):    
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = process_uploaded_image(uploaded_file, img_size)
        img_resized = cv2.resize(img.reshape(img_size, img_size), (img_size, img_size))
        
        prediction = predict_image(model, img_resized, img_size)
        st.image(uploaded_file, caption="Gambar yang diunggah")
        st.title(f"Prediksi: {prediction}")

def main():
    st.sidebar.title("Haryo | 078")
    option = st.sidebar.radio("Pilih Halaman:", ["Model Tanpa Ekstraksi Fitur", "Model dengan Ekstraksi Fitur", "Prediksi"])

    folder_path = './DS-9/'

    if option == "Model Tanpa Ekstraksi Fitur":
        st.title("Model Tanpa Ekstraksi Fitur (SVM)")
        test_size_percent = st.slider(
            "Pilih proporsi data uji (%)", 
            min_value=10, max_value=50, step=5, value=20
        )
        test_size = test_size_percent / 100  

        X, y = load_data(folder_path, img_size=30)
        X = X / 255.0
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
        
        st.subheader("Visualisasi Data")
        visualize_results(y_train, y_test)
        
        st.subheader("Evaluasi Model")
        model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        
        st.subheader("Preview Prediksi (Random Pick)")
        preview_predictions(model, X_test, y_test, X, img_size=30)
    
    elif option == "Model dengan Ekstraksi Fitur":
        st.title("Model dengan Ekstraksi Fitur (SVM + HOG)")
        
        test_size_percent = st.slider(
            "Pilih proporsi data uji (%)", 
            min_value=10, max_value=50, step=5, value=20
        )
        test_size = test_size_percent / 100  

        X, y = load_data(folder_path, img_size=40)
        X = X / 255.0
        hog_X = extract_hog_features(X, img_size=40)
        X_train, X_test, y_train, y_test = split_data(hog_X, y, test_size=test_size)
        
        st.subheader("Visualisasi Data")
        visualize_results(y_train, y_test)
        
        st.subheader("Evaluasi Model")
        model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        
        st.subheader("Preview Prediksi (Random Pick)")
        preview_predictions(model, X_test, y_test, X, img_size=40)

    elif option == "Prediksi":
        st.title("Halaman Prediksi")
        st.write("Statistik model yang dipakai(SVM + HOG):")
        X, y = load_data(folder_path, img_size=40)
        X = X / 255.0
        hog_X = extract_hog_features(X, img_size=40)
        X_train, X_test, y_train, y_test = split_data(hog_X, y, test_size=0.2)
        model = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        prediction_page(model)

if __name__ == "__main__":
    main()
