import streamlit as st
import torch
import CNN
from torchvision import transforms
from PIL import Image
import openai
import torchvision.transforms.functional as TF
import numpy as np
import time

# OpenAI setup
openai.api_key = ''

# Define the model class 
class PlantDiseaseNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseNet, self).__init__()
        # Example of layers (you should define these based on your actual CNN architecture)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(in_features=16*224*224, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x

# Load the plant disease prediction model
model = CNN.CNN(39)  # Ensure CNN.CNN(39) is correctly implemented
model_path = "plant_disease_model_1_latest.pt"
model.eval()

# Define a mapping from class indices to disease names
class_to_disease = {
    # (Same mapping as before)
}

# Define the rate limit
rate_limit = 20  # Set a lower rate limit, e.g., 20 requests per minute

# Keep track of the last request time
last_request_time = None

def ask_openai_with_rate_limit(question, retry_count=0):
    global last_request_time

    max_retries = 5
    base_delay = 60  # Start with 60 seconds delay

    # Calculate the time elapsed since the last request
    if last_request_time is not None:
        time_elapsed = time.time() - last_request_time
        if time_elapsed < 60 / rate_limit:
            time.sleep((60 / rate_limit) - time_elapsed)

    try:
        # Make the API request
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=question,
            max_tokens=150
        )

        # Update the last request time
        last_request_time = time.time()

        return response.choices[0].text.strip()

    except openai.error.RateLimitError:
        if retry_count < max_retries:
            delay = base_delay * (2 ** retry_count)  # Exponential backoff
            st.warning(f"Rate limit exceeded. Retrying in {delay} seconds.")
            time.sleep(delay)
            return ask_openai_with_rate_limit(question, retry_count + 1)
        else:
            st.error("Rate limit exceeded. Please try again later.")
            return "Rate limit exceeded. Please try again later."

    except openai.error.OpenAIError as e:
        st.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

def predict_disease(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Resize the image
    image = image.resize((224, 224))
    
    # Convert the image to a tensor
    input_data = TF.to_tensor(image)
    
    # Adjust tensor dimensions
    input_data = input_data.view((-1, 3, 224, 224))
    
    # Make prediction
    with torch.no_grad():
        output = model(input_data)
    
    # Convert tensor to numpy array
    output = output.detach().numpy()
    
    # Get index of the max value
    index = np.argmax(output)
    
    return class_to_disease[index]

def main():
    st.title("FarmSmartAI")

    st.sidebar.header("Choose a Feature")
    choice = st.sidebar.radio("", ("Conversational Agent", "Plant Disease Prediction"))

    if choice == "Conversational Agent":
        st.header("AI Assistant")
        user_input = st.text_area("Ask a question on Agriculture in Africa:")
        if st.button("Ask"):
            response = ask_openai_with_rate_limit(user_input)
            st.write(response)

    elif choice == "Plant Disease Prediction":
        st.header("Plant Disease Classifier")
        uploaded_file = st.file_uploader("Upload an image of a plant", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            st.write("Classifying...")
            prediction = predict_disease(uploaded_file)
            st.write(f"Predicted Disease: {prediction}")

if __name__ == "__main__":
    main()
