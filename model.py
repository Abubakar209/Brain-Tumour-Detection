import os
import streamlit as st
os.system("apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0")
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import google.generativeai as genai
from pinecone import Pinecone
from google.generativeai import embedding


# Initialize Pinecone client
pi_key=st.secrets["pi_key"]  # Replace with your Pinecone API key
environment = "us-east-1"  # Set to your Pinecone environment region
pc = Pinecone(api_key=pi_key, environment=environment)

# Connect to the Pinecone index
index_name = "brain-tumor-2"  # Replace with your Pinecone index name
index = pc.Index(index_name)

# Initialize the Sentence Transformer model (BERT-based)
genai.configure(api_key=st.secrets["api_key"])

 # You can use other models if needed

# Function to generate embeddings from the text
import google.generativeai as genai

def generate_embeddings(text):
    try:
        if not text:
            st.warning("⚠️ Empty text received for embedding.")
            return None

        response = genai.embed_content(
            model="models/embedding-001",  # ✅ Correct model path
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        st.error(f"Embedding generation failed: {e}")
        return None




# Function to query Pinecone
def query_pinecone(query, top_k=3):
    query_embedding = generate_embeddings(query)

    if query_embedding is None:
        return []

    query_results = index.query(
        vector=query_embedding,  # No need for .tolist()
        top_k=top_k,
        include_metadata=True
    )
    results = [result['metadata']['text'] for result in query_results['matches']]
    return results


# Function to generate a response using the Gemini API
def generate_response(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Function to predict and display brain tumor detection results
def predict_and_display(image_path, model):
    results = model.predict(source=image_path, save=True, save_txt=False)
    tumor_count = 0
    tumor_details = []
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = box.cls[0]
            
            if label == 0:  # Assuming '0' is the label for tumors
                tumor_count += 1
                tumor_details.append({"confidence": confidence, "coordinates": (x1, y1, x2, y2)})
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(img, f"Tumor: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Display the image with bounding boxes
    st.image(img, caption="Tumor Detection Results", use_column_width=True)
    
    # Display the tumor count and details on the Streamlit interface
    st.write(f"**Number of tumors detected:** {tumor_count}")
    if tumor_details:
        st.write("**Tumor Details:**")
        for i, detail in enumerate(tumor_details):
            st.write(f"- Tumor {i + 1}: Confidence = {detail['confidence']:.2f}, Coordinates = {detail['coordinates']}")

    return tumor_count, tumor_details

# Function to generate a response for user queries
def generate_response_with_context(tumor_info, relevant_chunks, user_query, api_key):
    detailed_prompt = (
        f"You are a helpful, empathetic,something like doctor,and knowledgeable medical assistant. "
        f"Based on the analysis, the image shows {tumor_info}. "
        f"Additionally, here are some relevant insights related to the query: {relevant_chunks}. "
        f"Provide any general guidance or suggestions the user might need but avoid using disclaimers. "
        f"Focus on being supportive and resourceful. "
        f"Answer the following user query in a general and helpful manner with practical, empathetic, and actionable advice: {user_query}. "
        f"If the query is about medications for brain tumors or related symptoms like headaches, include general medication options like common pain relievers or drugs used for such conditions. "
        f"Otherwise, respond empathetically with relevant advice tailored to the query."
    )
    
    return generate_response(detailed_prompt, api_key)



# Streamlit interface
def main():
    st.title('Brain Tumor Detection Chatbot')
    
    uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded.getbuffer())
        st.image(uploaded, caption="Uploaded Image", use_column_width=True)
        
        model = YOLO('best.pt')  # Path to your YOLO model
        tumor_count, tumor_details = predict_and_display("uploaded_image.jpg", model)
        
        user_query = st.text_input("Ask a question about the tumor results or general advice:")
        
        if user_query:
            tumor_info = f"The uploaded image contains {tumor_count} brain tumors. "
            tumor_info += "Details of the detected tumors: "
            
            for i, detail in enumerate(tumor_details):
                tumor_info += f"Tumor {i + 1}: Confidence = {detail['confidence']:.2f}, Bounding Box = {detail['coordinates']}. "
            
            relevant_chunks = query_pinecone(user_query)
            api_key = st.secrets["api_key"] # Replace with your Gemini API key
            response = generate_response_with_context(tumor_info, relevant_chunks, user_query, api_key)
            
            if response:
                st.write(f"Chatbot Response: {response}")
            else:
                st.write("Failed to generate a response.")

if __name__ == "__main__":
    main()
