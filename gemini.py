import base64
import io
import os
import time
from typing import Dict, List, Optional, Union
import tempfile

import streamlit as st
import google.generativeai as genai
from PIL import Image


st.set_page_config(
        page_title="Gemini 2.0 Flash Chatbot",
        page_icon="🤖",
        layout="wide"
    )

# Retrieve API key for Google GenAI from the environment variables
# or prompt the user to enter it if not available
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

if not st.session_state.GOOGLE_API_KEY:
    st.session_state.GOOGLE_API_KEY = st.text_input("Enter your Google API Key:", type="password")
    if not st.session_state.GOOGLE_API_KEY:
        st.warning("Please enter your Google API Key to continue.")
        st.stop()

# Initialize the client so that it can be reused across functions
genai.configure(api_key=st.session_state.GOOGLE_API_KEY)

# Constants
IMAGE_WIDTH = 512
VERSION = "1.0"

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Resize an image to a fixed width while maintaining the aspect ratio.

    Parameters:
        image (Image.Image): The original image.

    Returns:
        Image.Image: The resized image with width fixed at IMAGE_WIDTH.
    """
    image_height = int(image.height * IMAGE_WIDTH / image.width)
    return image.resize((IMAGE_WIDTH, image_height))

def get_image_html(image: Image.Image, max_width: int = 250) -> str:
    """
    Convert a PIL Image to an HTML img tag with a data URI.
    
    Parameters:
        image (Image.Image): The image to convert
        max_width (int): Maximum width for display
        
    Returns:
        str: HTML string with the embedded image
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{img_str}" width="{max_width}">'

def get_audio_html(audio_bytes: bytes, max_width: int = 250) -> str:
    """
    Create HTML audio tag for audio data.
    
    Parameters:
        audio_bytes (bytes): The audio data
        max_width (int): Maximum width for the audio player
        
    Returns:
        str: HTML string with the embedded audio player
    """
    b64_audio = base64.b64encode(audio_bytes).decode()
    return f'<audio controls style="max-width:{max_width}px;"><source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">Your browser does not support the audio element.</audio>'

def format_chat_message(message):
    """Format messages for display in the chat UI"""
    if message["role"] == "user":
        return f"👤 **You**: {message['content']}"
    else:
        return f"🤖 **Gemini**: {message['content']}"

def process_uploaded_files():
    """Process all uploaded files and add them to the chat history"""
    
    # Handle PDF uploads (documents)
    pdf_files = st.session_state.get("pdf_files", [])
    if pdf_files:
        for pdf_file in pdf_files:
            file_info = {
                "name": pdf_file.name,
                "type": "pdf",
                "data": pdf_file.getvalue()
            }
            
            st.session_state.uploaded_files.append(file_info)
            st.session_state.last_uploaded = file_info
            
            st.session_state.chat_history.append({
                "role": "user",
                "content": f"📄 Document uploaded: {pdf_file.name}",
                "file_ref": file_info
            })
    
    # Handle image uploads
    image_files = st.session_state.get("image_files", [])
    if image_files:
        for img_file in image_files:
            image = Image.open(img_file).convert("RGB")
            image = preprocess_image(image)
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            
            file_info = {
                "name": img_file.name,
                "type": "image",
                "data": img_byte_arr.getvalue()
            }
            
            st.session_state.uploaded_files.append(file_info)
            st.session_state.last_uploaded = file_info
            
            img_html = get_image_html(image)
            st.session_state.chat_history.append({
                "role": "user",
                "content": f"🖼️ Image uploaded: {img_file.name}<br>{img_html}",
                "file_ref": file_info
            })
    
    # Handle audio uploads
    audio_files = st.session_state.get("audio_files", [])
    if audio_files:
        for audio_file in audio_files:
            audio_bytes = audio_file.getvalue()
            
            file_info = {
                "name": audio_file.name,
                "type": "audio",
                "data": audio_bytes
            }
            
            st.session_state.uploaded_files.append(file_info)
            st.session_state.last_uploaded = file_info
            
            audio_html = get_audio_html(audio_bytes)
            st.session_state.chat_history.append({
                "role": "user",
                "content": f"🔊 Audio uploaded: {audio_file.name}<br>{audio_html}",
                "file_ref": file_info
            })
        
        # Clear the uploader state
        st.session_state.audio_files = []

def generate_gemini_response(query_text=None):
    """
    Generate a response from Gemini based on the chat history and the last uploaded file
    """
    if not query_text and not st.session_state.last_uploaded:
        st.warning("Please enter a message or upload a file to continue.")
        return
    
    # Add user query to chat history if provided
    if query_text:
        st.session_state.chat_history.append({
            "role": "user",
            "content": query_text
        })
        
    # Temporarily add a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
    
    # Configure the generation parameters
    generation_config = genai.GenerationConfig(
        temperature=0.4,
        max_output_tokens=4096,
        top_k=32,
        top_p=1,
    )
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    # Get last file if any
    last_file = st.session_state.last_uploaded
    
    # Prepare the contents based on what we have
    contents = []
    
    # If we have a query text from the current interaction
    if query_text:
        contents.append(query_text)
    # Otherwise get the last message if it's from the user
    elif st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        contents.append(st.session_state.chat_history[-1]["content"])
    
    # Add file content if we have a last uploaded file
    if last_file:
        if last_file["type"] == "image":
            # For images, use PIL to open the image data
            image = Image.open(io.BytesIO(last_file["data"])).convert("RGB")
            contents.append(image)
        elif last_file["type"] == "pdf" or last_file["type"] == "audio":
            # For PDFs and audio, add as binary data with mime type
            mime_type = "application/pdf" if last_file["type"] == "pdf" else "audio/mp3"
            # Check how your version of the API handles binary data
            try:
                # Try the direct approach
                contents.append({
                    "mime_type": mime_type,
                    "data": last_file["data"]
                })
            except Exception as e:
                # If direct approach fails, try another method specific to your API version
                st.error(f"Error processing file: {str(e)}")
                # Add a failure message to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Sorry, I couldn't process the {last_file['type']} file due to an error: {str(e)}"
                })
                # Update the UI
                message_placeholder.markdown(st.session_state.chat_history[-1]["content"])
                return
    
    # If we don't have any contents, construct from chat history
    if not contents:
        # Format recent chat history as a conversational prompt
        prompt = ""
        # Get the last few messages (limit to avoid token issues)
        recent_messages = st.session_state.chat_history[-10:] if len(st.session_state.chat_history) > 10 else st.session_state.chat_history
        for msg in recent_messages:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            # Strip HTML tags for better prompting
            content = msg["content"]
            # Simple HTML tag removal (not perfect but helps)
            content = content.replace("<br>", "\n")
            for tag in ["<img", "<audio", "<p", "</p", "<source"]:
                if tag in content:
                    content = content.split(tag)[0]
            prompt += f"{prefix}{content}\n"
        
        contents = [prompt]
    
    # Generate the response
    try:
        response = model.generate_content(
            contents=contents,
            generation_config=generation_config,
            stream=True
        )
        
        # Process the streaming response
        full_response = ""
        for chunk in response:
            if hasattr(chunk, 'text'):
                full_response += chunk.text
                # Update the message placeholder with the current text
                message_placeholder.markdown(full_response)
                time.sleep(0.01)
        
        # Add the complete response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response
        })
        
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": error_message
        })
        message_placeholder.markdown(error_message)

def clear_chat():
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.session_state.uploaded_files = []
    st.session_state.last_uploaded = None

def run_code_execution(code_prompt):
    """Execute code using Gemini"""
    if not code_prompt:
        return
    
    # Add user query to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"```\n{code_prompt}\n```"
    })
    
    # Temporarily add a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Executing code...")
    
    # Configure for code execution
    generation_config = {
        "tools": [{
            "code_execution": {}  # Tool specification for code execution
        }]
    }
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    try:
        response = model.generate_content(
            contents=code_prompt,
            generation_config=generation_config
        )
        
        output_text = ""
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text is not None:
                    output_text += f"{part.text}\n"
                
                # Handle different attribute names based on API version
                if hasattr(part, 'function_call') and part.function_call is not None:
                    if hasattr(part.function_call, 'name') and part.function_call.name == "execute_code":
                        code = part.function_call.args.get("code", "")
                        output_text += f"\n**Generated Code:**\n```python\n{code}\n```\n"
                
                if hasattr(part, 'executable_code') and part.executable_code is not None:
                    output_text += f"\n**Generated Code:**\n```python\n{part.executable_code.code}\n```\n"
                
                if hasattr(part, 'code_execution_result') and part.code_execution_result is not None:
                    output_text += f"\n**Output:**\n```\n{part.code_execution_result.output}\n```\n"
                
                if hasattr(part, 'inline_data') and part.inline_data is not None:
                    try:
                        image_data = base64.b64decode(part.inline_data.data)
                        image = Image.open(io.BytesIO(image_data))
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        b64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        img_html = f'<img src="data:image/png;base64,{b64_data}" alt="Generated Image" style="max-width:300px;"/>'
                        output_text += f"\n{img_html}\n"
                    except Exception as e:
                        output_text += f"\nError displaying image: {str(e)}\n"
        else:
            output_text = "No response or empty response received from the model."
        
        # Add to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": output_text
        })
        
        # Update the display
        message_placeholder.markdown(output_text, unsafe_allow_html=True)
        
    except Exception as e:
        error_message = f"Error executing code: {str(e)}"
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": error_message
        })
        message_placeholder.markdown(error_message)

def main():
    
    st.title("Gemini 2.0 Flash Multi-modal Chatbot")
    #st.caption(f"Version {VERSION} - Powered by Google's Gemini 2.0 Flash model")
    
    # Sidebar with controls
    with st.sidebar:
        st.header("Options")
        
        # Upload buttons for different media types
        st.subheader("Upload Files")
        
        # Document upload
        pdf_files = st.file_uploader(
            "Upload PDF documents", 
            type=["pdf"], 
            accept_multiple_files=True,
            key="pdf_files"
        )
        
        # Image upload
        image_files = st.file_uploader(
            "Upload images", 
            type=["jpg", "jpeg", "png",'webp'], 
            accept_multiple_files=True,
            key="image_files"
        )
        
        # Audio upload
        audio_files = st.file_uploader(
            "Upload audio", 
            type=["mp3", "wav"], 
            accept_multiple_files=True,
            key="audio_files" 
        )
        
        # Process file uploads when submitted
        if st.button("Submit Uploads", key="submit_uploads"):
            process_uploaded_files()
            st.success("Files uploaded successfully!")
            st.experimental_rerun()  # Rerun to update the UI
        
        # Add a clear button
        if st.button("Clear Chat", key="clear_chat"):
            clear_chat()
            st.success("Chat cleared!")
            st.rerun()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # Chat input
    tab1, tab2 = st.tabs(["Chat",'Code'])
    
    with tab1:
        # Regular chat interface
        chat_input = st.chat_input("Type your message here...")
        if chat_input:
            generate_gemini_response(chat_input)
    
    with tab2:
        # Code execution interface
        code_col1, code_col2 = st.columns([5, 1])
        with code_col1:
            code_input = st.text_area("Enter code to execute:", height=150)
        with code_col2:
            st.write("")
            st.write("")
            if st.button("Execute", key="execute_code"):
                run_code_execution(code_input)
    
    # Display currently selected file info
    if st.session_state.last_uploaded:
        st.sidebar.subheader("Current Active File")
        st.sidebar.info(f"Name: {st.session_state.last_uploaded['name']}\nType: {st.session_state.last_uploaded['type'].upper()}")

        # Option to use a specific file from history
        if len(st.session_state.uploaded_files) > 1:
            st.sidebar.subheader("Select File to Use")
            file_names = [f"{i+1}. {f['name']} ({f['type']})" for i, f in enumerate(st.session_state.uploaded_files)]
            selected_file = st.sidebar.selectbox("Choose file:", file_names)
            
            if st.sidebar.button("Use Selected File"):
                # Extract the index from the selected option
                index = int(selected_file.split('.')[0]) - 1
                st.session_state.last_uploaded = st.session_state.uploaded_files[index]
                st.success(f"Now using: {st.session_state.last_uploaded['name']}")
                st.rerun()

if __name__ == "__main__":
    main()
