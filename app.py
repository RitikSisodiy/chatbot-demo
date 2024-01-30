import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import pdb
from audio_recorder_streamlit import audio_recorder
from tempfile import NamedTemporaryFile
from gtts import gTTS
import tempfile
import time
import base64
import whisper
# Disable tokenizers parallelism


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute.")
        return result
    return wrapper
@timing_decorator
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]
@timing_decorator
def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

audio_container = st.empty()
def autoplay_audio(file_path: str):
    audio_container.empty()
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay style="display:none">
            <source src="data:audio/mp3;base64,{b64}"  type="audio/mp3">
            </audio>
            """
        
        # Display the new audio tag in the empty container
        audio_container.markdown(md, unsafe_allow_html=True)
        
@st.cache_resource
def load_audio_model():
    return whisper.load_model("base")

def transcribe_text_to_voice(audio):
    # audio= open(audio_location, "rb")
    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    model = load_audio_model()
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text
     

def text_to_speech(text,lang="en"):
    if lang == 'es':
        tts = gTTS(text, lang='es', tld="cl")
    else:
        tts = gTTS(text, lang=lang)
    return tts
    
def text_to_play_audio(text):
    with NamedTemporaryFile(suffix=".mp3") as temp:
        tts = text_to_speech(text)
        tempname = temp.name
        tts.save(tempname)
        autoplay_audio(tempname)
@timing_decorator
def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()
    generated= False
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')
        audio_bytes = audio_recorder(text="ask")
        if audio_bytes and not submit_button:
        ##Save the Recorded File
            audio_location = "audio_file.wav"
            with open(audio_location, "wb") as f:
                f.write(audio_bytes)
                user_input = transcribe_text_to_voice(audio_location)
        if (submit_button or audio_bytes) and user_input:
            with st.spinner('Generating text response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            generated = True
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
    if generated:
        with st.spinner('Generating audio response...'):
            text_to_play_audio(output)
@st.cache_resource
def load_model():
  llm = LlamaCpp(
      streaming = True,
      model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
      temperature=0.75,
      top_p=1,
      verbose=True,
      n_ctx=4096
  )
  return llm

@timing_decorator
def create_conversational_chain(vector_store):
    # Create llm
    
    llm = load_model()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    print("*"*20,"main is running","*"*20)
    # Initialize session state
    initialize_session_state()
    st.title("Multi-PDF ChatBot using Mistral-7B-Instruct :books:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)


    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        
        display_chat_history(chain)

if __name__ == "__main__":
    main()
