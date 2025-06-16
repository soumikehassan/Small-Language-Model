# ===================
# 1. STREAMLIT APP (EASIEST)
# ===================

# File: streamlit_app.py
import streamlit as st
import torch
import pickle
import sys
import os

# Add your model files to path if needed
# sys.path.append('./model_files')

# Import your model classes (from your training script)
from SLM import SmallLanguageModel, TextGenerator, TextPreprocessor

class StreamlitSLMApp:
    def __init__(self):
        self.model = None
        self.generator = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    @st.cache_resource
    def load_model(_self):
        """Load the trained model (cached for performance)"""
        try:
            # Load preprocessor
            with open('D:/SLM/preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            
            # Load model
            checkpoint = torch.load('D:/SLM/final_slm_model.pt', map_location=_self.device)
            config = checkpoint['config']
            
            model = SmallLanguageModel(
                vocab_size=len(checkpoint['vocab']),
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                n_layers=config['n_layers'],
                d_ff=config['d_ff']
            ).to(_self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            generator = TextGenerator(model, preprocessor, _self.device)
            
            return generator, preprocessor
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None
    
    def run(self):
        st.set_page_config(
            page_title="Academic SLM Generator",
            page_icon="🎓",
            layout="wide"
        )
        
        st.title("🎓 Academic Small Language Model")
        st.markdown("Generate academic content based on your lecture materials")
        
        # Load model
        generator, preprocessor = self.load_model()
        
        if generator is None:
            st.error("Failed to load model. Please check your model files.")
            return
        
        # Sidebar for settings
        st.sidebar.header("Generation Settings")
        max_length = st.sidebar.slider("Max Length", 10, 200, 100)
        temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        top_k = st.sidebar.slider("Top-k", 1, 100, 50)
        
        # Main interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input")
            prompt = st.text_area(
                "Enter your prompt:",
                height=100,
                placeholder="Multivariate Analysis"
            )
            
            generate_btn = st.button("Generate Text", type="primary")
            
            # Predefined prompts
            st.subheader("Quick Prompts")
            quick_prompts = [
                "Random Intercept Model",
                "Random Slope Model",
                "Fixed Effects",
                "Random Intercept and Slope Model ",
                "Random Effects"
            ]
            
            for qp in quick_prompts:
                if st.button(qp, key=f"quick_{qp}"):
                    prompt = qp
                    generate_btn = True
        
        with col2:
            st.subheader("Generated Text")
            
            if generate_btn and prompt:
                with st.spinner("Generating..."):
                    try:
                        generated_text = generator.generate_text(
                            prompt, 
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k
                        )
                        
                        st.success("Generation Complete!")
                        st.write("**Generated Text:**")
                        st.write(generated_text)
                        
                        # Copy to clipboard button
                        st.code(generated_text, language=None)
                        
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
            
            elif generate_btn:
                st.warning("Please enter a prompt first!")

# Run Streamlit app
if __name__ == "__main__":
    app = StreamlitSLMApp()
    app.run()
