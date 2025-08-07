import streamlit as st
import os
from pathlib import Path

    # Page configuration
st.set_page_config(
    page_title="Text Analysis & Classification Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .app-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    .app-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .app-title {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .app-description {
        color: #6c757d;
        margin-bottom: 1rem;
    }
    .nav-button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Text Analysis & Classification Suite</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div style='text-align: center; margin-bottom: 3rem;'>
        <p style='font-size: 1.1rem; color: #6c757d;'>
            A comprehensive toolkit for text processing, classification, and analysis. Choose from the specialized applications below to process your text data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Application cards
    col1, col2 = st.columns(2)
    
    # Define your applications here
    apps = [
        {
            "title": "📝 Word Classifier",
            "description": "Classify individual words into predefined categories using machine learning algorithms. Perfect for content categorization and word-level analysis.",
            "file": "pages/1_Classifier_Word.py",
            "icon": "📝"
        },
        {
            "title": "📄 Text Classification",
            "description": "Analyze and classify entire text documents or passages. Automatically categorize content based on topic, sentiment, or custom classification schemes.",
            "file": "pages/1_Text_Classification.py",
            "icon": "📄"
        },
        {
            "title": "📚 Dictionary Refinement",
            "description": "Refine and optimize your classification dictionaries. Add, remove, or modify terms to improve classification accuracy and coverage.",
            "file": "pages/3_Dictionary_Refinement.py",
            "icon": "📚"
        },
        {
            "title": "🏗️ Dictionary Classifier Creation",
            "description": "Build custom dictionary-based classifiers from scratch. Create rule-based classification systems using keyword dictionaries and pattern matching.",
            "file": "pages/4_Dictionary_Classifier_Creation.py",
            "icon": "🏗️"
        },
        {
            "title": "🔗 Join Table App",
            "description": "Merge and join multiple data tables based on common fields. Perform inner, outer, left, and right joins with an intuitive interface.",
            "file": "pages/5_Join_Table_App.py",
            "icon": "🔗"
        },
        {
            "title": "✂️ Text Sentence Tokenizer",
            "description": "Break down text into individual sentences with advanced tokenization. Handle complex punctuation, abbreviations, and multi-language text processing.",
            "file": "pages/6_Text_Sentence_Tokenizer.py",
            "icon": "✂️"
        },
        {
            "title": "🎯 Rolling Context Window Processor",
            "description": "Process text using sliding window techniques. Analyze text patterns and context within specified window sizes for advanced text analysis.",
            "file": "pages/7_Rolling_Context_Window_Processor.py",
            "icon": "🎯"
        }
    ]
    
    # Display apps in cards
    for i, app in enumerate(apps):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            with st.container():
                st.markdown(f"""
                <div class="app-card">
                    <div class="app-title">{app['title']}</div>
                    <div class="app-description">{app['description']}</div>
                    <p><strong>File:</strong> {app['file']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Check if file exists
                if os.path.exists(app['file']):
                    if st.button(f"Open {app['title']}", key=f"btn_{i}"):
                        st.switch_page(app['file'])
                else:
                    st.warning(f"⚠️ {app['file']} not found")
    
    # Sidebar with quick navigation
    with st.sidebar:
        st.markdown("### 🚀 Quick Navigation")
        st.markdown("---")
        
        # Group apps by category
        classification_apps = apps[:2]
        dictionary_apps = apps[2:4] 
        processing_apps = apps[4:]
        
        st.markdown("**📝 Classification Tools**")
        for app in classification_apps:
            if os.path.exists(app['file']):
                if st.button(f"{app['icon']} {app['title'].split(' ', 1)[1]}", key=f"sidebar_{app['file']}", use_container_width=True):
                    st.switch_page(app['file'])
        
        st.markdown("**📚 Dictionary Tools**")        
        for app in dictionary_apps:
            if os.path.exists(app['file']):
                if st.button(f"{app['icon']} {app['title'].split(' ', 1)[1]}", key=f"sidebar_{app['file']}", use_container_width=True):
                    st.switch_page(app['file'])
                    
        st.markdown("**🔧 Processing Tools**")
        for app in processing_apps:
            if os.path.exists(app['file']):
                if st.button(f"{app['icon']} {app['title'].split(' ', 1)[1]}", key=f"sidebar_{app['file']}", use_container_width=True):
                    st.switch_page(app['file'])
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.info("This is your central hub for text analysis and classification tools. Each app specializes in different aspects of natural language processing.")
        
        st.markdown("### 🆘 Help")
        with st.expander("How to use"):
            st.write("""
            - **Word/Text Classification**: Use these for categorizing content
            - **Dictionary Tools**: Build and refine classification dictionaries  
            - **Text Processing**: Tokenize and analyze text structure
            - **Data Tools**: Join and merge your datasets
            - Click on any application card to get started
            """)

# Alternative approach using multipage structure
def create_multipage_nav():
    """
    Alternative navigation using st.navigation for newer Streamlit versions
    """
    # Define pages
    pages = {
        "Home": [
            st.Page("home.py", title="🏠 Home", icon="🏠"),
        ],
        "Analytics": [
            st.Page("dashboard.py", title="📊 Dashboard", icon="📊"),
            st.Page("explorer.py", title="🔍 Data Explorer", icon="🔍"),
            st.Page("sales_analytics.py", title="📈 Sales Analytics", icon="📈"),
        ],
        "Tools": [
            st.Page("ml_predictor.py", title="🤖 ML Predictor", icon="🤖"),
            st.Page("report_generator.py", title="📋 Report Generator", icon="📋"),
            st.Page("settings.py", title="⚙️ Settings", icon="⚙️"),
        ]
    }
    
    # Create navigation
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    # Use the main function for the card-based approach
    main()
    
    # Uncomment the line below and comment out main() above 
    # if you prefer the newer navigation approach
    # create_multipage_nav()
