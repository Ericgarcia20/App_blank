import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List
import io

# Import the classifier class (assuming it's in the same file or imported)
class DictionaryClassifier:
    """Dictionary-based text classifier for binary classification tasks."""
    
    def __init__(self):
        self.data = None
        self.text_column = None
        self.label_column = None
        self.dictionary = []
        self.results = []
        self.keyword_analysis = {}
        self.tactic = ""
        
    def load_data(self, data: pd.DataFrame, text_column: str, label_column: str, tactic: str = ""):
        """Load dataset for classification."""
        self.data = data.copy()
        self.text_column = text_column
        self.label_column = label_column
        self.tactic = tactic
        
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
            
        self.data[label_column] = self.data[label_column].astype(int)
        
    def set_dictionary(self, keywords: List[str]):
        """Set the keyword dictionary for classification."""
        self.dictionary = [keyword.lower().strip() for keyword in keywords if keyword.strip()]
        
    def generate_default_dictionary(self, tactic_type: str = "promotional") -> List[str]:
        """Generate a default dictionary based on tactic type."""
        default_dictionaries = {
            "promotional": [
                "discount", "sale", "offer", "special", "limited", "free", 
                "save", "deal", "promotion", "exclusive", "off", "reduced"
            ],
            "urgent": [
                "urgent", "hurry", "deadline", "limited", "expires", "last", 
                "final", "ending", "quick", "fast", "now", "today"
            ],
            "social_proof": [
                "review", "testimonial", "customer", "rating", "recommend", 
                "satisfied", "happy", "feedback", "trusted", "proven"
            ],
            "gratitude": [
                "thank", "thanks", "grateful", "appreciate", "blessed", 
                "honor", "privilege", "gratitude", "thankful"
            ],
            "local_business": [
                "local", "community", "neighborhood", "hometown", "area", 
                "nearby", "regional", "city", "town", "location"
            ]
        }
        return default_dictionaries.get(tactic_type.lower(), default_dictionaries["promotional"])
    
    def classify_texts(self) -> Dict:
        """Classify texts using the current dictionary."""
        if self.data is None or not self.dictionary:
            raise ValueError("Data and dictionary must be set before classification")
            
        results = []
        
        for idx, row in self.data.iterrows():
            text = str(row[self.text_column]).lower()
            ground_truth = int(row[self.label_column])
            
            matched_keywords = [kw for kw in self.dictionary if kw in text]
            prediction = 1 if matched_keywords else 0
            
            if prediction == 1 and ground_truth == 1:
                classification_type = "TP"
            elif prediction == 1 and ground_truth == 0:
                classification_type = "FP"
            elif prediction == 0 and ground_truth == 1:
                classification_type = "FN"
            else:
                classification_type = "TN"
                
            results.append({
                'id': idx,
                'text': row[self.text_column],
                'prediction': prediction,
                'ground_truth': ground_truth,
                'matched_keywords': matched_keywords,
                'classification_type': classification_type,
                'is_correct': prediction == ground_truth
            })
            
        self.results = results
        metrics = self._calculate_overall_metrics()
        self.keyword_analysis = self._calculate_keyword_analysis()
        
        return {
            'results': results,
            'metrics': metrics,
            'keyword_analysis': self.keyword_analysis
        }
    
    def _calculate_overall_metrics(self) -> Dict:
        """Calculate overall classification metrics."""
        tp = sum(1 for r in self.results if r['classification_type'] == 'TP')
        fp = sum(1 for r in self.results if r['classification_type'] == 'FP')
        fn = sum(1 for r in self.results if r['classification_type'] == 'FN')
        tn = sum(1 for r in self.results if r['classification_type'] == 'TN')
        
        total = len(self.results)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'total_samples': total
        }
    
    def _calculate_keyword_analysis(self) -> Dict:
        """Calculate per-keyword performance metrics."""
        keyword_stats = {}
        
        for keyword in self.dictionary:
            keyword_results = [r for r in self.results if keyword in r['matched_keywords']]
            
            tp = sum(1 for r in keyword_results if r['classification_type'] == 'TP')
            fp = sum(1 for r in keyword_results if r['classification_type'] == 'FP')
            
            fn_candidates = [r for r in self.results if r['classification_type'] == 'FN']
            fn = sum(1 for r in fn_candidates if keyword in str(r['text']).lower())
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            tp_examples = [r for r in keyword_results if r['classification_type'] == 'TP'][:3]
            fp_examples = [r for r in keyword_results if r['classification_type'] == 'FP'][:3]
            fn_examples = [r for r in fn_candidates if keyword in str(r['text']).lower()][:3]
            
            keyword_stats[keyword] = {
                'keyword': keyword,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'examples': {
                    'true_positives': tp_examples,
                    'false_positives': fp_examples,
                    'false_negatives': fn_examples
                }
            }
            
        return keyword_stats
    
    def get_true_positives(self) -> List[Dict]:
        """Get all true positive results."""
        return [r for r in self.results if r['classification_type'] == 'TP']
    
    def get_top_keywords_by_metric(self, metric: str = 'f1_score', top_n: int = 10) -> List[Dict]:
        """Get top keywords sorted by specified metric."""
        if not self.keyword_analysis:
            return []
            
        sorted_keywords = sorted(
            self.keyword_analysis.values(), 
            key=lambda x: x[metric], 
            reverse=True
        )
        
        return sorted_keywords[:top_n]


# Streamlit App Configuration
st.set_page_config(
    page_title="📊 Instagram Category Classifier - Dictionary Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .keyword-tag {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        margin: 0.125rem;
        display: inline-block;
        font-size: 0.875rem;
    }
    .step-indicator {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = DictionaryClassifier()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'classification_done' not in st.session_state:
    st.session_state.classification_done = False

# Create sample data function
def create_sample_data():
    """Create sample promotional content data for testing."""
    sample_data = {
        'id': range(1, 21),
        'statement': [
            "Get 50% off all items this weekend only!",
            "Thank you for your feedback on our service",
            "Limited time offer - buy one get one free!",
            "We appreciate your continued support",
            "Special discount for VIP members today",
            "Our team is here to help with any questions",
            "Flash sale ends at midnight - hurry!",
            "Welcome to our community newsletter",
            "Exclusive deal: Save 30% on premium products",
            "Your satisfaction is our top priority",
            "Last chance to get free shipping",
            "We value your business and loyalty",
            "Today only: Extra 20% off clearance items",
            "Thank you for choosing our brand",
            "Don't miss out on these incredible savings",
            "We're committed to excellent customer service",
            "Limited quantities available - act fast!",
            "Your feedback helps us improve",
            "Special promotion ends soon",
            "We're here whenever you need us"
        ],
        'answer': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    return pd.DataFrame(sample_data)

# Helper functions for visualization
def create_confusion_matrix_plot(metrics):
    """Create confusion matrix visualization."""
    matrix = [[metrics['true_positives'], metrics['false_negatives']],
              [metrics['false_positives'], metrics['true_negatives']]]
    
    fig = px.imshow(matrix, 
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"))
    
    fig.update_xaxis(tickvals=[0, 1], ticktext=['Negative', 'Positive'])
    fig.update_yaxis(tickvals=[0, 1], ticktext=['Positive', 'Negative'])
    fig.update_layout(title="Confusion Matrix", height=400)
    
    return fig

def create_keyword_performance_plot(keyword_analysis, metric='f1_score', top_n=10):
    """Create keyword performance bar chart."""
    if not keyword_analysis:
        return None
        
    sorted_keywords = sorted(keyword_analysis.values(), 
                           key=lambda x: x[metric], reverse=True)[:top_n]
    
    keywords = [kw['keyword'] for kw in sorted_keywords]
    values = [kw[metric] for kw in sorted_keywords]
    
    fig = px.bar(x=values, y=keywords, orientation='h',
                 title=f'Top {top_n} Keywords by {metric.replace("_", " ").title()}',
                 labels={'x': metric.replace("_", " ").title(), 'y': 'Keywords'})
    
    fig.update_layout(height=400, showlegend=False)
    return fig

# Main App
def main():
    # Header
    st.markdown("<h1 class='main-header'>📊 Instagram Category Classifier</h1>", unsafe_allow_html=True)
    st.markdown("### Dictionary Classification Bot - AI-Powered Text Classification")
    
    # Sidebar
    st.sidebar.title("🔧 Navigation")
    
    # Navigation options
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🎯 Setup & Data", "📖 Dictionary Management", "🔍 Classification", "📊 Analysis & Results"]
    )
    
    # Show current status in sidebar
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data Loaded")
        if hasattr(st.session_state.classifier, 'data') and st.session_state.classifier.data is not None:
            st.sidebar.write(f"📄 Rows: {len(st.session_state.classifier.data)}")
    
    if st.session_state.classifier.dictionary:
        st.sidebar.success("✅ Dictionary Set")
        st.sidebar.write(f"📚 Keywords: {len(st.session_state.classifier.dictionary)}")
    
    if st.session_state.classification_done:
        st.sidebar.success("✅ Classification Complete")
    
    # Page content
    if page == "🎯 Setup & Data":
        setup_data_page()
    elif page == "📖 Dictionary Management":
        dictionary_management_page()
    elif page == "🔍 Classification":
        classification_page()
    elif page == "📊 Analysis & Results":
        analysis_results_page()

def setup_data_page():
    """Setup and data upload page."""
    st.markdown("## 🎯 Step 1: Define Tactic & Load Data")
    
    # Tactic definition
    st.markdown("### Define Classification Tactic")
    tactic = st.text_input(
        "What type of content are you trying to classify?",
        value=st.session_state.classifier.tactic,
        placeholder="e.g., promotional offers, urgent messaging, customer complaints",
        help="This will guide the dictionary generation and analysis"
    )
    
    if tactic:
        st.session_state.classifier.tactic = tactic
    
    st.markdown("### Load Your Dataset")
    
    # Data upload options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upload CSV File**")
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file with text data and binary labels (0/1)"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded: {len(data)} rows, {len(data.columns)} columns")
                
                # Show preview
                st.markdown("**Data Preview:**")
                st.dataframe(data.head(), use_container_width=True)
                
                # Column selection
                st.markdown("**Select Columns:**")
                text_columns = data.columns.tolist()
                label_columns = data.columns.tolist()
                
                col_a, col_b = st.columns(2)
                with col_a:
                    text_column = st.selectbox(
                        "Text Column",
                        options=text_columns,
                        help="Column containing the text to classify"
                    )
                
                with col_b:
                    label_column = st.selectbox(
                        "Label Column (0/1)",
                        options=label_columns,
                        help="Column containing ground truth labels (0=negative, 1=positive)"
                    )
                
                # Validate and load data
                if st.button("📊 Load Data", type="primary"):
                    try:
                        # Validate label column
                        unique_labels = data[label_column].unique()
                        if not all(label in [0, 1, '0', '1'] for label in unique_labels if pd.notna(label)):
                            st.error("Label column must contain only 0 and 1 values")
                        else:
                            st.session_state.classifier.load_data(
                                data=data,
                                text_column=text_column,
                                label_column=label_column,
                                tactic=tactic
                            )
                            st.session_state.data_loaded = True
                            st.success("🎉 Data loaded successfully!")
                            
                            # Show data statistics
                            positive_count = sum(data[label_column].astype(int))
                            negative_count = len(data) - positive_count
                            
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("Total Samples", len(data))
                            with col_stat2:
                                st.metric("Positive Cases", positive_count)
                            with col_stat3:
                                st.metric("Negative Cases", negative_count)
                            
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    with col2:
        st.markdown("**Use Sample Data**")
        st.info("Try the app with promotional content examples")
        
        if st.button("📝 Load Sample Data"):
            sample_data = create_sample_data()
            st.session_state.classifier.load_data(
                data=sample_data,
                text_column='statement',
                label_column='answer',
                tactic="promotional offers and special deals"
            )
            st.session_state.data_loaded = True
            st.success("✅ Sample data loaded!")
            
            # Show sample data preview
            st.markdown("**Sample Data Preview:**")
            st.dataframe(sample_data.head(), use_container_width=True)
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Samples", len(sample_data))
            with col_stat2:
                st.metric("Positive Cases", sum(sample_data['answer']))
            with col_stat3:
                st.metric("Negative Cases", len(sample_data) - sum(sample_data['answer']))

def dictionary_management_page():
    """Dictionary management page."""
    st.markdown("## 📖 Step 2: Dictionary Management")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Setup & Data section.")
        return
    
    # Dictionary options
    st.markdown("### Choose Dictionary Source")
    
    dict_option = st.radio(
        "How would you like to create your dictionary?",
        ["Use Default Template", "Upload Custom Dictionary", "Manual Entry"]
    )
    
    if dict_option == "Use Default Template":
        st.markdown("#### Select Template Type")
        
        template_type = st.selectbox(
            "Choose a template based on your classification tactic:",
            ["promotional", "urgent", "social_proof", "gratitude", "local_business"],
            help="Templates contain pre-defined keywords for common classification tasks"
        )
        
        # Show template keywords
        template_keywords = st.session_state.classifier.generate_default_dictionary(template_type)
        
        st.markdown("**Template Keywords:**")
        keywords_display = " • ".join([f'`{kw}`' for kw in template_keywords])
        st.markdown(keywords_display)
        
        # Option to modify template
        st.markdown("#### Customize Template")
        custom_keywords = st.text_area(
            "Add or remove keywords (comma-separated):",
            value=", ".join(template_keywords),
            height=100,
            help="Edit the keywords as needed for your specific use case"
        )
        
        if st.button("📚 Set Dictionary"):
            keywords = [kw.strip() for kw in custom_keywords.split(',') if kw.strip()]
            st.session_state.classifier.set_dictionary(keywords)
            st.success(f"✅ Dictionary set with {len(keywords)} keywords!")
    
    elif dict_option == "Upload Custom Dictionary":
        st.markdown("#### Upload Dictionary File")
        
        # File format options
        format_option = st.radio(
            "File format:",
            ["Text file (comma-separated)", "JSON file"]
        )
        
        uploaded_dict = st.file_uploader(
            "Choose dictionary file",
            type=['txt', 'json'] if format_option == "JSON file" else ['txt', 'csv']
        )
        
        if uploaded_dict is not None:
            try:
                if format_option == "JSON file":
                    dict_data = json.load(uploaded_dict)
                    if isinstance(dict_data, list):
                        keywords = dict_data
                    elif isinstance(dict_data, dict):
                        keywords = list(dict_data.keys()) if dict_data else []
                    else:
                        st.error("JSON must contain a list of keywords or a dictionary")
                        return
                else:
                    content = uploaded_dict.read().decode('utf-8')
                    keywords = [kw.strip() for kw in content.replace('\n', ',').split(',') if kw.strip()]
                
                st.success(f"📁 Loaded {len(keywords)} keywords from file")
                
                # Show keywords
                st.markdown("**Loaded Keywords:**")
                keywords_display = " • ".join([f'`{kw}`' for kw in keywords[:20]])
                if len(keywords) > 20:
                    keywords_display += f" ... and {len(keywords) - 20} more"
                st.markdown(keywords_display)
                
                if st.button("📚 Use These Keywords"):
                    st.session_state.classifier.set_dictionary(keywords)
                    st.success(f"✅ Dictionary set with {len(keywords)} keywords!")
                    
            except Exception as e:
                st.error(f"Error reading dictionary file: {str(e)}")
    
    elif dict_option == "Manual Entry":
        st.markdown("#### Enter Keywords Manually")
        
        manual_keywords = st.text_area(
            "Enter keywords (comma-separated):",
            placeholder="discount, sale, offer, special, limited, free",
            height=150,
            help="Enter keywords separated by commas. Each keyword will be used for text matching."
        )
        
        if manual_keywords:
            keywords = [kw.strip() for kw in manual_keywords.split(',') if kw.strip()]
            
            st.markdown(f"**Preview ({len(keywords)} keywords):**")
            for i, kw in enumerate(keywords):
                if i < 20:  # Show first 20
                    st.markdown(f'<span class="keyword-tag">{kw}</span>', unsafe_allow_html=True)
                elif i == 20:
                    st.markdown(f"... and {len(keywords) - 20} more keywords")
                    break
            
            if st.button("📚 Set Dictionary"):
                st.session_state.classifier.set_dictionary(keywords)
                st.success(f"✅ Dictionary set with {len(keywords)} keywords!")
    
    # Current dictionary status
    if st.session_state.classifier.dictionary:
        st.markdown("---")
        st.markdown("### Current Dictionary Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Keywords Count", len(st.session_state.classifier.dictionary))
        with col2:
            st.metric("Ready for Classification", "✅" if st.session_state.classifier.dictionary else "❌")
        
        # Show current keywords
        with st.expander("View Current Dictionary"):
            current_keywords = " • ".join([f'`{kw}`' for kw in st.session_state.classifier.dictionary])
            st.markdown(current_keywords)
        
        # Export dictionary option
        dict_export = "\n".join(st.session_state.classifier.dictionary)
        st.download_button(
            "📥 Download Dictionary",
            dict_export,
            file_name="dictionary.txt",
            mime="text/plain"
        )

def classification_page():
    """Classification execution page."""
    st.markdown("## 🔍 Step 3: Run Classification")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Setup & Data section.")
        return
    
    if not st.session_state.classifier.dictionary:
        st.warning("⚠️ Please set up a dictionary in the Dictionary Management section.")
        return
    
    # Classification settings
    st.markdown("### Classification Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Data:** {len(st.session_state.classifier.data)} samples")
        st.info(f"**Dictionary:** {len(st.session_state.classifier.dictionary)} keywords")
    
    with col2:
        st.info(f"**Tactic:** {st.session_state.classifier.tactic}")
        st.info(f"**Text Column:** {st.session_state.classifier.text_column}")
    
    # Run classification
    if st.button("🚀 Run Classification", type="primary", use_container_width=True):
        with st.spinner("Running classification..."):
            try:
                results = st.session_state.classifier.classify_texts()
                st.session_state.classification_done = True
                st.success("✅ Classification completed!")
                
                # Show quick results
                metrics = results['metrics']
                
                st.markdown("### Quick Results")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.1%}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.1%}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1_score']:.1%}")
                
                # True positives preview
                true_positives = st.session_state.classifier.get_true_positives()
                
                if true_positives:
                    st.markdown(f"### 🎯 True Positives Found ({len(true_positives)})")
                    
                    for i, tp in enumerate(true_positives[:5]):
                        with st.expander(f"Example {i+1}: \"{tp['text'][:50]}...\""):
                            st.write(f"**Full Text:** {tp['text']}")
                            st.write(f"**Matched Keywords:** {', '.join(tp['matched_keywords'])}")
                    
                    if len(true_positives) > 5:
                        st.info(f"Showing 5 of {len(true_positives)} true positives. See Analysis & Results for full details.")
                
            except Exception as e:
                st.error(f"Classification failed: {str(e)}")

def analysis_results_page():
    """Analysis and results page."""
    st.markdown("## 📊 Step 4: Analysis & Results")
    
    if not st.session_state.classification_done:
        st.warning("⚠️ Please run classification first.")
        return
    
    # Performance metrics
    st.markdown("### 🎯 Overall Performance")
    
    results = st.session_state.classifier.classify_texts()
    metrics = results['metrics']
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <h2>{metrics['accuracy']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Precision</h3>
            <h2>{metrics['precision']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Recall</h3>
            <h2>{metrics['recall']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>F1-Score</h3>
            <h2>{metrics['f1_score']:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Confusion matrix
    st.markdown("### 📊 Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        conf_matrix_fig = create_confusion_matrix_plot(metrics)
        st.plotly_chart(conf_matrix_fig, use_container_width=True)
    
    with col2:
        st.markdown("**Classification Breakdown:**")
        st.write(f"• True Positives: {metrics['true_positives']}")
        st.write(f"• False Positives: {metrics['false_positives']}")
        st.write(f"• False Negatives: {metrics['false_negatives']}")
        st.write(f"• True Negatives: {metrics['true_negatives']}")
        
        st.markdown("**Interpretation:**")
        if metrics['precision'] > 0.8:
            st.success("🎯 High precision - low false alarm rate")
        elif metrics['precision'] > 0.6:
            st.warning("⚠️ Moderate precision - some false alarms")
        else:
            st.error("❌ Low precision - many false alarms")
        
        if metrics['recall'] > 0.8:
            st.success("📡 High recall - catching most positive cases")
        elif metrics['recall'] > 0.6:
            st.warning("⚠️ Moderate recall - missing some cases")
        else:
            st.error("❌ Low recall - missing many positive cases")
    
    # Keyword analysis
    st.markdown("### 🔍 Keyword Performance Analysis")
    
    analysis_tab = st.selectbox(
        "Select analysis metric:",
        ["f1_score", "precision", "recall"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        keyword_fig = create_keyword_performance_plot(
            st.session_state.classifier.keyword_analysis, 
            analysis_tab, 
            10
        )
        if keyword_fig:
            st.plotly_chart(keyword_fig, use_container_width=True)
    
    with col2:
        st.markdown(f"**Top Keywords by {analysis_tab.replace('_', ' ').title()}:**")
        
        top_keywords = st.session_state.classifier.get_top_keywords_by_metric(analysis_tab, 5)
        
        for i, kw in enumerate(top_keywords, 1):
            metric_value = kw[analysis_tab]
            color = "🟢" if metric_value > 0.7 else "🟡" if metric_value > 0.4 else "🔴"
            
            st.write(f"{color} **{i}. '{kw['keyword']}'**")
            st.write(f"   {analysis_tab.replace('_', ' ').title()}: {metric_value:.1%}")
            st.write(f"   TP: {kw['true_positives']}, FP: {kw['false_positives']}, FN: {kw['false_negatives']}")
            st.write("")
    
    # Detailed results
    st.markdown("### 📝 Detailed Results")
    
    tab1, tab2, tab3 = st.tabs(["True Positives", "False Positives", "False Negatives"])
    
    with tab1:
        true_positives = [r for r in results['results'] if r['classification_type'] == 'TP']
        st.write(f"Found {len(true_positives)} true positives")
        
        for i, tp in enumerate(true_positives):
            with st.expander(f"TP {i+1}: \"{tp['text'][:60]}...\""):
                st.write(f"**Text:** {tp['text']}")
                st.write(f"**Keywords:** {', '.join(tp['matched_keywords'])}")
    
    with tab2:
        false_positives = [r for r in results['results'] if r['classification_type'] == 'FP']
        st.write(f"Found {len(false_positives)} false positives")
        
        for i, fp in enumerate(false_positives):
            with st.expander(f"FP {i+1}: \"{fp['text'][:60]}...\""):
                st.write(f"**Text:** {fp['text']}")
                st.write(f"**Keywords:** {', '.join(fp['matched_keywords'])}")
                st.write("*This was incorrectly classified as positive*")
    
    with tab3:
        false_negatives = [r for r in results['results'] if r['classification_type'] == 'FN']
        st.write(f"Found {len(false_negatives)} false negatives")
        
        for i, fn in enumerate(false_negatives):
            with st.expander(f"FN {i+1}: \"{fn['text'][:60]}...\""):
                st.write(f"**Text:** {fn['text']}")
                st.write("*This positive case was missed - no keywords matched*")
    
    # Export options
    st.markdown("### 📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export classification results
        results_df = pd.DataFrame(results['results'])
        csv_results = results_df.to_csv(index=False)
        
        st.download_button(
            "📊 Download Classification Results",
            csv_results,
            file_name="classification_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export keyword analysis
        if st.session_state.classifier.keyword_analysis:
            keyword_df = pd.DataFrame([
                {
                    'keyword': kw['keyword'],
                    'true_positives': kw['true_positives'],
                    'false_positives': kw['false_positives'],
                    'false_negatives': kw['false_negatives'],
                    'precision': kw['precision'],
                    'recall': kw['recall'],
                    'f1_score': kw['f1_score']
                }
                for kw in st.session_state.classifier.keyword_analysis.values()
            ])
            
            csv_keywords = keyword_df.to_csv(index=False)
            
            st.download_button(
                "🔍 Download Keyword Analysis",
                csv_keywords,
                file_name="keyword_analysis.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
