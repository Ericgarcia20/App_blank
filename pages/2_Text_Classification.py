import streamlit as st
import pandas as pd
import json
from typing import Dict, Set, List, Tuple
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Dictionary Text Classifier",
    page_icon="🔍",
    layout="wide"
)

# Default dictionaries
DEFAULT_DICTIONARIES = {
    'urgency_marketing': [
        'limited', 'limited time', 'limited run', 'limited edition', 'order now', 
        'last chance', 'hurry', 'while supplies last', 'before they\'re gone', 
        'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only', 
        'expires soon', 'final hours', 'almost gone'
    ],
    'exclusive_marketing': [
        'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal', 
        'members only', 'vip', 'special access', 'invitation only', 'premium', 
        'privileged', 'limited access', 'select customers', 'insider', 
        'private sale', 'early access'
    ]
}

def classify_text(text: str, dictionaries: Dict[str, List[str]]) -> Tuple[str, Dict[str, int], List[str]]:
    """Classify text based on dictionary matches."""
    if pd.isna(text):
        return 'unclassified', {}, []
    
    text_lower = text.lower()
    category_scores = {}
    all_matches = []
    
    # Count matches for each category
    for category, terms in dictionaries.items():
        matches = [term for term in terms if term.lower() in text_lower]
        category_scores[category] = len(matches)
        all_matches.extend(matches)
    
    # Determine primary classification
    if not any(category_scores.values()):
        primary_class = 'unclassified'
    elif max(category_scores.values()) == 1 and sum(category_scores.values()) > 1:
        primary_class = 'mixed_marketing'
    else:
        primary_class = max(category_scores, key=category_scores.get)
    
    return primary_class, category_scores, all_matches

def process_classifications(df: pd.DataFrame, text_column: str, dictionaries: Dict[str, List[str]]) -> pd.DataFrame:
    """Process DataFrame and add classification columns."""
    result = df.copy()
    
    # Apply classification
    classifications = result[text_column].apply(lambda x: classify_text(x, dictionaries))
    
    # Extract results
    result['primary_classification'] = [c[0] for c in classifications]
    result['all_matched_terms'] = [c[2] for c in classifications]
    result['total_matches'] = [len(c[2]) for c in classifications]
    
    # Add individual category scores
    for category in dictionaries.keys():
        result[f'{category}_score'] = [c[1].get(category, 0) for c in classifications]
    
    return result

def main():
    st.title("🔍 Dictionary Text Classifier")
    st.markdown("Upload your dataset and customize dictionaries to classify text content.")
    
    # Sidebar for dictionary management
    st.sidebar.header("📚 Dictionary Management")
    
    # Initialize session state for dictionaries
    if 'dictionaries' not in st.session_state:
        # Ensure all dictionary values are lists (convert sets if needed)
        clean_dicts = {}
        for key, value in DEFAULT_DICTIONARIES.items():
            clean_dicts[key] = list(value) if not isinstance(value, list) else value
        st.session_state.dictionaries = clean_dicts
    
    # Dictionary editor
    st.sidebar.subheader("Edit Dictionaries")
    
    # Add new category
    with st.sidebar.expander("➕ Add New Category"):
        new_category = st.text_input("Category Name", key="new_cat")
        new_terms = st.text_area("Terms (one per line)", key="new_terms")
        if st.button("Add Category"):
            if new_category and new_terms:
                terms_list = [term.strip() for term in new_terms.split('\n') if term.strip()]
                st.session_state.dictionaries[new_category] = terms_list
                st.success(f"Added category: {new_category}")
                st.rerun()
    
    # Edit existing categories
    for category in list(st.session_state.dictionaries.keys()):
        with st.sidebar.expander(f"✏️ Edit {category}"):
            # Ensure terms is a list for joining
            current_terms = st.session_state.dictionaries[category]
            terms_list = list(current_terms) if not isinstance(current_terms, list) else current_terms
            terms_text = '\n'.join(terms_list)
            edited_terms = st.text_area(f"Terms for {category}", value=terms_text, key=f"edit_{category}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Update", key=f"update_{category}"):
                    terms_list = [term.strip() for term in edited_terms.split('\n') if term.strip()]
                    st.session_state.dictionaries[category] = terms_list
                    st.success(f"Updated {category}")
                    st.rerun()
            
            with col2:
                if st.button(f"Delete", key=f"delete_{category}"):
                    del st.session_state.dictionaries[category]
                    st.success(f"Deleted {category}")
                    st.rerun()
    
    # Reset to defaults
    if st.sidebar.button("🔄 Reset to Defaults"):
        # Ensure all values are lists when resetting
        clean_dicts = {}
        for key, value in DEFAULT_DICTIONARIES.items():
            clean_dicts[key] = list(value) if not isinstance(value, list) else value
        st.session_state.dictionaries = clean_dicts
        st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} rows")
                
                # Column selection
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                selected_column = st.selectbox("Select text column to classify:", text_columns)
                
                if selected_column:
                    # Show sample data
                    st.subheader("Sample Data")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Process button
                    if st.button("🚀 Classify Text", type="primary"):
                        with st.spinner("Processing classifications..."):
                            classified_df = process_classifications(df, selected_column, st.session_state.dictionaries)
                        
                        # Results
                        st.header("📊 Classification Results")
                        
                        # Summary metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        class_counts = classified_df['primary_classification'].value_counts()
                        total_classified = len(classified_df) - class_counts.get('unclassified', 0)
                        
                        with col_a:
                            st.metric("Total Texts", len(classified_df))
                        with col_b:
                            st.metric("Classified", total_classified)
                        with col_c:
                            st.metric("Unclassified", class_counts.get('unclassified', 0))
                        with col_d:
                            st.metric("Categories", len(st.session_state.dictionaries))
                        
                        # Classification distribution
                        st.subheader("Classification Distribution")
                        st.bar_chart(class_counts)
                        
                        # Detailed results table
                        st.subheader("Detailed Results")
                        
                        # Filter options
                        filter_class = st.selectbox(
                            "Filter by classification:",
                            ['All'] + list(class_counts.index)
                        )
                        
                        if filter_class != 'All':
                            display_df = classified_df[classified_df['primary_classification'] == filter_class]
                        else:
                            display_df = classified_df
                        
                        # Display columns
                        display_cols = [selected_column, 'primary_classification', 'total_matches', 'all_matched_terms']
                        st.dataframe(display_df[display_cols], use_container_width=True)
                        
                        # Download processed data
                        csv_buffer = StringIO()
                        classified_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="classified_results.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with col2:
        st.header("📋 Current Dictionaries")
        
        for category, terms in st.session_state.dictionaries.items():
            # Ensure terms is a list
            terms_list = list(terms) if not isinstance(terms, list) else terms
            with st.expander(f"{category} ({len(terms_list)} terms)"):
                for term in terms_list[:10]:  # Show first 10 terms
                    st.text(f"• {term}")
                if len(terms_list) > 10:
                    st.text(f"... and {len(terms_list) - 10} more")
        
        # Export/Import dictionaries
        st.subheader("💾 Dictionary I/O")
        
        # Export
        # Ensure all dictionary values are JSON serializable (lists, not sets)
        serializable_dicts = {}
        for key, value in st.session_state.dictionaries.items():
            serializable_dicts[key] = list(value) if not isinstance(value, list) else value
        
        dict_json = json.dumps(serializable_dicts, indent=2)
        st.download_button(
            label="Export Dictionaries (JSON)",
            data=dict_json,
            file_name="dictionaries.json",
            mime="application/json"
        )
        
        # Import
        uploaded_dict = st.file_uploader("Import Dictionaries (JSON)", type=['json'])
        if uploaded_dict is not None:
            try:
                imported_dict = json.load(uploaded_dict)
                if st.button("Load Imported Dictionaries"):
                    # Ensure imported dictionary values are lists
                    clean_imported = {}
                    for key, value in imported_dict.items():
                        clean_imported[key] = list(value) if not isinstance(value, list) else value
                    st.session_state.dictionaries = clean_imported
                    st.success("Dictionaries imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing dictionaries: {str(e)}")

if __name__ == "__main__":
    main()
