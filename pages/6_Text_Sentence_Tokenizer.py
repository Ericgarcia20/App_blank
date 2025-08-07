import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Text Sentence Tokenizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class TextSentenceTokenizer:
    def __init__(self):
        self.raw_df = None
        self.tokenized_df = None
        self.sentence_stats = {}
        self.custom_patterns = []
        self.min_sentence_length = 10
        self.max_sentence_length = 1000
        self.sentence_patterns = []

    def load_data_from_uploads(self, uploaded_file):
        """Load data from Streamlit file upload"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                self.raw_df = pd.read_csv(uploaded_file)
                st.success(f"✅ CSV data loaded: {self.raw_df.shape[0]} rows, {self.raw_df.shape[1]} columns")
                
            elif file_extension == 'txt':
                # Read text file line by line
                text_content = uploaded_file.read().decode('utf-8').split('\n')
                text_content = [line.strip() for line in text_content if line.strip()]
                self.raw_df = pd.DataFrame({'text': text_content})
                st.success(f"✅ TXT data loaded: {len(text_content)} lines")
                
            elif file_extension in ['xlsx', 'xls']:
                self.raw_df = pd.read_excel(uploaded_file)
                st.success(f"✅ Excel data loaded: {self.raw_df.shape[0]} rows, {self.raw_df.shape[1]} columns")
                
            else:
                st.error(f"❌ Unsupported file format: {file_extension}")
                return False
                
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return False

    def select_text_column(self, selected_column):
        """Set the selected text column"""
        if self.raw_df is None:
            return False
            
        try:
            self.raw_df['selected_text'] = self.raw_df[selected_column].astype(str).fillna('')
            return True
        except KeyError:
            st.error(f"❌ Column '{selected_column}' not found")
            return False

    def setup_tokenization_options(self, complexity, min_length, max_length, custom_patterns_text=None):
        """Setup tokenization options"""
        self.min_sentence_length = min_length
        self.max_sentence_length = max_length
        
        # Basic sentence tokenization patterns
        self.sentence_patterns = [
            r'[.!?]+\s+',  # Standard sentence endings followed by space
            r'[.!?]+$',    # End of text
        ]
        
        if complexity == 'Advanced':
            self.setup_advanced_patterns()
        elif complexity == 'Custom' and custom_patterns_text:
            self.setup_custom_patterns_from_text(custom_patterns_text)
        
        return True

    def setup_advanced_patterns(self):
        """Setup advanced tokenization patterns"""
        # Common abbreviations that shouldn't end sentences
        abbreviations_list = [
            'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Sr', 'Jr',
            'vs', 'etc', 'Inc', 'Corp', 'Ltd', 'Co',
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]
        
        self.sentence_patterns = [
            r'[.!?]+\s+',  # Standard sentence endings
            r'[.!?]+$',    # End of text
            r'[.!?]+\s*\n+', # Line breaks
        ]
        
        self.abbreviation_replacements = {}
        self._abbrev_regex = re.compile(r'\b(' + '|'.join(re.escape(abbr) for abbr in abbreviations_list) + r')\.')
        self._placeholder_prefix = "__ABBREV_DOT__"
        
        # Handle other special cases
        self.special_cases = {
            'urls': r'https?://[^\s]+',
            'emails': r'\S+@\S+\.\S+',
            'numbers_with_dots': r'\d+\.\d+',
            'ellipsis': r'\.{3,}',
        }
        self._special_case_regexes = {name: re.compile(pattern) for name, pattern in self.special_cases.items()}
        self._special_case_placeholders = {name: f"__PROTECTED_{name.upper()}__" for name in self.special_cases}

    def setup_custom_patterns_from_text(self, patterns_text):
        """Setup custom tokenization patterns from text input"""
        custom_patterns = []
        for pattern in patterns_text.split('\n'):
            pattern = pattern.strip()
            if pattern:
                try:
                    re.compile(pattern)
                    custom_patterns.append(pattern)
                except re.error:
                    st.warning(f"⚠️ Invalid regex pattern: {pattern}")
        
        if custom_patterns:
            self.sentence_patterns = custom_patterns
            # Disable advanced special case handling if custom patterns are used
            if hasattr(self, 'special_cases'):
                del self.special_cases
            if hasattr(self, '_special_case_regexes'):
                del self._special_case_regexes
            if hasattr(self, '_abbrev_regex'):
                del self._abbrev_regex
        else:
            st.warning("⚠️ No valid custom patterns, using basic patterns")
            self.sentence_patterns = [r'[.!?]+\s+', r'[.!?]+$']

    def tokenize_text_to_sentences(self):
        """Tokenize text data into sentences - FIXED to only output specific columns"""
        if self.raw_df is None or 'selected_text' not in self.raw_df.columns:
            st.error("❌ No text data available for tokenization")
            return False

        all_sentences = []
        total_texts = len(self.raw_df)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, row in self.raw_df.iterrows():
            if idx % 100 == 0:
                progress = (idx + 1) / total_texts
                progress_bar.progress(progress)
                status_text.text(f"Processing: {idx+1}/{total_texts} texts ({progress*100:.1f}%)")

            text = str(row['selected_text'])
            if pd.isna(text) or text.strip() == '':
                continue

            # Tokenize into sentences
            sentences = self.split_into_sentences(text)

            # Filter sentences by length
            valid_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if self.min_sentence_length <= len(sentence) <= self.max_sentence_length:
                    valid_sentences.append(sentence)

            # Store sentences with ONLY the required columns
            for sent_idx, sentence in enumerate(valid_sentences):
                sentence_data = {}
                
                # Only include the 5 specific columns shown in the image
                required_columns = ['shortcode', 'turn', 'caption', 'transcript', 'post_url']
                
                for col in required_columns:
                    if col in self.raw_df.columns:
                        sentence_data[col] = row[col]
                    else:
                        # Handle missing columns gracefully
                        if col == 'turn':
                            sentence_data[col] = sent_idx  # Use sentence position as turn
                        elif col == 'caption' or col == 'transcript':
                            sentence_data[col] = sentence  # Use the tokenized sentence
                        else:
                            sentence_data[col] = ''  # Empty string for missing columns
                
                # If caption or transcript was the selected text column, update it with the sentence
                if 'caption' in sentence_data and 'selected_text' in self.raw_df.columns:
                    if self.raw_df.columns[self.raw_df.columns.get_loc('selected_text')] in ['caption', 'transcript']:
                        sentence_data['caption' if 'caption' in required_columns else 'transcript'] = sentence
                
                all_sentences.append(sentence_data)

        # Create tokenized dataframe with only the required columns
        self.tokenized_df = pd.DataFrame(all_sentences)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Tokenization completed!")
        
        return True

    def split_into_sentences(self, text):
        """Split text into sentences using configured patterns"""
        sentences = [text]

        # If using advanced patterns, temporarily replace special cases and abbreviations
        if hasattr(self, 'special_cases') or hasattr(self, '_abbrev_regex'):
            text_to_split = text
            self.abbreviation_replacements = {}
            special_case_replacements = {}

            # Protect abbreviations first
            if hasattr(self, '_abbrev_regex'):
                def replace_abbrev(match):
                    original = match.group(0)
                    placeholder = f"{self._placeholder_prefix}{len(self.abbreviation_replacements)}__"
                    self.abbreviation_replacements[placeholder] = original
                    return original[:-1] + placeholder
                text_to_split = self._abbrev_regex.sub(replace_abbrev, text_to_split)

            # Protect other special cases
            if hasattr(self, '_special_case_regexes'):
                for name, regex in self._special_case_regexes.items():
                    placeholder = self._special_case_placeholders[name]
                    matches = regex.findall(text_to_split)
                    for i, match in enumerate(matches):
                        unique_placeholder = f"{placeholder}_{i}__"
                        special_case_replacements[unique_placeholder] = match
                        text_to_split = text_to_split.replace(match, unique_placeholder, 1)

            # Combine all replacements
            all_replacements = {**self.abbreviation_replacements, **special_case_replacements}

            # Apply sentence splitting patterns to the protected text
            current_sentences = [text_to_split]

            for pattern in self.sentence_patterns:
                next_sentences = []
                for sentence in current_sentences:
                    parts = re.split(f'({pattern})', sentence)
                    temp_sentence = ""
                    for part in parts:
                        if re.match(pattern, part):
                            temp_sentence += part
                            if temp_sentence.strip():
                                next_sentences.append(temp_sentence.strip())
                            temp_sentence = ""
                        else:
                            temp_sentence += part
                    if temp_sentence.strip():
                        next_sentences.append(temp_sentence.strip())
                current_sentences = next_sentences
                if len(current_sentences) > len(sentences):
                    sentences = current_sentences

            # Restore the original special cases and abbreviations
            restored_sentences = []
            for sentence in sentences:
                restored_sentence = sentence
                for placeholder, original in sorted(all_replacements.items(), key=lambda item: len(item[0]), reverse=True):
                    restored_sentence = restored_sentence.replace(placeholder, original)
                restored_sentences.append(restored_sentence)
            sentences = restored_sentences

        else:  # Basic tokenization
            current_sentences = [text]
            for pattern in self.sentence_patterns:
                next_sentences = []
                for sentence in current_sentences:
                    parts = re.split(f'({pattern})', sentence)
                    temp_sentence = ""
                    for part in parts:
                        if re.match(pattern, part):
                            temp_sentence += part
                            if temp_sentence.strip():
                                next_sentences.append(temp_sentence.strip())
                            temp_sentence = ""
                        else:
                            temp_sentence += part
                    if temp_sentence.strip():
                        next_sentences.append(temp_sentence.strip())
                current_sentences = next_sentences
                if len(current_sentences) > len(sentences):
                    sentences = current_sentences

        return [s for s in sentences if s.strip()]

    def analyze_sentence_statistics(self):
        """Analyze statistics of the tokenized sentences"""
        if self.tokenized_df is None or self.tokenized_df.empty:
            return

        total_sentences = len(self.tokenized_df)
        
        # Since we're only keeping specific columns, we need to adapt the analysis
        sentence_lengths = []
        word_counts = []
        
        # Determine which column contains the actual sentence text
        text_columns = ['caption', 'transcript']
        sentence_text_col = None
        
        for col in text_columns:
            if col in self.tokenized_df.columns:
                sentence_text_col = col
                break
        
        if sentence_text_col:
            for text in self.tokenized_df[sentence_text_col]:
                if pd.notna(text) and str(text).strip():
                    sentence_lengths.append(len(str(text)))
                    word_counts.append(len(str(text).split()))

        # Calculate statistics
        length_stats_desc = pd.Series(sentence_lengths).describe() if sentence_lengths else pd.Series([])
        word_count_stats_desc = pd.Series(word_counts).describe() if word_counts else pd.Series([])

        # Content analysis
        punct_count = 0
        cap_count = 0
        
        if sentence_text_col:
            for text in self.tokenized_df[sentence_text_col]:
                if pd.notna(text) and str(text).strip():
                    if re.search(r'[.!?]', str(text)):
                        punct_count += 1
                    if re.search(r'[A-Z]', str(text)):
                        cap_count += 1

        punct_pct = punct_count / total_sentences * 100 if total_sentences > 0 else 0
        cap_pct = cap_count / total_sentences * 100 if total_sentences > 0 else 0

        # Store statistics
        self.sentence_stats = {
            'total_sentences': total_sentences,
            'length_stats': length_stats_desc,
            'word_count_stats': word_count_stats_desc,
            'punctuation_pct': punct_pct,
            'capitalization_pct': cap_pct
        }

    def search_sentences(self, query, max_results=20):
        """Search sentences containing specific text"""
        if self.tokenized_df is None or self.tokenized_df.empty:
            return pd.DataFrame()

        # Search in caption and transcript columns
        text_columns = ['caption', 'transcript']
        matches = pd.DataFrame()
        
        for col in text_columns:
            if col in self.tokenized_df.columns:
                col_matches = self.tokenized_df[
                    self.tokenized_df[col].astype(str).str.contains(query, case=False, na=False)
                ]
                matches = pd.concat([matches, col_matches], ignore_index=True)
        
        # Remove duplicates and return top results
        matches = matches.drop_duplicates().head(max_results)
        return matches

# Initialize session state
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = TextSentenceTokenizer()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'tokenization_done' not in st.session_state:
    st.session_state.tokenization_done = False

def data_upload_page():
    st.markdown("<h2 class='section-header'>📁 Data Upload</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload your text data file to get started:
    - **CSV**: Must have a text column
    - **TXT**: Each line will be treated as a separate text entry
    - **Excel**: Must have a text column
    """)

    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'txt', 'xlsx', 'xls'],
        help="Upload CSV, TXT, or Excel files containing text data"
    )

    if uploaded_file is not None:
        if st.session_state.tokenizer.load_data_from_uploads(uploaded_file):
            st.session_state.data_loaded = True
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(st.session_state.tokenizer.raw_df.head(), use_container_width=True)
            
            # Column selection
            if len(st.session_state.tokenizer.raw_df.columns) > 1:
                st.subheader("📋 Select Text Column")
                
                # Show column options with sample data
                col_options = {}
                for col in st.session_state.tokenizer.raw_df.columns:
                    sample_text = str(st.session_state.tokenizer.raw_df[col].iloc[0])
                    preview = sample_text[:50] + "..." if len(sample_text) > 50 else sample_text
                    col_options[col] = f"{col}: {preview}"
                
                selected_column = st.selectbox(
                    "Choose the column containing text data:",
                    options=list(col_options.keys()),
                    format_func=lambda x: col_options[x],
                    index=0 if 'text' not in st.session_state.tokenizer.raw_df.columns else list(st.session_state.tokenizer.raw_df.columns).index('text')
                )
                
                if st.button("✅ Confirm Column Selection", type="primary"):
                    if st.session_state.tokenizer.select_text_column(selected_column):
                        st.success(f"✅ Selected column: {selected_column}")
                        st.session_state.column_selected = True
            else:
                # Auto-select single column
                st.session_state.tokenizer.select_text_column(st.session_state.tokenizer.raw_df.columns[0])
                st.success(f"✅ Auto-selected column: {st.session_state.tokenizer.raw_df.columns[0]}")
                st.session_state.column_selected = True

def tokenization_page():
    st.markdown("<h2 class='section-header'>🔧 Tokenization Configuration</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the Data Upload section.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Tokenization Settings")
        
        # Complexity selection
        complexity = st.selectbox(
            "Choose tokenization complexity:",
            ["Basic", "Advanced", "Custom"],
            help="Basic: Standard punctuation | Advanced: Handles abbreviations, URLs | Custom: Define your own patterns"
        )
        
        # Length filters
        min_length = st.slider("Minimum sentence length (characters)", 1, 100, 10)
        max_length = st.slider("Maximum sentence length (characters)", 100, 2000, 1000)
        
        # Custom patterns if selected
        custom_patterns_text = None
        if complexity == "Custom":
            st.subheader("📝 Custom Patterns")
            custom_patterns_text = st.text_area(
                "Enter regex patterns (one per line):",
                height=150,
                placeholder="r'[.!?]+\\s+'\nr'\\n\\n+'\nr'[.!?]+\\s*(?=[A-Z])'"
            )
            
    with col2:
        st.subheader("📊 Current Data Info")
        if hasattr(st.session_state.tokenizer, 'raw_df') and st.session_state.tokenizer.raw_df is not None:
            st.metric("Total Texts", len(st.session_state.tokenizer.raw_df))
            
            if 'selected_text' in st.session_state.tokenizer.raw_df.columns:
                avg_length = st.session_state.tokenizer.raw_df['selected_text'].str.len().mean()
                st.metric("Average Text Length", f"{avg_length:.0f} chars")
        
        # Expected output columns info
        st.subheader("📋 Output Columns")
        st.info("The output CSV will contain only these columns:\n• shortcode\n• turn\n• caption\n• transcript\n• post_url")
        
        # Pattern examples
        st.subheader("💡 Pattern Examples")
        with st.expander("See examples"):
            st.code("""
# Basic patterns
r'[.!?]+\\s+'    # Standard punctuation
r'[.!?]+$'       # End of text

# Advanced patterns  
r'\\n\\n+'       # Paragraph breaks
r'[.!?]+\\s*(?=[A-Z])'  # Before capitals

# Social media patterns
r'(?<=[.!?])\\s*#'  # Before hashtags
r'(?<=[.!?])\\s*@'  # Before mentions
            """)

    # Tokenization button
    if st.button("🚀 Start Tokenization", type="primary"):
        if st.session_state.tokenizer.setup_tokenization_options(
            complexity, min_length, max_length, custom_patterns_text
        ):
            with st.spinner("Tokenizing text data..."):
                if st.session_state.tokenizer.tokenize_text_to_sentences():
                    st.session_state.tokenization_done = True
                    st.success("🎉 Tokenization completed!")
                    
                    # Show quick stats
                    total_sentences = len(st.session_state.tokenizer.tokenized_df)
                    st.metric("Sentences Created", total_sentences)
                    
                    # Show column preview
                    st.subheader("📋 Output Preview")
                    st.dataframe(st.session_state.tokenizer.tokenized_df.head(), use_container_width=True)

def analysis_page():
    st.markdown("<h2 class='section-header'>📊 Analysis & Results</h2>", unsafe_allow_html=True)
    
    if not st.session_state.tokenization_done:
        st.warning("⚠️ Please complete tokenization first.")
        return

    # Calculate statistics
    st.session_state.tokenizer.analyze_sentence_statistics()
    stats = st.session_state.tokenizer.sentence_stats

    # Check if stats is valid
    if stats is None or not isinstance(stats, dict):
        st.error("❌ Unable to generate statistics. Please try tokenizing again.")
        return

    # Overview metrics
    st.subheader("📈 Overview Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sentences = stats.get('total_sentences', 0)
        st.metric("Total Sentences", f"{total_sentences:,}")
    with col2:
        punctuation_pct = stats.get('punctuation_pct', 0)
        st.metric("With Punctuation", f"{punctuation_pct:.1f}%")
    with col3:
        capitalization_pct = stats.get('capitalization_pct', 0)
        st.metric("With Capitalization", f"{capitalization_pct:.1f}%")
    with col4:
        # Show number of output columns
        if st.session_state.tokenizer.tokenized_df is not None:
            st.metric("Output Columns", len(st.session_state.tokenizer.tokenized_df.columns))

    # Detailed statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📏 Length Statistics")
        try:
            length_stats = stats.get('length_stats')
            if length_stats is not None and hasattr(length_stats, 'index') and hasattr(length_stats, 'values'):
                length_df = pd.DataFrame({
                    'Statistic': length_stats.index,
                    'Value': length_stats.values
                })
                st.dataframe(length_df, use_container_width=True)
            else:
                st.info("Length statistics not available")
        except Exception as e:
            st.info("Length statistics not available")
    
    with col2:
        st.subheader("📝 Word Count Statistics")
        try:
            word_count_stats = stats.get('word_count_stats')
            if word_count_stats is not None and hasattr(word_count_stats, 'index') and hasattr(word_count_stats, 'values'):
                word_df = pd.DataFrame({
                    'Statistic': word_count_stats.index,
                    'Value': word_count_stats.values
                })
                st.dataframe(word_df, use_container_width=True)
            else:
                st.info("Word count statistics not available")
        except Exception as e:
            st.info("Word count statistics not available")

    # Show output format
    st.subheader("📋 Output Format Preview")
    if st.session_state.tokenizer.tokenized_df is not None:
        st.dataframe(st.session_state.tokenizer.tokenized_df.head(), use_container_width=True)
        st.info(f"Output contains exactly {len(st.session_state.tokenizer.tokenized_df.columns)} columns: {', '.join(st.session_state.tokenizer.tokenized_df.columns)}")

def sample_sentences_page():
    st.markdown("<h2 class='section-header'>📝 Sample Sentences</h2>", unsafe_allow_html=True)
    
    if not st.session_state.tokenization_done:
        st.warning("⚠️ Please complete tokenization first.")
        return

    tokenized_df = st.session_state.tokenizer.tokenized_df
    
    if tokenized_df is None or tokenized_df.empty:
        st.error("❌ No tokenized data available")
        return
    
    # Show sample entries
    st.subheader("🔀 Random Samples")
    random_samples = tokenized_df.sample(min(10, len(tokenized_df)))
    
    for idx, (_, row) in enumerate(random_samples.iterrows(), 1):
        with st.expander(f"Sample {idx}"):
            for col in row.index:
                if pd.notna(row[col]) and str(row[col]).strip():
                    st.write(f"**{col}:** {row[col]}")

def search_page():
    st.markdown("<h2 class='section-header'>🔍 Search Sentences</h2>", unsafe_allow_html=True)
    
    if not st.session_state.tokenization_done:
        st.warning("⚠️ Please complete tokenization first.")
        return

    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter search query:", placeholder="Search for specific text...")
    
    with col2:
        max_results = st.selectbox("Max results:", [10, 20, 50, 100], index=1)

    if query:
        matches = st.session_state.tokenizer.search_sentences(query, max_results)
        
        if not matches.empty:
            st.success(f"✅ Found {len(matches)} sentences containing '{query}'")
            
            # Display results
            for idx, (_, row) in enumerate(matches.iterrows(), 1):
                with st.expander(f"Result {idx}"):
                    for col in row.index:
                        if pd.notna(row[col]) and str(row[col]).strip():
                            # Highlight the query if it's in this field
                            text = str(row[col])
                            if query.lower() in text.lower():
                                highlighted = re.sub(f'({re.escape(query)})', r'**\1**', text, flags=re.IGNORECASE)
                                st.markdown(f"**{col}:** {highlighted}")
                            else:
                                st.write(f"**{col}:** {text}")
        else:
            st.warning("❌ No sentences found containing the query")

def export_page():
    st.markdown("<h2 class='section-header'>💾 Export Results</h2>", unsafe_allow_html=True)
    
    if not st.session_state.tokenization_done:
        st.warning("⚠️ Please complete tokenization first.")
        return

    tokenized_df = st.session_state.tokenizer.tokenized_df
    
    if tokenized_df is None or tokenized_df.empty:
        st.error("❌ No tokenized data to export")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Tokenized Sentences")
        st.success(f"✅ {len(tokenized_df)} sentences available")
        st.info(f"Columns: {', '.join(tokenized_df.columns)}")
        
        # Convert DataFrame to CSV
        csv_data = tokenized_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Tokenized Sentences (CSV)",
            data=csv_data,
            file_name="tokenized_sentences.csv",
            mime="text/csv"
        )
        
        # Preview
        with st.expander("👀 Preview Data"):
            st.dataframe(tokenized_df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📋 Export Info")
        st.write("**Output format confirmed:**")
        st.write("✅ Contains only required columns")
        st.write("✅ Ready for immediate use")
        
        # Column verification
        expected_columns = ['shortcode', 'turn', 'caption', 'transcript', 'post_url']
        actual_columns = list(tokenized_df.columns)
        
        st.write("**Column verification:**")
        for col in expected_columns:
            if col in actual_columns:
                st.write(f"✅ {col}")
            else:
                st.write(f"❌ {col} (missing)")
        
        # Sample export
        st.subheader("📝 Sample Export")
        sample_size = st.slider("Sample size:", 10, min(100, len(tokenized_df)), 20)
        sample_df = tokenized_df.head(sample_size)
        
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample (CSV)",
            data=sample_csv,
            file_name=f"sample_tokenized_{sample_size}.csv",
            mime="text/csv"
        )

def show_progress():
    """Show progress in sidebar"""
    st.sidebar.markdown("### 📋 Progress")
    
    steps = [
        ("📁 Data Upload", st.session_state.data_loaded),
        ("🔧 Tokenization", st.session_state.tokenization_done),
    ]
    
    for step, completed in steps:
        if completed:
            st.sidebar.success(f"✅ {step}")
        else:
            st.sidebar.info(f"⏳ {step}")

def show_data_stats():
    """Show data statistics in sidebar"""
    if st.session_state.data_loaded:
        tokenizer = st.session_state.tokenizer
        
        st.sidebar.markdown("### 📊 Data Info")
        
        if tokenizer.raw_df is not None:
            st.sidebar.metric("Original Texts", len(tokenizer.raw_df))
        
        if st.session_state.tokenization_done and tokenizer.tokenized_df is not None:
            st.sidebar.metric("Total Sentences", len(tokenizer.tokenized_df))
            st.sidebar.metric("Output Columns", len(tokenizer.tokenized_df.columns))

def main():
    st.markdown("<h1 class='main-header'>📝 Text Sentence Tokenizer</h1>", unsafe_allow_html=True)
    st.markdown("### Transform your text data into sentence-level format with specific columns")

    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    
    # Show progress and stats
    show_progress()
    show_data_stats()
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📁 Data Upload", "🔧 Tokenization", "📊 Analysis", "📝 Sample Sentences", "🔍 Search", "💾 Export"]
    )

    # Add important note about output format
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Output Format")
    st.sidebar.info("CSV will contain ONLY these columns:\n• shortcode\n• turn\n• caption\n• transcript\n• post_url")
    
    # Add help section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ❓ Help")
    
    with st.sidebar.expander("📋 Supported Formats"):
        st.markdown("""
        **CSV Files:**
        - Must have at least one text column
        - Column headers in first row
        
        **TXT Files:**
        - Each line becomes a separate text entry
        - UTF-8 encoding recommended
        
        **Excel Files:**
        - .xlsx or .xls format
        - Must have at least one text column
        """)
    
    with st.sidebar.expander("🔧 Tokenization Types"):
        st.markdown("""
        **Basic:** Standard punctuation splitting
        - Periods, exclamations, questions
        - Simple and fast
        
        **Advanced:** Smart splitting
        - Handles abbreviations (Dr., Mr., etc.)
        - Protects URLs and emails
        - More accurate results
        
        **Custom:** Your own patterns
        - Define regex patterns
        - Maximum flexibility
        - Requires regex knowledge
        """)
    
    with st.sidebar.expander("💡 Tips"):
        st.markdown("""
        - Upload data with required columns if available
        - Select the text column to tokenize
        - Output will only contain the 5 specified columns
        - Missing columns will be handled automatically
        - Check sample results before full export
        """)

    # Main page routing
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "🔧 Tokenization":
        tokenization_page()
    elif page == "📊 Analysis":
        analysis_page()
    elif page == "📝 Sample Sentences":
        sample_sentences_page()
    elif page == "🔍 Search":
        search_page()
    elif page == "💾 Export":
        export_page()

# Run the app
if __name__ == "__main__":
    main()
