import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import json
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Rolling Context Window Processor",
    page_icon="🔄",
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

class RollingContextProcessor:
    def __init__(self):
        self.raw_df = None
        self.processed_df = None
        self.context_windows = []
        self.window_stats = {}
        self.conversation_id_col = None
        self.message_col = None
        self.timestamp_col = None
        self.speaker_col = None
        self.id_col = None  # Added for ID column
        self.window_size = 5
        self.overlap = 1
        self.include_system_msgs = True
        self.format_type = 'structured'

    def load_data_from_uploads(self, uploaded_file):
        """Load data from Streamlit file upload"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                self.raw_df = pd.read_csv(uploaded_file)
                st.success(f"✅ CSV data loaded: {self.raw_df.shape[0]} rows, {self.raw_df.shape[1]} columns")
                
            elif file_extension == 'json':
                json_data = json.loads(uploaded_file.read().decode('utf-8'))
                
                if isinstance(json_data, list):
                    if all(isinstance(item, dict) for item in json_data):
                        self.raw_df = pd.DataFrame(json_data)
                    else:
                        st.error("❌ Unsupported JSON structure")
                        return False
                elif isinstance(json_data, dict):
                    if 'conversations' in json_data:
                        conversations = []
                        for conv_id, conv_data in enumerate(json_data['conversations']):
                            if isinstance(conv_data, list):
                                for msg_id, message in enumerate(conv_data):
                                    message_entry = {
                                        'conversation_id': conv_id,
                                        'message_id': msg_id,
                                        'message': message if isinstance(message, str) else str(message)
                                    }
                                    conversations.append(message_entry)
                        self.raw_df = pd.DataFrame(conversations)
                    else:
                        self.raw_df = pd.json_normalize(json_data)
                        
                st.success(f"✅ JSON data loaded: {self.raw_df.shape[0]} rows, {self.raw_df.shape[1]} columns")
                
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

    def configure_columns(self, message_col, conversation_id_col=None, speaker_col=None, timestamp_col=None, id_col=None):
        """Configure which columns contain conversation data"""
        self.message_col = message_col
        self.conversation_id_col = conversation_id_col if conversation_id_col != "None" else None
        self.speaker_col = speaker_col if speaker_col != "None" else None
        self.timestamp_col = timestamp_col if timestamp_col != "None" else None
        self.id_col = id_col if id_col != "None" else None
        
        # Auto-generate conversation IDs if not provided
        if self.conversation_id_col is None:
            self.raw_df['auto_conversation_id'] = 0
            self.conversation_id_col = 'auto_conversation_id'
        
        return True

    def setup_window_parameters(self, window_size, overlap, format_type, include_system_msgs):
        """Setup rolling window parameters"""
        self.window_size = window_size
        self.overlap = overlap
        self.format_type = format_type
        self.include_system_msgs = include_system_msgs
        return True

    def process_rolling_windows(self):
        """Process conversations into rolling context windows - FIXED to output specific columns only"""
        if self.raw_df is None:
            return False
        
        self.context_windows = []
        window_count = 0
        
        # Group by conversation
        conversations = self.raw_df.groupby(self.conversation_id_col)
        total_conversations = len(conversations)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for conv_idx, (conv_id, conv_data) in enumerate(conversations):
            progress = (conv_idx + 1) / total_conversations
            progress_bar.progress(progress)
            status_text.text(f"Processing conversation {conv_idx+1}/{total_conversations}")
            
            # Sort by timestamp if available
            if self.timestamp_col and self.timestamp_col in conv_data.columns:
                try:
                    conv_data = conv_data.sort_values(self.timestamp_col)
                except:
                    pass
            
            # Filter messages if needed
            messages = conv_data.copy()
            if not self.include_system_msgs and self.speaker_col:
                system_indicators = ['system', 'bot', 'assistant', 'auto', 'server']
                system_mask = messages[self.speaker_col].astype(str).str.lower().str.contains('|'.join(system_indicators), na=False)
                messages = messages[~system_mask]
            
            # Create rolling windows for this conversation
            num_messages = len(messages)
            if num_messages < self.window_size:
                window = self.create_context_window(messages, conv_id, window_count)
                if window:
                    self.context_windows.append(window)
                    window_count += 1
            else:
                stride = self.window_size - self.overlap
                for start_idx in range(0, num_messages - self.window_size + 1, stride):
                    end_idx = start_idx + self.window_size
                    window_messages = messages.iloc[start_idx:end_idx]
                    
                    window = self.create_context_window(window_messages, conv_id, window_count)
                    if window:
                        self.context_windows.append(window)
                        window_count += 1
        
        # Create processed DataFrame with ONLY the required columns
        if self.context_windows:
            self.processed_df = pd.DataFrame(self.context_windows)
            progress_bar.progress(1.0)
            status_text.text("✅ Processing completed!")
            return True
        else:
            status_text.text("❌ No context windows were created")
            return False

    def create_context_window(self, messages, conv_id, window_count):
        """Create a single context window with only the required columns"""
        try:
            # Format the context based on format type
            if self.format_type == 'simple':
                context = ' '.join(messages[self.message_col].astype(str).tolist())
                
            elif self.format_type == 'structured':
                context_parts = []
                for _, msg in messages.iterrows():
                    if self.speaker_col and pd.notna(msg[self.speaker_col]):
                        context_parts.append(f"Turn {len(context_parts)+1} ({msg[self.speaker_col]}): {msg[self.message_col]}")
                    else:
                        context_parts.append(f"Turn {len(context_parts)+1}: {msg[self.message_col]}")
                context = '\n'.join(context_parts)
                
            elif self.format_type == 'json':
                json_messages = []
                for i, (_, msg) in enumerate(messages.iterrows()):
                    msg_obj = {'turn': i+1, 'message': str(msg[self.message_col])}
                    if self.speaker_col and pd.notna(msg[self.speaker_col]):
                        msg_obj['speaker'] = str(msg[self.speaker_col])
                    if self.timestamp_col and pd.notna(msg[self.timestamp_col]):
                        msg_obj['timestamp'] = str(msg[self.timestamp_col])
                    json_messages.append(msg_obj)
                context = json.dumps(json_messages, ensure_ascii=False)
                
            elif self.format_type == 'conversation':
                context_parts = []
                for i, (_, msg) in enumerate(messages.iterrows()):
                    speaker = msg[self.speaker_col] if self.speaker_col and pd.notna(msg[self.speaker_col]) else f"Speaker{i%2+1}"
                    context_parts.append(f"Turn {i+1} ({speaker}): {msg[self.message_col]}")
                context = '\n'.join(context_parts)
            
            # Get the first message for Statement column and other required fields
            first_message = messages.iloc[0]
            last_message = messages.iloc[-1]
            
            # Create window with ONLY the 5 required columns
            window = {
                'ID': first_message[self.id_col] if self.id_col and self.id_col in first_message.index else f"WIN_{window_count:06d}",
                'Turn': len(messages),  # Number of turns/messages in this window
                'Speaker': first_message[self.speaker_col] if self.speaker_col and pd.notna(first_message[self.speaker_col]) else 'unknown',
                'Context': context,
                'Statement': str(last_message[self.message_col])  # The last message as the statement
            }
            
            return window
            
        except Exception as e:
            st.error(f"Error creating window: {e}")
            return None

    def analyze_window_statistics(self):
        """Analyze statistics of the processed context windows"""
        if self.processed_df is None or self.processed_df.empty:
            return

        total_windows = len(self.processed_df)
        
        # Calculate statistics based on available columns
        context_lengths = self.processed_df['Context'].str.len()
        word_counts = self.processed_df['Context'].str.split().str.len()
        turn_counts = self.processed_df['Turn']

        self.window_stats = {
            'total_windows': total_windows,
            'context_length_stats': context_lengths.describe(),
            'word_count_stats': word_counts.describe(),
            'turn_count_stats': turn_counts.describe(),
            'unique_speakers': self.processed_df['Speaker'].nunique() if 'Speaker' in self.processed_df.columns else 0,
            'format_type': self.format_type
        }

    def search_windows(self, query, max_results=20):
        """Search context windows containing specific text"""
        if self.processed_df is None or self.processed_df.empty:
            return pd.DataFrame()

        # Search in Context and Statement columns
        context_matches = self.processed_df[
            self.processed_df['Context'].str.contains(query, case=False, na=False)
        ]
        
        statement_matches = self.processed_df[
            self.processed_df['Statement'].str.contains(query, case=False, na=False)
        ]
        
        # Combine and remove duplicates
        matches = pd.concat([context_matches, statement_matches]).drop_duplicates()
        return matches.head(max_results)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = RollingContextProcessor()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'columns_configured' not in st.session_state:
    st.session_state.columns_configured = False
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

def data_upload_page():
    st.markdown("<h2 class='section-header'>📁 Data Upload</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload your conversation dataset to get started:
    - **CSV**: Conversation data with message columns
    - **JSON**: Conversation arrays or nested structures
    - **Excel**: Spreadsheet-based conversation data
    """)

    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['csv', 'json', 'xlsx', 'xls'],
        help="Upload CSV, JSON, or Excel files containing conversation data"
    )

    if uploaded_file is not None:
        if st.session_state.processor.load_data_from_uploads(uploaded_file):
            st.session_state.data_loaded = True
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(st.session_state.processor.raw_df.head(), use_container_width=True)
            
            # Show column info
            st.subheader("📋 Column Information")
            col_info = pd.DataFrame({
                'Column': st.session_state.processor.raw_df.columns,
                'Type': st.session_state.processor.raw_df.dtypes,
                'Sample': [str(st.session_state.processor.raw_df[col].iloc[0])[:50] + "..." 
                          if len(str(st.session_state.processor.raw_df[col].iloc[0])) > 50 
                          else str(st.session_state.processor.raw_df[col].iloc[0]) 
                          for col in st.session_state.processor.raw_df.columns]
            })
            st.dataframe(col_info, use_container_width=True)

def column_configuration_page():
    st.markdown("<h2 class='section-header'>📋 Column Configuration</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the Data Upload section.")
        return

    st.markdown("Configure which columns contain your conversation data:")
    
    columns = list(st.session_state.processor.raw_df.columns)
    column_options = ["None"] + columns
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Required: Message column
        st.subheader("📝 Required Fields")
        message_col = st.selectbox(
            "Message/Text Column *",
            columns,
            help="Column containing the actual message text"
        )
        
        # Optional: ID column for the output ID field
        id_col = st.selectbox(
            "ID Column",
            column_options,
            help="Column to use for the output ID field (will auto-generate if None)"
        )
        
        # Optional: Conversation ID
        conversation_id_col = st.selectbox(
            "Conversation ID Column",
            column_options,
            help="Column that groups messages into conversations (will auto-generate if None)"
        )
    
    with col2:
        # Optional: Speaker column
        st.subheader("🔗 Optional Fields")
        speaker_col = st.selectbox(
            "Speaker/Role Column",
            column_options,
            help="Column indicating who sent each message"
        )
        
        # Optional: Timestamp column
        timestamp_col = st.selectbox(
            "Timestamp Column",
            column_options,
            help="Column with message timestamps for chronological ordering"
        )
    
    # Expected output format info
    st.subheader("📋 Output Format")
    st.info("""
    The output will contain exactly these 5 columns:
    • **ID** - Unique identifier for each window
    • **Turn** - Number of turns/messages in the window
    • **Speaker** - Speaker from the first message in window
    • **Context** - Formatted conversation context
    • **Statement** - The last message in the window
    """)
    
    # Configuration preview
    st.subheader("🔍 Configuration Preview")
    
    if message_col:
        preview_data = {
            'Message': st.session_state.processor.raw_df[message_col].head(3).tolist(),
        }
        
        if id_col != "None":
            preview_data['ID'] = st.session_state.processor.raw_df[id_col].head(3).tolist()
        
        if conversation_id_col != "None":
            preview_data['Conversation ID'] = st.session_state.processor.raw_df[conversation_id_col].head(3).tolist()
        
        if speaker_col != "None":
            preview_data['Speaker'] = st.session_state.processor.raw_df[speaker_col].head(3).tolist()
            
        if timestamp_col != "None":
            preview_data['Timestamp'] = st.session_state.processor.raw_df[timestamp_col].head(3).tolist()
        
        preview_df = pd.DataFrame(preview_data)
        st.dataframe(preview_df, use_container_width=True)
    
    # Confirm configuration
    if st.button("✅ Confirm Configuration", type="primary"):
        if st.session_state.processor.configure_columns(
            message_col, conversation_id_col, speaker_col, timestamp_col, id_col
        ):
            st.session_state.columns_configured = True
            st.success("🎉 Configuration saved successfully!")
            
            # Show configuration summary
            st.info(f"""
            **Configuration Summary:**
            - Message column: {message_col}
            - ID column: {id_col if id_col != 'None' else 'Auto-generated'}
            - Conversation ID: {conversation_id_col if conversation_id_col != 'None' else 'Auto-generated'}
            - Speaker column: {speaker_col if speaker_col != 'None' else 'Not specified'}
            - Timestamp column: {timestamp_col if timestamp_col != 'None' else 'Not specified'}
            """)

def window_configuration_page():
    st.markdown("<h2 class='section-header'>🔧 Window Configuration</h2>", unsafe_allow_html=True)
    
    if not st.session_state.columns_configured:
        st.warning("⚠️ Please configure columns first.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Window Parameters")
        
        # Window size
        window_size = st.slider(
            "Window Size (messages per window)",
            min_value=2,
            max_value=20,
            value=5,
            help="Number of messages to include in each context window"
        )
        
        # Overlap
        overlap = st.slider(
            "Overlap (messages)",
            min_value=0,
            max_value=min(window_size-1, 10),
            value=1,
            help="Number of messages to overlap between consecutive windows"
        )
        
        # Calculate stride
        stride = window_size - overlap
        st.info(f"**Stride:** {stride} messages (window will advance by {stride} messages each time)")
        
        # System message handling
        include_system_msgs = st.checkbox(
            "Include system messages",
            value=True,
            help="Whether to include system/bot messages in context windows"
        )
    
    with col2:
        st.subheader("📄 Output Format")
        
        format_type = st.selectbox(
            "Context Format",
            ["structured", "simple", "json", "conversation"],
            help="How to format the context window output"
        )
        
        # Expected output preview
        st.subheader("📋 Expected Output Columns")
        expected_columns = ["ID", "Turn", "Speaker", "Context", "Statement"]
        for i, col in enumerate(expected_columns, 1):
            st.write(f"{i}. **{col}**")
        
        # Format examples
        st.subheader("💡 Format Examples")
        
        with st.expander("See format examples"):
            if format_type == "structured":
                st.code("Turn 1 (User): Hello, I need help\nTurn 2 (Assistant): How can I help you?\nTurn 3 (User): I have a question")
            elif format_type == "simple":
                st.code("Hello, I need help How can I help you? I have a question")
            elif format_type == "json":
                st.code('[{"turn": 1, "speaker": "User", "message": "Hello, I need help"}, {"turn": 2, "speaker": "Assistant", "message": "How can I help you?"}]')
            elif format_type == "conversation":
                st.code("Turn 1 (User): Hello, I need help\nTurn 2 (Assistant): How can I help you?\nTurn 3 (User): I have a question")
    
    # Current data stats
    st.subheader("📊 Dataset Statistics")
    if st.session_state.processor.raw_df is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_messages = len(st.session_state.processor.raw_df)
            st.metric("Total Messages", f"{total_messages:,}")
        
        with col2:
            if st.session_state.processor.conversation_id_col:
                unique_convs = st.session_state.processor.raw_df[st.session_state.processor.conversation_id_col].nunique()
                st.metric("Conversations", f"{unique_convs:,}")
                
                # Estimate windows
                avg_msgs_per_conv = total_messages / unique_convs
                est_windows_per_conv = max(1, int((avg_msgs_per_conv - window_size) / stride + 1))
                est_total_windows = unique_convs * est_windows_per_conv
                
                with col3:
                    st.metric("Estimated Windows", f"{est_total_windows:,}")
    
    # Process button
    if st.button("🚀 Process Context Windows", type="primary"):
        if st.session_state.processor.setup_window_parameters(
            window_size, overlap, format_type, include_system_msgs
        ):
            with st.spinner("Processing rolling context windows..."):
                if st.session_state.processor.process_rolling_windows():
                    st.session_state.processing_done = True
                    st.success("🎉 Processing completed!")
                    
                    # Show quick stats and preview
                    processed_df = st.session_state.processor.processed_df
                    total_windows = len(processed_df)
                    st.metric("Windows Created", f"{total_windows:,}")
                    
                    # Show output preview with correct columns
                    st.subheader("📋 Output Preview")
                    st.dataframe(processed_df.head(), use_container_width=True)
                    st.success(f"✅ Output contains exactly {len(processed_df.columns)} columns: {', '.join(processed_df.columns)}")

def analysis_page():
    st.markdown("<h2 class='section-header'>📊 Analysis & Results</h2>", unsafe_allow_html=True)
    
    if not st.session_state.processing_done:
        st.warning("⚠️ Please process context windows first.")
        return

    # Calculate statistics
    st.session_state.processor.analyze_window_statistics()
    stats = st.session_state.processor.window_stats

    # Overview metrics
    st.subheader("📈 Overview Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Windows", f"{stats['total_windows']:,}")
    with col2:
        st.metric("Unique Speakers", f"{stats['unique_speakers']:,}")
    with col3:
        avg_length = stats['context_length_stats']['mean']
        st.metric("Avg Context Length", f"{avg_length:.0f} chars")
    with col4:
        st.metric("Output Columns", "5")

    # Detailed statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📏 Context Length Stats")
        if not stats['context_length_stats'].empty:
            length_df = pd.DataFrame({
                'Statistic': stats['context_length_stats'].index,
                'Value': stats['context_length_stats'].values
            })
            st.dataframe(length_df, use_container_width=True)
    
    with col2:
        st.subheader("📝 Word Count Stats")
        if not stats['word_count_stats'].empty:
            word_df = pd.DataFrame({
                'Statistic': stats['word_count_stats'].index,
                'Value': stats['word_count_stats'].values
            })
            st.dataframe(word_df, use_container_width=True)
    
    with col3:
        st.subheader("🔄 Turn Count Stats")
        if not stats['turn_count_stats'].empty:
            turn_df = pd.DataFrame({
                'Statistic': stats['turn_count_stats'].index,
                'Value': stats['turn_count_stats'].values
            })
            st.dataframe(turn_df, use_container_width=True)

    # Show output format
    st.subheader("📋 Output Format Preview")
    processed_df = st.session_state.processor.processed_df
    st.dataframe(processed_df.head(), use_container_width=True)
    st.info(f"✅ Output contains exactly {len(processed_df.columns)} columns: {', '.join(processed_df.columns)}")

def sample_windows_page():
    st.markdown("<h2 class='section-header'>📝 Sample Context Windows</h2>", unsafe_allow_html=True)
    
    if not st.session_state.processing_done:
        st.warning("⚠️ Please process context windows first.")
        return

    processed_df = st.session_state.processor.processed_df
    
    if processed_df is None or processed_df.empty:
        st.error("❌ No processed data available")
        return
    
    # Show sample entries
    st.subheader("🔀 Random Samples")
    random_samples = processed_df.sample(min(10, len(processed_df)))
    
    for idx, (_, row) in enumerate(random_samples.iterrows(), 1):
        with st.expander(f"Sample {idx} - ID: {row['ID']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**ID:** {row['ID']}")
                st.write(f"**Turn:** {row['Turn']}")
                st.write(f"**Speaker:** {row['Speaker']}")
            with col2:
                st.write(f"**Statement:** {row['Statement'][:100]}...")
            
            st.write(f"**Context:**")
            st.text(row['Context'][:300] + "..." if len(str(row['Context'])) > 300 else str(row['Context']))

def search_page():
    st.markdown("<h2 class='section-header'>🔍 Search Context Windows</h2>", unsafe_allow_html=True)
    
    if not st.session_state.processing_done:
        st.warning("⚠️ Please process context windows first.")
        return

    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter search query:", placeholder="Search for specific text in context windows...")
    
    with col2:
        max_results = st.selectbox("Max results:", [10, 20, 50, 100], index=1)

    if query:
        matches = st.session_state.processor.search_windows(query, max_results)
        
        if not matches.empty:
            st.success(f"✅ Found {len(matches)} windows containing '{query}'")
            
            # Display results
            for idx, (_, row) in enumerate(matches.iterrows(), 1):
                with st.expander(f"Result {idx} - ID: {row['ID']}"):
                    # Show all 5 columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ID", row['ID'])
                        st.metric("Turn", row['Turn'])
                    with col2:
                        st.metric("Speaker", row['Speaker'])
                    with col3:
                        st.write("**Statement:**")
                        st.write(row['Statement'][:100] + "..." if len(row['Statement']) > 100 else row['Statement'])
                    
                    st.write("**Context:**")
                    # Highlight the query in the context
                    highlighted = re.sub(f'({re.escape(query)})', r'**\1**', str(row['Context']), flags=re.IGNORECASE)
                    st.markdown(highlighted[:500] + "..." if len(highlighted) > 500 else highlighted)
        else:
            st.warning("❌ No context windows found containing the query")

def export_page():
    st.markdown("<h2 class='section-header'>💾 Export Results</h2>", unsafe_allow_html=True)
    
    if not st.session_state.processing_done:
        st.warning("⚠️ Please process context windows first.")
        return

    processed_df = st.session_state.processor.processed_df
    
    if processed_df is None or processed_df.empty:
        st.error("❌ No processed data to export")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Complete Dataset")
        st.success(f"✅ {len(processed_df)} context windows available")
        st.info(f"Columns: {', '.join(processed_df.columns)}")
        
        # Complete dataset export
        csv_data = processed_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Dataset (CSV)",
            data=csv_data,
            file_name="rolling_context_windows.csv",
            mime="text/csv"
        )
        
        # Preview
        with st.expander("👀 Preview Data"):
            st.dataframe(processed_df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("📋 Export Verification")
        st.write("**Column verification:**")
        expected_columns = ['ID', 'Turn', 'Speaker', 'Context', 'Statement']
        actual_columns = list(processed_df.columns)
        
        for col in expected_columns:
            if col in actual_columns:
                st.write(f"✅ {col}")
            else:
                st.write(f"❌ {col} (missing)")
        
        # Sample export
        st.subheader("📝 Sample Export")
        sample_size = st.slider("Sample size:", 10, min(100, len(processed_df)), 20)
        sample_df = processed_df.head(sample_size)
        
        sample_csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample (CSV)",
            data=sample_csv,
            file_name=f"sample_context_windows_{sample_size}.csv",
            mime="text/csv"
        )
        
        # JSONL format for ML training
        st.subheader("🎯 ML Training Format")
        jsonl_data = []
        for _, row in processed_df.iterrows():
            jsonl_entry = {
                'id': row['ID'],
                'turn': row['Turn'],
                'speaker': row['Speaker'],
                'context': row['Context'],
                'statement': row['Statement']
            }
            jsonl_data.append(json.dumps(jsonl_entry, ensure_ascii=False))
        
        jsonl_content = '\n'.join(jsonl_data)
        st.download_button(
            label="📥 Download JSONL Format",
            data=jsonl_content,
            file_name="context_windows.jsonl",
            mime="application/json"
        )

def show_progress():
    """Show progress in sidebar"""
    st.sidebar.markdown("### 📋 Progress")
    
    steps = [
        ("📁 Data Upload", st.session_state.data_loaded),
        ("📋 Column Config", st.session_state.columns_configured),
        ("🔧 Window Processing", st.session_state.processing_done),
    ]
    
    for step, completed in steps:
        if completed:
            st.sidebar.success(f"✅ {step}")
        else:
            st.sidebar.info(f"⏳ {step}")

def show_data_stats():
    """Show data statistics in sidebar"""
    if st.session_state.data_loaded:
        processor = st.session_state.processor
        
        st.sidebar.markdown("### 📊 Data Info")
        
        if processor.raw_df is not None:
            st.sidebar.metric("Total Messages", len(processor.raw_df))
            
            if processor.conversation_id_col and processor.conversation_id_col in processor.raw_df.columns:
                unique_convs = processor.raw_df[processor.conversation_id_col].nunique()
                st.sidebar.metric("Conversations", unique_convs)
        
        if st.session_state.processing_done and processor.processed_df is not None:
            st.sidebar.metric("Context Windows", len(processor.processed_df))
            st.sidebar.metric("Output Columns", len(processor.processed_df.columns))

def main():
    st.markdown("<h1 class='main-header'>🔄 Rolling Context Window Processor</h1>", unsafe_allow_html=True)
    st.markdown("### Process conversation data into rolling context windows with specific output format")

    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    
    # Show progress and stats
    show_progress()
    show_data_stats()
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📁 Data Upload", "📋 Column Config", "🔧 Window Config", "📊 Analysis", "📝 Sample Windows", "🔍 Search", "💾 Export"]
    )

    # Add important note about output format
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Output Format")
    st.sidebar.info("CSV will contain ONLY these columns:\n• ID\n• Turn\n• Speaker\n• Context\n• Statement")

    # Add help section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ❓ Help")
    
    with st.sidebar.expander("📋 Supported Formats"):
        st.markdown("""
        **CSV Files:**
        - Standard conversation datasets
        - Must have message column
        
        **JSON Files:**
        - Conversation arrays
        - Nested conversation structures
        - Message objects with metadata
        
        **Excel Files:**
        - Spreadsheet conversation data
        - Multiple sheets supported
        """)
    
    with st.sidebar.expander("🔧 Window Parameters"):
        st.markdown("""
        **Window Size:** Number of messages per window
        - Larger = more context, fewer windows
        - Smaller = less context, more windows
        
        **Overlap:** Messages shared between windows
        - Higher = more continuity, more windows
        - Lower = less continuity, fewer windows
        
        **Stride:** Window advancement (size - overlap)
        - Determines how much windows advance
        """)
    
    with st.sidebar.expander("📄 Output Columns"):
        st.markdown("""
        **ID:** Unique identifier for each window
        **Turn:** Number of turns/messages in window
        **Speaker:** Speaker from first message
        **Context:** Formatted conversation context
        **Statement:** Last message in the window
        """)
    
    with st.sidebar.expander("💡 Use Cases"):
        st.markdown("""
        - **Chatbot Training:** Dialogue model datasets
        - **Context Learning:** Sliding window training
        - **Conversation Analysis:** Pattern discovery
        - **NLP Research:** Conversational AI datasets
        - **Fine-tuning:** Language model adaptation
        """)

    # Main page routing
    if page == "📁 Data Upload":
        data_upload_page()
    elif page == "📋 Column Config":
        column_configuration_page()
    elif page == "🔧 Window Config":
        window_configuration_page()
    elif page == "📊 Analysis":
        analysis_page()
    elif page == "📝 Sample Windows":
        sample_windows_page()
    elif page == "🔍 Search":
        search_page()
    elif page == "💾 Export":
        export_page()

# Run the app
if __name__ == "__main__":
    main()
