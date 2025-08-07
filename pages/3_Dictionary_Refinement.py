import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Keyword Dictionary Refinement Tool",
    page_icon="🔍",
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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .keyword-box {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem;
        display: inline-block;
    }
    .remove-keyword {
        background-color: #ffebee;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem;
        display: inline-block;
    }
    .new-keyword {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🔍 Keyword Dictionary Refinement Tool</h1>', unsafe_allow_html=True)

# Default keywords
DEFAULT_KEYWORDS = [
    "personalized", "custom", "tailored", "bespoke", "individual",
    "human", "experience", "connection", "care", "thoughtful", 
    "responsive", "concierge", "attentive", "dedicated", "unique",
    "special", "exclusive", "premium", "luxury", "curated",
    "handpicked", "artisan", "boutique", "intimate", "personal",
    "authentic", "genuine", "original", "distinctive", "exceptional",
    "remarkable", "outstanding", "extraordinary", "memorable", "unforgettable",
    "amazing", "incredible", "fantastic", "wonderful", "beautiful",
    "stunning", "gorgeous", "elegant", "sophisticated", "refined"
]

# Initialize session state
if 'posts_df' not in st.session_state:
    st.session_state.posts_df = None
if 'engagement_df' not in st.session_state:
    st.session_state.engagement_df = None
if 'current_keywords' not in st.session_state:
    st.session_state.current_keywords = DEFAULT_KEYWORDS.copy()
if 'keyword_stats' not in st.session_state:
    st.session_state.keyword_stats = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Sidebar for configuration
st.sidebar.header("🔧 Configuration")

# Data upload section
st.sidebar.subheader("📁 Data Upload")

# Posts data upload
posts_file = st.sidebar.file_uploader(
    "Upload Instagram Posts Data (CSV)",
    type=['csv'],
    help="CSV file with 'ID' and 'Statement' columns"
)

if posts_file:
    try:
        st.session_state.posts_df = pd.read_csv(posts_file)
        st.sidebar.success(f"✅ Posts loaded: {st.session_state.posts_df.shape[0]} rows")
        
        # Validate columns
        required_cols = ['ID', 'Statement']
        missing_cols = [col for col in required_cols if col not in st.session_state.posts_df.columns]
        if missing_cols:
            st.sidebar.error(f"❌ Missing columns: {missing_cols}")
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading posts: {str(e)}")

# Engagement data upload (optional)
engagement_file = st.sidebar.file_uploader(
    "Upload Engagement Data (CSV) - Optional",
    type=['csv'],
    help="CSV with 'shortcode', 'number_likes', 'number_comments' columns"
)

if engagement_file:
    try:
        st.session_state.engagement_df = pd.read_csv(engagement_file)
        st.sidebar.success(f"✅ Engagement loaded: {st.session_state.engagement_df.shape[0]} rows")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading engagement: {str(e)}")

# Analysis parameters
st.sidebar.subheader("⚙️ Analysis Settings")
min_word_count = st.sidebar.number_input("Minimum words per post", min_value=1, value=5)
min_frequency = st.sidebar.number_input("Min frequency to keep keyword", min_value=1, value=3)
min_posts = st.sidebar.number_input("Min posts to keep keyword", min_value=1, value=2)

# Helper functions
def count_keyword_matches(caption, keywords):
    """Count keyword matches in caption"""
    if pd.isna(caption):
        return 0, []
    
    text = str(caption).lower()
    words = re.findall(r'\b\w+\b', text)
    matched_keywords = [word for word in words if word in keywords]
    
    return len(matched_keywords), matched_keywords

def analyze_keyword_performance(posts_df, engagement_df, keywords, min_word_count):
    """Analyze individual keyword performance"""
    keyword_metrics = defaultdict(lambda: {
        'frequency': 0,
        'posts_used': set(),
        'total_engagement': 0,
        'avg_engagement': 0,
        'posts_with_engagement': []
    })
    
    for idx, row in posts_df.iterrows():
        if pd.isna(row['Statement']):
            continue
            
        text = str(row['Statement']).lower()
        words = re.findall(r'\b\w+\b', text)
        
        if len(words) < min_word_count:
            continue
            
        for keyword in keywords:
            if keyword in words:
                keyword_metrics[keyword]['frequency'] += 1
                keyword_metrics[keyword]['posts_used'].add(row['ID'])
                
                # Add engagement data if available
                if engagement_df is not None:
                    engagement_row = engagement_df[
                        engagement_df['shortcode'] == row['ID']
                    ]
                    if not engagement_row.empty:
                        likes = engagement_row['number_likes'].iloc[0]
                        comments = engagement_row['number_comments'].iloc[0]
                        total_eng = likes + comments
                        keyword_metrics[keyword]['total_engagement'] += total_eng
                        keyword_metrics[keyword]['posts_with_engagement'].append(total_eng)
    
    # Calculate averages
    for keyword in keyword_metrics:
        posts_count = len(keyword_metrics[keyword]['posts_used'])
        keyword_metrics[keyword]['posts_count'] = posts_count
        
        if keyword_metrics[keyword]['posts_with_engagement']:
            keyword_metrics[keyword]['avg_engagement'] = np.mean(
                keyword_metrics[keyword]['posts_with_engagement']
            )
        
        keyword_metrics[keyword]['posts_used'] = posts_count
    
    return dict(keyword_metrics)

def discover_new_keywords(posts_df, existing_keywords, top_n=20, min_length=4):
    """Discover potential new keywords"""
    all_words = []
    for statement in posts_df['Statement'].dropna():
        words = re.findall(r'\b\w+\b', str(statement).lower())
        all_words.extend([w for w in words if len(w) >= min_length])
    
    word_counts = Counter(all_words)
    
    # Stop words and existing keywords to exclude
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 
        'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
        'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'this', 'that', 'with',
        'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
        'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them',
        'well', 'will', 'your', 'about', 'would', 'there', 'could', 'other', 'after', 'first', 'never',
        'these', 'think', 'where', 'being', 'every', 'great', 'might', 'shall', 'still', 'those', 'under',
        'while', 'should', 'instagram', 'post', 'posts', 'follow', 'following', 'followers', 'like', 'likes'
    }
    
    existing_keywords_set = set(existing_keywords)
    
    potential_keywords = []
    for word, count in word_counts.most_common(top_n * 3):
        if (word not in existing_keywords_set and 
            word not in stop_words and 
            len(word) >= min_length and
            word.isalpha()):
            potential_keywords.append((word, count))
            
    return potential_keywords[:top_n]

# Main app content
if st.session_state.posts_df is not None:
    # Data overview
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Posts", len(st.session_state.posts_df))
    with col2:
        st.metric("Unique IDs", st.session_state.posts_df['ID'].nunique())
    with col3:
        missing_statements = st.session_state.posts_df['Statement'].isna().sum()
        st.metric("Missing Statements", missing_statements)
    with col4:
        engagement_available = "Yes" if st.session_state.engagement_df is not None else "No"
        st.metric("Engagement Data", engagement_available)
    
    # Preview data
    with st.expander("📋 Preview Data"):
        st.dataframe(st.session_state.posts_df.head(), use_container_width=True)

# Keyword management section
st.header("🎯 Keyword Management")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Current Keywords")
    
    # Display current keywords
    st.write(f"**Total Keywords:** {len(st.session_state.current_keywords)}")
    
    # Show keywords in a nice format
    keywords_text = ""
    for i, keyword in enumerate(st.session_state.current_keywords):
        keywords_text += f'<span class="keyword-box">{keyword}</span>'
        if (i + 1) % 6 == 0:  # New line every 6 keywords
            keywords_text += "<br>"
    
    st.markdown(keywords_text, unsafe_allow_html=True)

with col2:
    st.subheader("Modify Keywords")
    
    # Reset to default
    if st.button("🔄 Reset to Default"):
        st.session_state.current_keywords = DEFAULT_KEYWORDS.copy()
        st.success("Keywords reset to default list!")
        st.rerun()
    
    # Add new keyword
    new_keyword = st.text_input("Add keyword:")
    if st.button("➕ Add Keyword") and new_keyword:
        if new_keyword.lower() not in st.session_state.current_keywords:
            st.session_state.current_keywords.append(new_keyword.lower())
            st.success(f"Added: {new_keyword}")
            st.rerun()
        else:
            st.warning("Keyword already exists!")
    
    # Remove keyword
    if st.session_state.current_keywords:
        keyword_to_remove = st.selectbox("Remove keyword:", 
                                       [""] + st.session_state.current_keywords)
        if st.button("➖ Remove Keyword") and keyword_to_remove:
            st.session_state.current_keywords.remove(keyword_to_remove)
            st.success(f"Removed: {keyword_to_remove}")
            st.rerun()

# Bulk keyword input
with st.expander("📝 Bulk Keyword Input"):
    keywords_text_area = st.text_area(
        "Enter keywords (one per line):",
        value="\n".join(st.session_state.current_keywords),
        height=200
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Update Keywords"):
            new_keywords = [k.strip().lower() for k in keywords_text_area.split('\n') if k.strip()]
            st.session_state.current_keywords = new_keywords
            st.success(f"Updated to {len(new_keywords)} keywords!")
            st.rerun()
    
    with col2:
        # Upload keywords file
        keywords_file = st.file_uploader("Upload keywords file", type=['txt', 'csv'])
        if keywords_file:
            try:
                if keywords_file.name.endswith('.txt'):
                    keywords_content = keywords_file.read().decode('utf-8')
                    uploaded_keywords = [k.strip().lower() for k in keywords_content.split('\n') if k.strip()]
                else:
                    df_keywords = pd.read_csv(keywords_file)
                    uploaded_keywords = df_keywords.iloc[:, 0].str.lower().tolist()
                
                st.session_state.current_keywords = uploaded_keywords
                st.success(f"Loaded {len(uploaded_keywords)} keywords from file!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error loading keywords: {str(e)}")

# Analysis section
if st.session_state.posts_df is not None and st.session_state.current_keywords:
    st.header("🔍 Keyword Analysis")
    
    if st.button("🚀 Analyze Keywords", type="primary"):
        with st.spinner("Analyzing keyword performance..."):
            st.session_state.keyword_stats = analyze_keyword_performance(
                st.session_state.posts_df,
                st.session_state.engagement_df,
                st.session_state.current_keywords,
                min_word_count
            )
            st.session_state.analysis_complete = True
            st.success("Analysis completed!")

# Results section
if st.session_state.analysis_complete and st.session_state.keyword_stats:
    st.header("📈 Analysis Results")
    
    # Convert to DataFrame for easier handling
    stats_df = pd.DataFrame.from_dict(st.session_state.keyword_stats, orient='index')
    stats_df = stats_df.sort_values('frequency', ascending=False)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Keywords Analyzed", len(stats_df))
    with col2:
        used_keywords = len(stats_df[stats_df['frequency'] > 0])
        st.metric("Keywords Found", used_keywords)
    with col3:
        avg_frequency = stats_df['frequency'].mean()
        st.metric("Avg Frequency", f"{avg_frequency:.1f}")
    with col4:
        unused_count = len(st.session_state.current_keywords) - len(stats_df)
        st.metric("Unused Keywords", unused_count)
    
    # Performance insights
    st.subheader("🏆 Top Performing Keywords")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Most Frequent Keywords:**")
        top_freq = stats_df.head(10)
        for keyword, row in top_freq.iterrows():
            st.write(f"• **{keyword}**: {row['frequency']} times in {row['posts_count']} posts")
    
    with col2:
        if st.session_state.engagement_df is not None and 'avg_engagement' in stats_df.columns:
            st.write("**Highest Engagement Keywords:**")
            stats_df_eng = stats_df[stats_df['avg_engagement'] > 0].sort_values('avg_engagement', ascending=False)
            top_eng = stats_df_eng.head(10)
            for keyword, row in top_eng.iterrows():
                st.write(f"• **{keyword}**: {row['avg_engagement']:.1f} avg engagement")
        else:
            st.info("Upload engagement data to see engagement metrics")
    
    # Visualizations
    st.subheader("📊 Performance Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Frequency Distribution**")
        freq_data = pd.DataFrame({
            'Frequency': stats_df['frequency']
        })
        st.bar_chart(freq_data)
    
    with col2:
        st.write("**Top 15 Keywords by Frequency**")
        top_15 = stats_df.head(15)
        chart_data = pd.DataFrame({
            'Keyword': top_15.index,
            'Frequency': top_15['frequency'].values
        })
        st.bar_chart(chart_data.set_index('Keyword'))
    
    # Refinement suggestions
    st.subheader("💡 Refinement Suggestions")
    
    # Keywords to keep
    keep_keywords = stats_df[
        (stats_df['frequency'] >= min_frequency) & 
        (stats_df['posts_count'] >= min_posts)
    ].index.tolist()
    
    # Keywords to remove
    remove_keywords = stats_df[
        (stats_df['frequency'] < min_frequency) | 
        (stats_df['posts_count'] < min_posts)
    ].index.tolist()
    
    # Unused keywords
    unused_keywords = [kw for kw in st.session_state.current_keywords if kw not in st.session_state.keyword_stats]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"**✅ Keep ({len(keep_keywords)} keywords)**")
        st.write("Good performance, recommend keeping:")
        for kw in keep_keywords[:10]:
            freq = stats_df.loc[kw, 'frequency']
            posts = stats_df.loc[kw, 'posts_count']
            st.write(f"• {kw} (freq: {freq}, posts: {posts})")
        if len(keep_keywords) > 10:
            st.write(f"... and {len(keep_keywords) - 10} more")
    
    with col2:
        st.warning(f"**⚠️ Consider Removing ({len(remove_keywords)} keywords)**")
        st.write(f"Low frequency (< {min_frequency}) or few posts (< {min_posts}):")
        for kw in remove_keywords[:10]:
            freq = stats_df.loc[kw, 'frequency']
            posts = stats_df.loc[kw, 'posts_count']
            st.write(f"• {kw} (freq: {freq}, posts: {posts})")
        if len(remove_keywords) > 10:
            st.write(f"... and {len(remove_keywords) - 10} more")
    
    with col3:
        st.error(f"**❌ Unused ({len(unused_keywords)} keywords)**")
        st.write("Never appear in your data:")
        for kw in unused_keywords[:10]:
            st.write(f"• {kw}")
        if len(unused_keywords) > 10:
            st.write(f"... and {len(unused_keywords) - 10} more")
    
    # Interactive refinement
    st.subheader("🔧 Interactive Refinement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Apply Keep Recommendations"):
            st.session_state.current_keywords = keep_keywords
            st.success(f"Updated to {len(keep_keywords)} recommended keywords!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Remove Poor Performers"):
            st.session_state.current_keywords = [kw for kw in st.session_state.current_keywords 
                                               if kw not in remove_keywords + unused_keywords]
            st.success(f"Removed {len(remove_keywords) + len(unused_keywords)} poor performers!")
            st.rerun()
    
    # Discover new keywords
    st.subheader("🆕 Discover New Keywords")
    
    if st.button("🔍 Find Potential Keywords"):
        with st.spinner("Discovering new keywords..."):
            potential_keywords = discover_new_keywords(
                st.session_state.posts_df, 
                st.session_state.current_keywords, 
                top_n=20
            )
            
            if potential_keywords:
                st.write("**Potential new keywords found in your data:**")
                
                # Display in columns
                cols = st.columns(3)
                for i, (word, count) in enumerate(potential_keywords):
                    with cols[i % 3]:
                        col_inner1, col_inner2 = st.columns([3, 1])
                        with col_inner1:
                            st.write(f"**{word}** ({count} times)")
                        with col_inner2:
                            if st.button("➕", key=f"add_{word}"):
                                if word not in st.session_state.current_keywords:
                                    st.session_state.current_keywords.append(word)
                                    st.success(f"Added {word}!")
                                    st.rerun()
            else:
                st.info("No new keywords discovered with current settings.")

# Export section
if st.session_state.current_keywords:
    st.header("💾 Export Keywords")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export current keywords
        keywords_text = "\n".join(sorted(st.session_state.current_keywords))
        st.download_button(
            label="📥 Download Current Keywords",
            data=keywords_text,
            file_name=f"keywords_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        # Export analysis results
        if st.session_state.analysis_complete and st.session_state.keyword_stats:
            # Create analysis report
            stats_df = pd.DataFrame.from_dict(st.session_state.keyword_stats, orient='index')
            csv_buffer = io.StringIO()
            stats_df.to_csv(csv_buffer)
            
            st.download_button(
                label="📊 Download Analysis Report",
                data=csv_buffer.getvalue(),
                file_name=f"keyword_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Help section
with st.expander("❓ How to Use This Tool"):
    st.markdown("""
    ### 📋 Step-by-Step Guide:
    
    1. **Upload Data**: Upload your Instagram posts CSV file (required: 'ID' and 'Statement' columns)
    2. **Optional**: Upload engagement data CSV for engagement analysis
    3. **Manage Keywords**: 
       - View current keyword list
       - Add/remove individual keywords
       - Bulk edit using text area or file upload
    4. **Analyze**: Click "Analyze Keywords" to run performance analysis
    5. **Review Results**: 
       - See which keywords perform well vs poorly
       - View frequency and engagement metrics
       - Get refinement recommendations
    6. **Discover**: Find new potential keywords from your data
    7. **Refine**: Apply recommendations or manually adjust keywords
    8. **Export**: Download refined keyword list and analysis report
    
    ### ⚙️ Settings:
    - **Min words per post**: Posts with fewer words are ignored
    - **Min frequency**: Keywords used less than this are flagged for removal
    - **Min posts**: Keywords in fewer posts are flagged for removal
    
    ### 💡 Tips:
    - Start with default keywords and refine based on your data
    - Higher frequency doesn't always mean better - consider engagement too
    - Regularly update your keyword list as your content evolves
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔍 Keyword Dictionary Refinement Tool</p>
    <p>Optimize your personalized language keywords based on actual data performance</p>
</div>
""", unsafe_allow_html=True)
