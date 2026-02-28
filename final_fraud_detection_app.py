import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import joblib
from datetime import datetime
import io

# Configure page settings
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2193b0 0%, #6dd5ed 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

# Sidebar navigation
st.sidebar.markdown("# 🔒 Fraud Detection System")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "Navigate to:",
    [
        "👥 About Project",
        "🏠 Home",
        "📤 Data Upload",
        "📊 Data Visualization",
        "🔧 Data Preprocessing",
        "🤖 Model Training",
        "📈 Model Evaluation",
        "🔮 Fraud Prediction"
    ]
)


# Helper functions
@st.cache_data
def load_data(file):
    """Load dataset from uploaded file"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def create_fraud_distribution_plot(df):
    """Create fraud distribution visualization"""
    fraud_counts = df['is_fraud'].value_counts()

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "xy"}]])

    # Pie chart
    fig.add_trace(go.Pie(labels=['Not Fraud', 'Fraud'],
                         values=fraud_counts.values,
                         name="Fraud Distribution",
                         marker_colors=['#2ecc71', '#e74c3c']), 1, 1)

    # Bar chart
    fig.add_trace(go.Bar(x=['Not Fraud', 'Fraud'],
                         y=fraud_counts.values,
                         marker_color=['#2ecc71', '#e74c3c'],
                         name="Count"), 1, 2)

    fig.update_layout(title_text="Fraud vs Non-Fraud Distribution", showlegend=True)
    return fig


def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical features"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()

    fig = px.imshow(corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    title="Correlation Heatmap of Numerical Features")
    return fig


# Page content based on navigation
if page == "👥 About Project":
    st.markdown('<h1 class="main-header">👥 About This Project</h1>', unsafe_allow_html=True)

    # Course and University Information
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: white; text-align: center;">UNIVERSITY OF GHANA</h2>
        <h3 style="color: white; text-align: center;">Department of Operations and Management Information Systems</h3>
        <p style="color: white; text-align: center; font-size: 1.2rem;">
            <strong>Course:</strong> OMIS 304 - Machine Learning 1 (3 Credits)<br>
            <strong>Academic Year:</strong> Second Semester 2022/2023<br>
            <strong>Assignment:</strong> Class Project on Machine Learning Applications
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Project Background
    st.markdown('<h2 class="sub-header">📋 Project Background & Purpose</h2>', unsafe_allow_html=True)

    st.markdown("""
    **Credit card fraud** is a serious and growing challenge for financial institutions and consumers alike. 
    Fraudulent transactions can result in significant financial losses, undermine trust in financial 
    systems, and burden fraud investigation teams. Detecting such transactions in real time is difficult 
    due to the highly imbalanced nature of fraud datasets and the evolving tactics used by fraudsters.

    The purpose of this project is to develop a **multi-page interactive web application** using Streamlit 
    that classifies credit card transactions as **"Fraud"** or **"Not Fraud"** based on transaction-level data. 
    This application leverages machine learning classification techniques to analyze historical 
    transaction patterns and identify suspicious behavior. Through a clean and user-friendly 
    dashboard, the system supports fraud monitoring, feature analysis, and predictive capabilities.
    """)

    # Group Information
    st.markdown('<h2 class="sub-header">👥 Group 10 Members</h2>', unsafe_allow_html=True)

    group_members = [
        {"Name": "Abdul Samed Al-Hassan Bin Abubakr", "ID": "11232733"},
        {"Name": "John Adu Acheampong", "ID": "11012084"},
        {"Name": "Zanu Christopher", "ID": "11179138"},
        {"Name": "Clifford Mante Yeboah", "ID": "11235040"},
        {"Name": "Stephanie Sakyiwaa Ayeh", "ID": "11257061"}
    ]

    # Display members in cards
    for i in range(0, len(group_members), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(group_members):
                member = group_members[i + j]
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{member['Name']}</h4>
                        <p><strong>Student ID:</strong> {member['ID']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # Project Requirements
    st.markdown('<h2 class="sub-header">🎯 Project Requirements</h2>', unsafe_allow_html=True)

    requirements = [
        {
            "title": "🔄 Data Import and Overview",
            "description": "Interface for uploading/loading datasets with summary statistics and basic exploratory visualizations including histograms and correlation matrices."
        },
        {
            "title": "🔧 Data Preprocessing",
            "description": "Handle missing values, apply encoding techniques for categorical variables (LabelEncoding/One-Hot Encoding), and standardize/normalize numerical variables."
        },
        {
            "title": "🤖 Model Training",
            "description": "Implement and train multiple classification models including K-Nearest Neighbors, Support Vector Machines, and Random Forest."
        },
        {
            "title": "📊 Model Evaluation",
            "description": "Split dataset using Holdout Validation (Train/Test Split) and report key metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC Curves."
        },
        {
            "title": "🔮 Prediction Page",
            "description": "User-friendly form for entering transaction features with trained model predictions and associated probability/confidence scores."
        },
        {
            "title": "📈 Interpretation & Conclusions",
            "description": "Summary interpreting most predictive features, model performance comparisons, and visual aids like feature importance plots."
        }
    ]

    for req in requirements:
        with st.expander(req["title"]):
            st.write(req["description"])

    # Dataset Information
    st.markdown('<h2 class="sub-header">📊 Dataset Information</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-card">
            <h4>📁 Dataset Source</h4>
            <p><strong>Platform:</strong> Kaggle</p>
            <p><strong>Dataset:</strong> Fraud Detection</p>
            <p><strong>URL:</strong> <a href="https://www.kaggle.com/datasets/kartik2112/fraud-detection" target="_blank" style="color: white;">kaggle.com/datasets/kartik2112/fraud-detection</a></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-card">
            <h4>🔍 Key Features</h4>
            <p>• Transaction amounts and merchant details</p>
            <p>• Geographic location data</p>
            <p>• Customer demographics</p>
            <p>• Temporal transaction patterns</p>
            <p>• Binary fraud classification target</p>
        </div>
        """, unsafe_allow_html=True)

    # Technical Specifications
    st.markdown('<h2 class="sub-header">⚙️ Technical Specifications</h2>', unsafe_allow_html=True)

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("""
        **🐍 Programming Language**
        - Python 3.8+

        **📚 Core Libraries**
        - Streamlit (Web App)
        - Pandas (Data Manipulation)
        - NumPy (Numerical Computing)
        """)

    with tech_col2:
        st.markdown("""
        **🤖 Machine Learning**
        - Scikit-learn
        - K-Nearest Neighbors
        - Support Vector Machine
        - Random Forest
        """)

    with tech_col3:
        st.markdown("""
        **📊 Visualization**
        - Plotly Express
        - Plotly Graph Objects  
        - Matplotlib
        - Seaborn
        """)

    # Project Deliverables
    st.markdown('<h2 class="sub-header">📦 Expected Deliverables</h2>', unsafe_allow_html=True)

    deliverables = [
        "✅ Fully working Streamlit application with all required features",
        "✅ Well-commented Python script with clear documentation",
        "✅ Comprehensive written project report",
        "✅ Public web link to deployed application (per student)",
        "✅ Team presentation with complete code walkthrough"
    ]

    for deliverable in deliverables:
        st.markdown(f"- {deliverable}")

    # Assignment Timeline
    st.markdown('<h2 class="sub-header">📅 Project Timeline</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="success-card">
            <h4>📅 Key Dates</h4>
            <p><strong>Deadline:</strong> August 15th, 2025</p>
            <p><strong>Submission:</strong> Via SAKAI Platform</p>
            <p><strong>Presentation:</strong> In-class demonstration</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Success Criteria</h4>
            <p>• Complete application functionality</p>
            <p>• Proper code documentation</p>
            <p>• Successful deployment</p>
            <p>• Effective team presentation</p>
        </div>
        """, unsafe_allow_html=True)

    # Innovation Note
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: white; text-align: center;">💡 Innovation Encouraged</h3>
        <p style="color: white; text-align: center; font-size: 1.1rem;">
            Innovative projects will attract additional marks. This application goes beyond basic requirements 
            with advanced visualizations, comprehensive analysis, and enhanced user experience.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Hint
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>👉 <strong>Ready to explore?</strong> Navigate to the <strong>Home</strong> page to begin using the fraud detection system!</p>
    </div>
    """, unsafe_allow_html=True)
elif page == "🏠 Home":

    st.markdown('<h1 class="main-header">🔒 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)

    # Hero section with gradient background
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: white; text-align: center;">Protecting Financial Transactions with AI</h2>
        <p style="color: white; text-align: center; font-size: 1.2rem;">
            Advanced machine learning system for real-time fraud detection and prevention
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 High Accuracy</h3>
            <p>Multiple ML algorithms for optimal detection rates</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time</h3>
            <p>Instant fraud prediction and alerts</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analytics</h3>
            <p>Comprehensive data visualization and insights</p>
        </div>
        """, unsafe_allow_html=True)

    # System overview
    st.markdown('<h2 class="sub-header">System Features</h2>', unsafe_allow_html=True)

    features = [
        "📤 **Data Upload**: Support for CSV and Excel files",
        "📊 **Data Visualization**: Interactive charts and graphs",
        "🔧 **Data Preprocessing**: Automated data cleaning and preparation",
        "🤖 **Model Training**: K-NN, SVM, and Random Forest algorithms",
        "📈 **Model Evaluation**: Comprehensive performance metrics",
        "🔮 **Fraud Prediction**: Real-time transaction classification"
    ]

    for feature in features:
        st.markdown(f"- {feature}")

    # Instructions
    st.markdown('<h2 class="sub-header">Getting Started</h2>', unsafe_allow_html=True)
    st.markdown("""
    1. **Upload your dataset** using the Data Upload page
    2. **Explore your data** with interactive visualizations
    3. **Preprocess the data** to prepare it for machine learning
    4. **Train multiple models** to find the best performer
    5. **Evaluate model performance** with detailed metrics
    6. **Make predictions** on new transactions
    """)

elif page == "📤 Data Upload":
    st.markdown('<h1 class="main-header">📤 Data Upload</h1>', unsafe_allow_html=True)

    # File upload section
    st.markdown('<h2 class="sub-header">Upload Your Dataset</h2>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx'],
        help="Upload your credit card transaction dataset"
    )

    if uploaded_file is not None:
        # Load the data
        with st.spinner('Loading dataset...'):
            data = load_data(uploaded_file)

        if data is not None:
            st.session_state.data = data

            st.markdown("""
            <div class="success-card">
                <h3>✅ Dataset Loaded Successfully!</h3>
                <p>Your data has been loaded and is ready for analysis.</p>
            </div>
            """, unsafe_allow_html=True)

            # Display basic information about the dataset
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Rows", f"{data.shape[0]:,}")
            with col2:
                st.metric("Total Columns", data.shape[1])
            with col3:
                fraud_count = data['is_fraud'].sum() if 'is_fraud' in data.columns else 0
                st.metric("Fraud Cases", f"{fraud_count:,}")
            with col4:
                fraud_rate = (fraud_count / len(data) * 100) if len(data) > 0 else 0
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

            # Display sample data
            st.markdown('<h3 class="sub-header">Data Preview</h3>', unsafe_allow_html=True)
            st.dataframe(data.head(10), use_container_width=True)

            # Display column information
            st.markdown('<h3 class="sub-header">Column Information</h3>', unsafe_allow_html=True)

            col_info = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes,
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum(),
                'Unique Values': data.nunique()
            })

            st.dataframe(col_info, use_container_width=True)

            # Data quality summary
            st.markdown('<h3 class="sub-header">Data Quality Summary</h3>', unsafe_allow_html=True)

            total_nulls = data.isnull().sum().sum()
            completeness = ((data.size - total_nulls) / data.size) * 100

            quality_col1, quality_col2 = st.columns(2)

            with quality_col1:
                st.metric("Data Completeness", f"{completeness:.2f}%")
                st.metric("Total Missing Values", f"{total_nulls:,}")

            with quality_col2:
                duplicates = data.duplicated().sum()
                st.metric("Duplicate Rows", f"{duplicates:,}")
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    else:
        st.markdown("""
        <div class="warning-card">
            <h3>📁 No File Selected</h3>
            <p>Please upload a CSV or Excel file to begin the analysis.</p>
            <p><strong>Expected columns:</strong> merchant, category, amount, street, city, state, zip, 
            latitude, longitude, city_population, job, unix_time, merchant_latitude, 
            merchant_longitude, is_fraud, transaction_date, transaction_time</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 Data Visualization":
    st.markdown('<h1 class="main-header">📊 Data Visualization</h1>', unsafe_allow_html=True)

    if st.session_state.data is not None:
        data = st.session_state.data

        # Fraud Distribution
        st.markdown('<h2 class="sub-header">Fraud Distribution Analysis</h2>', unsafe_allow_html=True)

        if 'is_fraud' in data.columns:
            fig_fraud_dist = create_fraud_distribution_plot(data)
            st.plotly_chart(fig_fraud_dist, use_container_width=True)

            # Fraud statistics
            fraud_stats_col1, fraud_stats_col2, fraud_stats_col3 = st.columns(3)

            with fraud_stats_col1:
                total_transactions = len(data)
                st.metric("Total Transactions", f"{total_transactions:,}")

            with fraud_stats_col2:
                fraud_transactions = data['is_fraud'].sum()
                st.metric("Fraudulent Transactions", f"{fraud_transactions:,}")

            with fraud_stats_col3:
                fraud_percentage = (fraud_transactions / total_transactions) * 100
                st.metric("Fraud Percentage", f"{fraud_percentage:.3f}%")

        # Transaction Amount Analysis
        if 'amount' in data.columns:
            st.markdown('<h2 class="sub-header">Transaction Amount Analysis</h2>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # Amount distribution by fraud status
                fig_amount = px.box(data, x='is_fraud', y='amount',
                                    title="Transaction Amount Distribution by Fraud Status")
                st.plotly_chart(fig_amount, use_container_width=True)

            with col2:
                # Amount histogram
                fig_hist = px.histogram(data, x='amount', nbins=50,
                                        title="Transaction Amount Distribution",
                                        color='is_fraud' if 'is_fraud' in data.columns else None)
                st.plotly_chart(fig_hist, use_container_width=True)

            # New visualizations for amount analysis
            st.markdown("### 💰 Advanced Amount Analysis")

            col3, col4 = st.columns(2)

            with col3:
                # Amount statistics by fraud type
                amount_stats = data.groupby('is_fraud')['amount'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
                amount_stats.index = ['Legitimate', 'Fraudulent']

                fig_amount_stats = go.Figure()

                metrics = ['mean', 'median', 'std']
                colors = ['#3498db', '#e74c3c', '#f39c12']

                for i, metric in enumerate(metrics):
                    fig_amount_stats.add_trace(go.Bar(
                        name=metric.capitalize(),
                        x=amount_stats.index,
                        y=amount_stats[metric],
                        marker_color=colors[i]
                    ))

                fig_amount_stats.update_layout(
                    title="Amount Statistics by Transaction Type",
                    barmode='group'
                )
                st.plotly_chart(fig_amount_stats, use_container_width=True)

            with col4:
                # Amount ranges analysis
                data['amount_range'] = pd.cut(data['amount'],
                                              bins=[0, 50, 100, 500, 1000, float('inf')],
                                              labels=['$0-50', '$50-100', '$100-500', '$500-1000', '$1000+'])

                amount_range_fraud = data.groupby('amount_range')['is_fraud'].agg(['count', 'sum']).reset_index()
                amount_range_fraud['fraud_rate'] = (amount_range_fraud['sum'] / amount_range_fraud['count']) * 100

                fig_amount_range = px.bar(amount_range_fraud,
                                          x='amount_range',
                                          y='fraud_rate',
                                          title="Fraud Rate by Amount Range",
                                          color='fraud_rate',
                                          color_continuous_scale='Reds')
                st.plotly_chart(fig_amount_range, use_container_width=True)

        # Category Analysis
        if 'category' in data.columns:
            st.markdown('<h2 class="sub-header">Transaction Category Analysis</h2>', unsafe_allow_html=True)

            # Category fraud distribution
            category_fraud = data.groupby('category')['is_fraud'].agg(['count', 'sum']).reset_index()
            category_fraud['fraud_rate'] = (category_fraud['sum'] / category_fraud['count']) * 100
            category_fraud = category_fraud.sort_values('fraud_rate', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                fig_category = px.bar(category_fraud, x='category', y='fraud_rate',
                                      title="Fraud Rate by Transaction Category",
                                      color='fraud_rate',
                                      color_continuous_scale='Reds')
                fig_category.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_category, use_container_width=True)

            with col2:
                # Category volume vs fraud rate scatter
                fig_volume_fraud = px.scatter(category_fraud,
                                              x='count',
                                              y='fraud_rate',
                                              size='sum',
                                              hover_name='category',
                                              title="Transaction Volume vs Fraud Rate by Category",
                                              labels={'count': 'Total Transactions', 'fraud_rate': 'Fraud Rate (%)'})
                st.plotly_chart(fig_volume_fraud, use_container_width=True)

            # New category visualizations
            st.markdown("### 🏪 Category Deep Dive")

            col3, col4 = st.columns(2)

            with col3:
                # Top risky categories
                top_risky = category_fraud.nlargest(10, 'fraud_rate')
                fig_risky = px.bar(top_risky,
                                   x='fraud_rate',
                                   y='category',
                                   orientation='h',
                                   title="Top 10 Riskiest Categories",
                                   color='fraud_rate',
                                   color_continuous_scale='Reds')
                fig_risky.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_risky, use_container_width=True)

            with col4:
                # Category transaction amounts
                if 'amount' in data.columns:
                    category_amount = data.groupby('category')['amount'].mean().reset_index()
                    category_amount = category_amount.sort_values('amount', ascending=False).head(15)

                    fig_cat_amount = px.bar(category_amount,
                                            x='category',
                                            y='amount',
                                            title="Average Transaction Amount by Category",
                                            color='amount',
                                            color_continuous_scale='Blues')
                    fig_cat_amount.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_cat_amount, use_container_width=True)

        # Geographic Analysis
        if all(col in data.columns for col in ['latitude', 'longitude', 'is_fraud']):
            st.markdown('<h2 class="sub-header">Geographic Distribution</h2>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                # Sample data for performance (show max 1000 points)
                sample_data = data.sample(min(1000, len(data)))

                fig_geo = px.scatter_mapbox(sample_data,
                                            lat='latitude', lon='longitude',
                                            color='is_fraud',
                                            size='amount' if 'amount' in data.columns else None,
                                            hover_data=['city',
                                                        'state'] if 'city' in data.columns and 'state' in data.columns else None,
                                            mapbox_style="open-street-map",
                                            title="Geographic Distribution of Transactions")
                fig_geo.update_layout(height=500)
                st.plotly_chart(fig_geo, use_container_width=True)

            with col2:
                # State-wise fraud analysis
                if 'state' in data.columns:
                    state_fraud = data.groupby('state')['is_fraud'].agg(['count', 'sum']).reset_index()
                    state_fraud['fraud_rate'] = (state_fraud['sum'] / state_fraud['count']) * 100
                    state_fraud = state_fraud.sort_values('fraud_rate', ascending=False).head(15)

                    fig_state = px.bar(state_fraud,
                                       x='state',
                                       y='fraud_rate',
                                       title="Top 15 States by Fraud Rate",
                                       color='fraud_rate',
                                       color_continuous_scale='Reds')
                    fig_state.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_state, use_container_width=True)

        # New geographic visualizations
        if all(col in data.columns for col in ['city', 'state']):
            st.markdown("### 🌍 Geographic Deep Dive")

            col1, col2 = st.columns(2)

            with col1:
                # City population vs fraud rate
                if 'city_population' in data.columns:
                    city_analysis = data.groupby('city').agg({
                        'is_fraud': ['count', 'sum'],
                        'city_population': 'first'
                    }).reset_index()
                    city_analysis.columns = ['city', 'total_trans', 'fraud_trans', 'population']
                    city_analysis['fraud_rate'] = (city_analysis['fraud_trans'] / city_analysis['total_trans']) * 100
                    city_analysis = city_analysis[
                        city_analysis['total_trans'] >= 10]  # Filter cities with at least 10 transactions

                    fig_pop_fraud = px.scatter(city_analysis,
                                               x='population',
                                               y='fraud_rate',
                                               size='total_trans',
                                               hover_name='city',
                                               title="City Population vs Fraud Rate",
                                               labels={'population': 'City Population', 'fraud_rate': 'Fraud Rate (%)'})
                    st.plotly_chart(fig_pop_fraud, use_container_width=True)

            with col2:
                # Distance from merchant analysis
                if all(col in data.columns for col in ['merchant_latitude', 'merchant_longitude']):
                    # Calculate distance between customer and merchant
                    def haversine_distance(lat1, lon1, lat2, lon2):
                        R = 6371  # Earth's radius in km
                        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                        return 2 * R * np.arcsin(np.sqrt(a))


                    sample_geo = data.sample(min(5000, len(data)))  # Limit for performance
                    sample_geo['distance'] = haversine_distance(
                        sample_geo['latitude'], sample_geo['longitude'],
                        sample_geo['merchant_latitude'], sample_geo['merchant_longitude']
                    )

                    # Create distance ranges
                    sample_geo['distance_range'] = pd.cut(sample_geo['distance'],
                                                          bins=[0, 1, 5, 10, 50, float('inf')],
                                                          labels=['<1km', '1-5km', '5-10km', '10-50km', '>50km'])

                    distance_fraud = sample_geo.groupby('distance_range')['is_fraud'].agg(
                        ['count', 'sum']).reset_index()
                    distance_fraud['fraud_rate'] = (distance_fraud['sum'] / distance_fraud['count']) * 100

                    fig_distance = px.bar(distance_fraud,
                                          x='distance_range',
                                          y='fraud_rate',
                                          title="Fraud Rate by Customer-Merchant Distance",
                                          color='fraud_rate',
                                          color_continuous_scale='Reds')
                    st.plotly_chart(fig_distance, use_container_width=True)

        # Job Analysis
        if 'job' in data.columns:
            st.markdown('<h2 class="sub-header">Job Category Analysis</h2>', unsafe_allow_html=True)

            job_fraud = data.groupby('job')['is_fraud'].agg(['count', 'sum']).reset_index()
            job_fraud['fraud_rate'] = (job_fraud['sum'] / job_fraud['count']) * 100
            job_fraud = job_fraud[job_fraud['count'] >= 10]  # Filter jobs with at least 10 transactions

            col1, col2 = st.columns(2)

            with col1:
                top_risky_jobs = job_fraud.nlargest(15, 'fraud_rate')
                fig_risky_jobs = px.bar(top_risky_jobs,
                                        x='fraud_rate',
                                        y='job',
                                        orientation='h',
                                        title="Top 15 Riskiest Job Categories",
                                        color='fraud_rate',
                                        color_continuous_scale='Reds')
                fig_risky_jobs.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_risky_jobs, use_container_width=True)

            with col2:
                # Job volume analysis
                top_volume_jobs = job_fraud.nlargest(15, 'count')
                fig_volume_jobs = px.bar(top_volume_jobs,
                                         x='job',
                                         y='count',
                                         title="Top 15 Job Categories by Transaction Volume",
                                         color='fraud_rate',
                                         color_continuous_scale='Blues')
                fig_volume_jobs.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_volume_jobs, use_container_width=True)

        # Correlation Analysis
        st.markdown('<h2 class="sub-header">Feature Correlation Analysis</h2>', unsafe_allow_html=True)

        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            col1, col2 = st.columns(2)

            with col1:
                fig_corr = create_correlation_heatmap(data)
                st.plotly_chart(fig_corr, use_container_width=True)

            with col2:
                # Fraud correlation specifically
                if 'is_fraud' in numerical_cols:
                    fraud_corr = data[numerical_cols].corr()['is_fraud'].sort_values(key=abs, ascending=False)[
                                 1:]  # Exclude self-correlation
                    fraud_corr_df = pd.DataFrame({
                        'Feature': fraud_corr.index,
                        'Correlation': fraud_corr.values
                    })

                    fig_fraud_corr = px.bar(fraud_corr_df,
                                            x='Correlation',
                                            y='Feature',
                                            orientation='h',
                                            title="Feature Correlation with Fraud",
                                            color='Correlation',
                                            color_continuous_scale='RdBu',
                                            color_continuous_midpoint=0)
                    fig_fraud_corr.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_fraud_corr, use_container_width=True)

        # Time-based Analysis (if date columns exist)
        if 'transaction_date' in data.columns:
            st.markdown('<h2 class="sub-header">Time-based Analysis</h2>', unsafe_allow_html=True)

            # Convert to datetime if it's not already
            data['transaction_date'] = pd.to_datetime(data['transaction_date'])

            col1, col2 = st.columns(2)

            with col1:
                # Daily fraud trends
                daily_fraud = data.groupby(data['transaction_date'].dt.date)['is_fraud'].agg(
                    ['count', 'sum']).reset_index()
                daily_fraud['fraud_rate'] = (daily_fraud['sum'] / daily_fraud['count']) * 100

                fig_time = px.line(daily_fraud, x='transaction_date', y='fraud_rate',
                                   title="Daily Fraud Rate Trends")
                st.plotly_chart(fig_time, use_container_width=True)

            with col2:
                # Weekly pattern
                data['day_of_week'] = data['transaction_date'].dt.day_name()
                weekly_fraud = data.groupby('day_of_week')['is_fraud'].agg(['count', 'sum']).reset_index()
                weekly_fraud['fraud_rate'] = (weekly_fraud['sum'] / weekly_fraud['count']) * 100

                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_fraud['day_of_week'] = pd.Categorical(weekly_fraud['day_of_week'], categories=day_order,
                                                             ordered=True)
                weekly_fraud = weekly_fraud.sort_values('day_of_week')

                fig_weekly = px.bar(weekly_fraud,
                                    x='day_of_week',
                                    y='fraud_rate',
                                    title="Fraud Rate by Day of Week",
                                    color='fraud_rate',
                                    color_continuous_scale='Reds')
                st.plotly_chart(fig_weekly, use_container_width=True)

        # Time analysis if time column exists
        if 'transaction_time' in data.columns:
            st.markdown("### ⏰ Hourly Pattern Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Extract hour from transaction time
                data['hour'] = pd.to_datetime(data['transaction_time']).dt.hour
                hourly_fraud = data.groupby('hour')['is_fraud'].agg(['count', 'sum']).reset_index()
                hourly_fraud['fraud_rate'] = (hourly_fraud['sum'] / hourly_fraud['count']) * 100

                fig_hourly = px.line(hourly_fraud,
                                     x='hour',
                                     y='fraud_rate',
                                     title="Fraud Rate by Hour of Day",
                                     markers=True)
                fig_hourly.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
                st.plotly_chart(fig_hourly, use_container_width=True)

            with col2:
                # Hour volume analysis
                fig_hourly_volume = px.bar(hourly_fraud,
                                           x='hour',
                                           y='count',
                                           title="Transaction Volume by Hour",
                                           color='fraud_rate',
                                           color_continuous_scale='Reds')
                fig_hourly_volume.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
                st.plotly_chart(fig_hourly_volume, use_container_width=True)

        # Statistical Summary Dashboard
        st.markdown('<h2 class="sub-header">Statistical Summary Dashboard</h2>', unsafe_allow_html=True)

        # Create comprehensive statistical summary
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("#### 📊 Dataset Overview")
            st.metric("Total Records", f"{len(data):,}")
            st.metric("Features", len(data.columns))
            st.metric("Fraud Cases", f"{data['is_fraud'].sum():,}" if 'is_fraud' in data.columns else "N/A")

        with col2:
            st.markdown("#### 💰 Amount Statistics")
            if 'amount' in data.columns:
                st.metric("Average Amount", f"${data['amount'].mean():.2f}")
                st.metric("Median Amount", f"${data['amount'].median():.2f}")
                st.metric("Max Amount", f"${data['amount'].max():.2f}")

        with col3:
            st.markdown("#### 🌍 Geographic Coverage")
            if 'state' in data.columns:
                st.metric("States Covered", data['state'].nunique())
            if 'city' in data.columns:
                st.metric("Cities Covered", data['city'].nunique())
            if 'zip' in data.columns:
                st.metric("ZIP Codes", data['zip'].nunique())

        with col4:
            st.markdown("#### 🏪 Business Metrics")
            if 'category' in data.columns:
                st.metric("Categories", data['category'].nunique())
            if 'merchant' in data.columns:
                st.metric("Merchants", data['merchant'].nunique())
            if 'job' in data.columns:
                st.metric("Job Types", data['job'].nunique())

        # Advanced Distribution Analysis
        st.markdown('<h2 class="sub-header">Advanced Distribution Analysis</h2>', unsafe_allow_html=True)

        if 'amount' in data.columns:
            col1, col2 = st.columns(2)

            with col1:
                # Amount distribution comparison
                fig_violin = px.violin(data,
                                       x='is_fraud',
                                       y='amount',
                                       title="Amount Distribution Comparison (Violin Plot)",
                                       box=True)
                fig_violin.update_xaxes(tickvals=[0, 1], ticktext=['Legitimate', 'Fraudulent'])
                st.plotly_chart(fig_violin, use_container_width=True)

            with col2:
                # Log scale amount analysis
                data_log = data[data['amount'] > 0].copy()
                data_log['log_amount'] = np.log10(data_log['amount'])

                fig_log_hist = px.histogram(data_log,
                                            x='log_amount',
                                            color='is_fraud',
                                            title="Log-Scale Amount Distribution",
                                            marginal='box',
                                            nbins=50)
                st.plotly_chart(fig_log_hist, use_container_width=True)

    else:
        st.markdown("""
        <div class="warning-card">
            <h3>📊 No Data Available</h3>
            <p>Please upload a dataset first using the Data Upload page.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔧 Data Preprocessing":
    st.markdown('<h1 class="main-header">🔧 Data Preprocessing</h1>', unsafe_allow_html=True)

    if st.session_state.data is not None:
        data = st.session_state.data.copy()

        st.markdown('<h2 class="sub-header">Preprocessing Steps</h2>', unsafe_allow_html=True)

        # Step 1: Handle Missing Values
        st.markdown("### Step 1: Handle Missing Values")

        missing_values = data.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing Percentage': (missing_values.values / len(data)) * 100
        }).sort_values('Missing Count', ascending=False)

        st.dataframe(missing_df, use_container_width=True)

        if missing_values.sum() > 0:
            st.warning(f"Found {missing_values.sum()} missing values in the dataset.")

            # Handle missing values
            # For numerical columns, fill with median
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if data[col].isnull().sum() > 0:
                    median_val = data[col].median()
                    data[col].fillna(median_val, inplace=True)
                    st.info(f"Filled missing values in '{col}' with median: {median_val}")

            # For categorical columns, fill with mode
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if data[col].isnull().sum() > 0:
                    mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown'
                    data[col].fillna(mode_val, inplace=True)
                    st.info(f"Filled missing values in '{col}' with mode: {mode_val}")
        else:
            st.success("✅ No missing values found in the dataset!")

        # Step 2: Encode Categorical Variables
        st.markdown("### Step 2: Encode Categorical Variables")

        categorical_cols = data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['transaction_date', 'transaction_time']]

        encoded_data = pd.DataFrame()
        label_encoders = {}

        if len(categorical_cols) > 0:
            st.info(f"Found categorical columns: {list(categorical_cols)}")

            for col in categorical_cols:
                if col in data.columns:
                    le = LabelEncoder()
                    # Create encoded column in the new DataFrame
                    encoded_data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
                    label_encoders[col] = le

                    # Show encoding example
                    unique_vals = data[col].unique()[:5]
                    encoded_vals = le.transform(unique_vals.astype(str))

                    encoding_df = pd.DataFrame({
                        'Original': unique_vals,
                        'Encoded': encoded_vals
                    })

                    st.write(f"**{col}** encoding preview:")
                    st.dataframe(encoding_df)

            st.session_state.label_encoders = label_encoders
            st.success(f"✅ Encoded {len(categorical_cols)} categorical columns!")
        else:
            st.info("No categorical columns found that need encoding.")

        # Step 3: Feature Selection
        st.markdown("### Step 3: Feature Selection")

        # Select features for modeling
        numerical_cols = data.select_dtypes(include=[np.number]).columns

        # Combine numerical features and the newly created encoded features
        X = pd.concat([data[numerical_cols], encoded_data], axis=1)

        # Drop the target variable and other non-feature columns
        X = X.drop(columns=['is_fraud'], errors='ignore')
        y = data['is_fraud'] if 'is_fraud' in data.columns else None

        if y is not None:
            # Step 4: Feature Scaling
            st.markdown("### Step 4: Feature Scaling")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

            st.session_state.scaler = scaler

            # Show scaling statistics
            scaling_stats = pd.DataFrame({
                'Feature': X.columns,
                'Original Mean': X.mean().round(4),
                'Original Std': X.std().round(4),
                'Scaled Mean': X_scaled_df.mean().round(4),
                'Scaled Std': X_scaled_df.std().round(4)
            })

            st.dataframe(scaling_stats, use_container_width=True)
            st.success("✅ Features scaled using StandardScaler!")

            # Step 5: Final Dataset
            st.markdown("### Step 5: Final Preprocessed Dataset")

            processed_data = {
                'X': X_scaled_df,
                'y': y,
                'feature_names': list(X.columns)
            }

            st.session_state.processed_data = processed_data

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Number of Features", len(X.columns))
                st.metric("Number of Samples", len(X))

            with col2:
                st.metric("Fraud Cases", y.sum())
                st.metric("Normal Cases", (y == 0).sum())

            # Show preprocessed data preview
            preview_data = pd.concat([X_scaled_df.head(), y.head()], axis=1)
            st.write("**Preprocessed Data Preview:**")
            st.dataframe(preview_data, use_container_width=True)

            st.markdown("""
            <div class="success-card">
                <h3>✅ Data Preprocessing Complete!</h3>
                <p>Your data is now ready for machine learning model training.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error("Target variable 'is_fraud' not found in the dataset!")

    else:
        st.markdown("""
        <div class="warning-card">
            <h3>🔧 No Data Available</h3>
            <p>Please upload a dataset first using the Data Upload page.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🤖 Model Training":
    st.markdown('<h1 class="main-header">🤖 Model Training</h1>', unsafe_allow_html=True)

    if st.session_state.processed_data is not None:
        processed_data = st.session_state.processed_data
        X = processed_data['X']
        y = processed_data['y']

        st.markdown('<h2 class="sub-header">Train Machine Learning Models</h2>', unsafe_allow_html=True)

        # Model selection
        st.markdown("### Select Models to Train")

        col1, col2, col3 = st.columns(3)

        with col1:
            train_knn = st.checkbox("📍 K-Nearest Neighbors", value=True)
        with col2:
            train_svm = st.checkbox("⚡ Support Vector Machine", value=True)
        with col3:
            train_rf = st.checkbox("🌳 Random Forest", value=True)

        # Model parameters
        st.markdown("### Model Parameters")

        param_col1, param_col2, param_col3 = st.columns(3)

        with param_col1:
            if train_knn:
                st.write("**K-NN Parameters**")
                knn_k = st.slider("Number of Neighbors (k)", 3, 21, 5, step=2)
                knn_weights = st.selectbox("Weight Function", ['uniform', 'distance'])

        with param_col2:
            if train_svm:
                st.write("**SVM Parameters**")
                svm_kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
                svm_c = st.slider("Regularization (C)", 0.1, 10.0, 1.0, step=0.1)

        with param_col3:
            if train_rf:
                st.write("**Random Forest Parameters**")
                rf_n_estimators = st.slider("Number of Trees", 10, 200, 100, step=10)
                rf_max_depth = st.slider("Max Depth", 3, 20, 10)
        # Advanced Model Configuration
        st.markdown("### Advanced Configuration")

        with st.expander("Hyperparameter Suggestions"):
            st.markdown("""
            **Recommended hyperparameters based on dataset characteristics:**

            **For Imbalanced Datasets:**
            - **Random Forest**: Increase `class_weight='balanced'` or `class_weight='balanced_subsample'`
            - **SVM**: Use `class_weight='balanced'` parameter
            - **K-NN**: Consider using `weights='distance'` for better handling of minority class

            **For Large Datasets (>10,000 samples):**
            - **Random Forest**: n_estimators=200-500, max_depth=10-15
            - **SVM**: Consider using `kernel='linear'` for faster training
            - **K-NN**: Use smaller k values (3-7) for faster inference

            **For Small Datasets (<1,000 samples):**
            - **Random Forest**: n_estimators=50-100, max_depth=5-10 
            - **SVM**: Use `kernel='rbf'` with cross-validation for C parameter
            - **K-NN**: Use larger k values (7-15) to reduce overfitting
            """)
        # Train models button
        if st.button("🚀 Train Selected Models", type="primary"):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            models = {}
            training_progress = st.progress(0)
            status_text = st.empty()

            model_count = sum([train_knn, train_svm, train_rf])
            current_model = 0

            # Train K-NN
            if train_knn:
                current_model += 1
                status_text.text(f"Training K-Nearest Neighbors... ({current_model}/{model_count})")

                knn = KNeighborsClassifier(n_neighbors=knn_k, weights=knn_weights)
                knn.fit(X_train, y_train)
                models['K-NN'] = knn

                training_progress.progress(current_model / model_count)
                st.success(f"✅ K-NN trained successfully with k={knn_k}")

            # Train SVM
            if train_svm:
                current_model += 1
                status_text.text(f"Training Support Vector Machine... ({current_model}/{model_count})")

                svm = SVC(kernel=svm_kernel, C=svm_c, probability=True, random_state=42)
                svm.fit(X_train, y_train)
                models['SVM'] = svm

                training_progress.progress(current_model / model_count)
                st.success(f"✅ SVM trained successfully with kernel={svm_kernel}, C={svm_c}")

            # Train Random Forest
            if train_rf:
                current_model += 1
                status_text.text(f"Training Random Forest... ({current_model}/{model_count})")

                rf = RandomForestClassifier(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    random_state=42
                )
                rf.fit(X_train, y_train)
                models['Random Forest'] = rf

                training_progress.progress(current_model / model_count)
                st.success(f"✅ Random Forest trained successfully with {rf_n_estimators} trees")

            # Store models and data splits
            st.session_state.models = models
            st.session_state.data_splits = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }

            status_text.text("Training complete!")
            training_progress.progress(1.0)

            st.markdown("""
            <div class="success-card">
                <h3>🎉 Model Training Complete!</h3>
                <p>All selected models have been trained successfully. You can now proceed to model evaluation.</p>
            </div>
            """, unsafe_allow_html=True)

            # Show model summary
            st.markdown("### 📋 Training Summary")

            summary_data = []
            for name, model in models.items():
                summary_data.append({
                    'Model': name,
                    'Type': type(model).__name__,
                    'Training Samples': len(X_train),
                    'Features': len(X.columns)
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

    else:
        st.markdown("""
        <div class="warning-card">
            <h3>🤖 No Preprocessed Data Available</h3>
            <p>Please complete data preprocessing first before training models.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📈 Model Evaluation":
    st.markdown('<h1 class="main-header">📈 Model Evaluation</h1>', unsafe_allow_html=True)

    if st.session_state.models and st.session_state.get('data_splits'):
        models = st.session_state.models
        data_splits = st.session_state.data_splits
        X_train, X_test, y_train, y_test = data_splits['X_train'], data_splits['X_test'], data_splits['y_train'], \
            data_splits['y_test']

        st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)

        # Evaluate all models
        results = {}

        for name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

        st.session_state.model_results = results

        # Display performance metrics
        st.markdown("### 📊 Performance Metrics")

        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1']:.4f}"
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

        # Performance visualization
        st.markdown("### 📊 Performance Comparison")

        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        model_names = list(results.keys())

        fig_metrics = go.Figure()

        for metric in metric_names:
            # Fixed the key mapping here
            metric_key_map = {
                'Accuracy': 'accuracy',
                'Precision': 'precision',
                'Recall': 'recall',
                'F1-Score': 'f1'  # Changed from 'f1_score' to 'f1'
            }

            metric_key = metric_key_map[metric]
            metric_values = [results[model][metric_key] for model in model_names]

            fig_metrics.add_trace(go.Bar(
                name=metric,
                x=model_names,
                y=metric_values,
                text=[f'{val:.3f}' for val in metric_values],
                textposition='auto'
            ))

        fig_metrics.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            barmode='group',
            height=500
        )

        st.plotly_chart(fig_metrics, use_container_width=True)

        # Enhanced Performance Analysis
        st.markdown("### 🎯 Advanced Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Radar chart for model comparison
            fig_radar = go.Figure()

            for name, result in results.items():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[result['accuracy'], result['precision'], result['recall'], result['f1']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    fill='toself',
                    name=name
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Model Performance Radar Chart",
                showlegend=True
            )

            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            # Performance ranking
            ranking_data = []
            for name, result in results.items():
                avg_score = np.mean([result['accuracy'], result['precision'], result['recall'], result['f1']])
                ranking_data.append({
                    'Model': name,
                    'Average Score': avg_score,
                    'Rank': 0  # Will be filled after sorting
                })

            ranking_df = pd.DataFrame(ranking_data).sort_values('Average Score', ascending=False).reset_index(drop=True)
            ranking_df['Rank'] = range(1, len(ranking_df) + 1)

            fig_ranking = px.bar(ranking_df,
                                 x='Average Score',
                                 y='Model',
                                 orientation='h',
                                 title="Model Ranking by Average Score",
                                 color='Average Score',
                                 color_continuous_scale='Viridis',
                                 text='Rank')
            fig_ranking.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig_ranking.update_traces(texttemplate='#%{text}', textposition='outside')

            st.plotly_chart(fig_ranking, use_container_width=True)

        # Confusion Matrices
        st.markdown("### 📊 Confusion Matrices")

        cols = st.columns(len(models))

        for i, (name, result) in enumerate(results.items()):
            with cols[i]:
                cm = confusion_matrix(y_test, result['y_pred'])

                fig_cm = px.imshow(cm,
                                   text_auto=True,
                                   aspect="auto",
                                   color_continuous_scale="Blues",
                                   title=f'{name} Confusion Matrix')
                fig_cm.update_layout(
                    xaxis=dict(title="Predicted", tickvals=[0, 1], ticktext=['Not Fraud', 'Fraud']),
                    yaxis=dict(title="Actual", tickvals=[0, 1], ticktext=['Not Fraud', 'Fraud'])
                )

                st.plotly_chart(fig_cm, use_container_width=True)

        # Enhanced Confusion Matrix Analysis
        st.markdown("### 🔍 Detailed Confusion Matrix Analysis")

        cm_analysis_data = []
        for name, result in results.items():
            cm = confusion_matrix(y_test, result['y_pred'])
            tn, fp, fn, tp = cm.ravel()

            cm_analysis_data.append({
                'Model': name,
                'True Negatives': tn,
                'False Positives': fp,
                'False Negatives': fn,
                'True Positives': tp,
                'Specificity': tn / (tn + fp),
                'Sensitivity (Recall)': tp / (tp + fn),
                'False Positive Rate': fp / (fp + tn),
                'False Negative Rate': fn / (fn + tp)
            })

        cm_analysis_df = pd.DataFrame(cm_analysis_data)
        st.dataframe(cm_analysis_df.round(4), use_container_width=True)

        # ROC Curves
        st.markdown("### 📈 ROC Curves")

        col1, col2 = st.columns(2)

        with col1:
            fig_roc = go.Figure()

            auc_scores = {}
            for name, result in results.items():
                if result['y_pred_proba'] is not None:
                    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                    auc_score = auc(fpr, tpr)
                    auc_scores[name] = auc_score

                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{name} (AUC = {auc_score:.3f})',
                        line=dict(width=3)
                    ))

            # Add diagonal line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(dash='dash', color='gray', width=2)
            ))

            fig_roc.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500,
                showlegend=True
            )

            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            # AUC comparison bar chart
            if auc_scores:
                auc_df = pd.DataFrame(list(auc_scores.items()), columns=['Model', 'AUC Score'])
                auc_df = auc_df.sort_values('AUC Score', ascending=True)

                fig_auc = px.bar(auc_df,
                                 x='AUC Score',
                                 y='Model',
                                 orientation='h',
                                 title="AUC Score Comparison",
                                 color='AUC Score',
                                 color_continuous_scale='Viridis',
                                 text='AUC Score')
                fig_auc.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig_auc.update_traces(texttemplate='%{text:.3f}', textposition='outside')

                st.plotly_chart(fig_auc, use_container_width=True)

        # Precision-Recall Curves
        st.markdown("### 📈 Precision-Recall Curves")

        from sklearn.metrics import precision_recall_curve, average_precision_score

        col1, col2 = st.columns(2)

        with col1:
            fig_pr = go.Figure()

            ap_scores = {}
            for name, result in results.items():
                if result['y_pred_proba'] is not None:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_test, result['y_pred_proba'])
                    ap_score = average_precision_score(y_test, result['y_pred_proba'])
                    ap_scores[name] = ap_score

                    fig_pr.add_trace(go.Scatter(
                        x=recall_curve, y=precision_curve,
                        mode='lines',
                        name=f'{name} (AP = {ap_score:.3f})',
                        line=dict(width=3)
                    ))

            # Add baseline
            baseline = y_test.sum() / len(y_test)
            fig_pr.add_trace(go.Scatter(
                x=[0, 1], y=[baseline, baseline],
                mode='lines',
                name=f'Baseline (AP = {baseline:.3f})',
                line=dict(dash='dash', color='gray', width=2)
            ))

            fig_pr.update_layout(
                title='Precision-Recall Curves',
                xaxis_title='Recall',
                yaxis_title='Precision',
                height=500
            )

            st.plotly_chart(fig_pr, use_container_width=True)

        with col2:
            # Average Precision comparison
            if ap_scores:
                ap_df = pd.DataFrame(list(ap_scores.items()), columns=['Model', 'Average Precision'])
                ap_df = ap_df.sort_values('Average Precision', ascending=True)

                fig_ap = px.bar(ap_df,
                                x='Average Precision',
                                y='Model',
                                orientation='h',
                                title="Average Precision Score Comparison",
                                color='Average Precision',
                                color_continuous_scale='Plasma',
                                text='Average Precision')
                fig_ap.update_layout(yaxis={'categoryorder': 'total ascending'})
                fig_ap.update_traces(texttemplate='%{text:.3f}', textposition='outside')

                st.plotly_chart(fig_ap, use_container_width=True)

        # Model Comparison Heatmap
        st.markdown("### 🌡️ Model Performance Heatmap")

        # Create performance matrix
        performance_matrix = []
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        for name in model_names:
            row = [results[name][metric] for metric in metrics]
            performance_matrix.append(row)

        performance_df = pd.DataFrame(performance_matrix,
                                      index=model_names,
                                      columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

        fig_heatmap = px.imshow(performance_df,
                                text_auto='.3f',
                                aspect="auto",
                                color_continuous_scale="RdYlGn",
                                title="Model Performance Heatmap")
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Error Analysis
        st.markdown("### 🔍 Error Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # False Positive vs False Negative Analysis
            error_data = []
            for name, result in results.items():
                cm = confusion_matrix(y_test, result['y_pred'])
                tn, fp, fn, tp = cm.ravel()

                error_data.append({
                    'Model': name,
                    'False Positives': fp,
                    'False Negatives': fn,
                    'Total Errors': fp + fn
                })

            error_df = pd.DataFrame(error_data)

            fig_errors = px.scatter(error_df,
                                    x='False Positives',
                                    y='False Negatives',
                                    size='Total Errors',
                                    hover_name='Model',
                                    title="False Positives vs False Negatives",
                                    color='Total Errors',
                                    color_continuous_scale='Reds')
            st.plotly_chart(fig_errors, use_container_width=True)

        with col2:
            # Cost-Benefit Analysis (assuming cost ratios)
            cost_fp = 10  # Cost of false positive
            cost_fn = 100  # Cost of false negative (much higher for fraud)

            cost_data = []
            for name, result in results.items():
                cm = confusion_matrix(y_test, result['y_pred'])
                tn, fp, fn, tp = cm.ravel()

                total_cost = (fp * cost_fp) + (fn * cost_fn)
                cost_data.append({
                    'Model': name,
                    'False Positive Cost': fp * cost_fp,
                    'False Negative Cost': fn * cost_fn,
                    'Total Cost': total_cost
                })

            cost_df = pd.DataFrame(cost_data).sort_values('Total Cost')

            fig_cost = px.bar(cost_df,
                              x='Model',
                              y=['False Positive Cost', 'False Negative Cost'],
                              title="Cost Analysis (FP=$10, FN=$100)",
                              color_discrete_map={
                                  'False Positive Cost': '#3498db',
                                  'False Negative Cost': '#e74c3c'
                              })
            st.plotly_chart(fig_cost, use_container_width=True)

        # Threshold Analysis
        st.markdown("### 🎚️ Threshold Analysis")

        # Allow user to select model for threshold analysis
        selected_model_threshold = st.selectbox(
            "Select model for threshold analysis:",
            options=[name for name, result in results.items() if result['y_pred_proba'] is not None]
        )

        if selected_model_threshold:
            result = results[selected_model_threshold]

            # Generate different thresholds
            thresholds = np.arange(0.1, 1.0, 0.05)
            threshold_metrics = []

            for thresh in thresholds:
                y_pred_thresh = (result['y_pred_proba'] >= thresh).astype(int)

                acc = accuracy_score(y_test, y_pred_thresh)
                prec = precision_score(y_test, y_pred_thresh, zero_division=0)
                rec = recall_score(y_test, y_pred_thresh, zero_division=0)
                f1_thresh = f1_score(y_test, y_pred_thresh, zero_division=0)

                threshold_metrics.append({
                    'Threshold': thresh,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1-Score': f1_thresh
                })

            threshold_df = pd.DataFrame(threshold_metrics)

            col1, col2 = st.columns(2)

            with col1:
                # Threshold curves
                fig_thresh = go.Figure()

                for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                    fig_thresh.add_trace(go.Scatter(
                        x=threshold_df['Threshold'],
                        y=threshold_df[metric],
                        mode='lines+markers',
                        name=metric
                    ))

                fig_thresh.update_layout(
                    title=f'Performance vs Threshold - {selected_model_threshold}',
                    xaxis_title='Threshold',
                    yaxis_title='Score',
                    height=400
                )

                st.plotly_chart(fig_thresh, use_container_width=True)

            with col2:
                # Optimal threshold recommendation
                f1_scores = threshold_df['F1-Score'].values
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = threshold_df.iloc[optimal_idx]['Threshold']
                optimal_f1 = threshold_df.iloc[optimal_idx]['F1-Score']

                st.markdown(f"""
                <div class="success-card">
                    <h4>🎯 Optimal Threshold</h4>
                    <p><strong>Threshold:</strong> {optimal_threshold:.2f}</p>
                    <p><strong>F1-Score:</strong> {optimal_f1:.4f}</p>
                    <p><strong>Model:</strong> {selected_model_threshold}</p>
                </div>
                """, unsafe_allow_html=True)

                # Show metrics at optimal threshold
                optimal_metrics = threshold_df.iloc[optimal_idx]
                metrics_display = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    'Score': [optimal_metrics['Accuracy'], optimal_metrics['Precision'],
                              optimal_metrics['Recall'], optimal_metrics['F1-Score']]
                }).round(4)

                st.dataframe(metrics_display, use_container_width=True)

        # Best model recommendation
        st.markdown("### 🏆 Best Model Recommendation")

        # Find best model based on F1 score
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
        best_f1 = results[best_model_name]['f1']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="success-card">
                <h3>🥇 Recommended Model: {best_model_name}</h3>
                <p><strong>F1-Score:</strong> {best_f1:.4f}</p>
                <p>This model provides the best balance between precision and recall for fraud detection.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            # Model recommendation reasoning
            best_result = results[best_model_name]

            st.markdown("#### 📋 Model Strengths")
            strengths = []

            if best_result['precision'] > 0.8:
                strengths.append("🎯 High precision - Low false positive rate")
            if best_result['recall'] > 0.8:
                strengths.append("📡 High recall - Catches most fraud cases")
            if best_result['accuracy'] > 0.9:
                strengths.append("🎪 High overall accuracy")
            if best_result['f1'] > 0.8:
                strengths.append("⚖️ Well-balanced performance")

            for strength in strengths:
                st.markdown(f"- {strength}")

            if not strengths:
                st.markdown("- 🔄 Best performing among available models")

        # Detailed classification reports
        st.markdown("### 📋 Detailed Classification Reports")

        for name, result in results.items():
            with st.expander(f"{name} Classification Report"):
                report = classification_report(y_test, result['y_pred'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)

                # Additional insights
                cm = confusion_matrix(y_test, result['y_pred'])
                tn, fp, fn, tp = cm.ravel()

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("True Positives", tp)
                    st.metric("False Positives", fp)

                with col2:
                    st.metric("True Negatives", tn)
                    st.metric("False Negatives", fn)

                with col3:
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                    st.metric("Specificity", f"{specificity:.4f}")
                    st.metric("Negative Predictive Value", f"{npv:.4f}")

        # Feature importance (for Random Forest)
        if 'Random Forest' in models:
            st.markdown("### 🌟 Feature Importance Analysis")

            rf_model = models['Random Forest']
            feature_names = st.session_state.processed_data['feature_names']

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                fig_importance = px.bar(importance_df.head(15),
                                        x='Importance', y='Feature',
                                        orientation='h',
                                        title='Top 15 Most Important Features',
                                        color='Importance',
                                        color_continuous_scale='Viridis')
                fig_importance.update_layout(yaxis=dict(categoryorder='total ascending'))
                st.plotly_chart(fig_importance, use_container_width=True)

            with col2:
                # Feature importance statistics
                st.markdown("#### 📊 Feature Importance Statistics")

                importance_stats = {
                    'Total Features': len(importance_df),
                    'Top 5 Features Contribution': f"{importance_df.head(5)['Importance'].sum():.2%}",
                    'Top 10 Features Contribution': f"{importance_df.head(10)['Importance'].sum():.2%}",
                    'Most Important Feature': importance_df.iloc[0]['Feature'],
                    'Least Important Feature': importance_df.iloc[-1]['Feature']
                }

                for key, value in importance_stats.items():
                    st.write(f"**{key}:** {value}")

                # Feature importance distribution
                fig_importance_dist = px.histogram(importance_df,
                                                   x='Importance',
                                                   nbins=20,
                                                   title="Feature Importance Distribution")
                st.plotly_chart(fig_importance_dist, use_container_width=True)

        # Model Comparison Summary
        st.markdown("### 📊 Executive Summary")

        # Create summary table
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.1%}",
                'Precision': f"{result['precision']:.1%}",
                'Recall': f"{result['recall']:.1%}",
                'F1-Score': f"{result['f1']:.1%}",
                'Recommendation': '🥇 Best' if name == best_model_name else '✅ Good' if result['f1'] > 0.7 else '⚠️ Fair'
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        # Key insights
        st.markdown("#### 🔍 Key Insights")

        insights = []

        # Performance insights
        avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
        if avg_accuracy > 0.9:
            insights.append("🎯 All models show high accuracy (>90%)")

        precision_values = [r['precision'] for r in results.values()]
        if max(precision_values) - min(precision_values) > 0.1:
            insights.append("📊 Significant precision variation between models")

        recall_values = [r['recall'] for r in results.values()]
        if max(recall_values) > 0.9:
            insights.append("📡 Excellent fraud detection capability achieved")

        if len(insights) == 0:
            insights.append("📈 Models show consistent performance across metrics")

        for insight in insights:
            st.markdown(f"- {insight}")

    else:
        st.markdown("""
        <div class="warning-card">
            <h3>📈 No Models Available</h3>
            <p>Please train models first using the Model Training page.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔮 Fraud Prediction":
    st.markdown('<h1 class="main-header">🔮 Fraud Prediction</h1>', unsafe_allow_html=True)

    if st.session_state.models and st.session_state.scaler:
        models = st.session_state.models
        scaler = st.session_state.scaler
        feature_names = st.session_state.processed_data['feature_names']
        label_encoders = st.session_state.label_encoders

        st.markdown('<h2 class="sub-header">Predict Transaction Fraud</h2>', unsafe_allow_html=True)

        # Model selection for prediction
        selected_model = st.selectbox(
            "Select Model for Prediction:",
            options=list(models.keys()),
            help="Choose which trained model to use for prediction"
        )

        st.markdown("### 🔍 Enter Transaction Details")

        # Create input form based on original features
        col1, col2, col3 = st.columns(3)

        # Initialize input values
        input_values = {}

        with col1:
            st.markdown("**Transaction Details**")
            input_values['amount'] = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=0.01)

            # Category selection (if categorical features exist)
            if 'category' in st.session_state.data.columns:
                categories = st.session_state.data['category'].unique()
                selected_category = st.selectbox("Category", options=categories)
                input_values['category'] = selected_category

            if 'merchant' in st.session_state.data.columns:
                merchants = st.session_state.data['merchant'].unique()[:50]  # Limit to first 50 for performance
                selected_merchant = st.selectbox("Merchant", options=merchants)
                input_values['merchant'] = selected_merchant

        with col2:
            st.markdown("**Location Details**")
            if 'city' in st.session_state.data.columns:
                cities = st.session_state.data['city'].unique()[:100]  # Limit for performance
                selected_city = st.selectbox("City", options=cities)
                input_values['city'] = selected_city

            if 'state' in st.session_state.data.columns:
                states = st.session_state.data['state'].unique()
                selected_state = st.selectbox("State", options=states)
                input_values['state'] = selected_state

            input_values['latitude'] = st.number_input("Latitude", value=40.7128, step=0.0001, format="%.4f")
            input_values['longitude'] = st.number_input("Longitude", value=-74.0060, step=0.0001, format="%.4f")

        with col3:
            st.markdown("**Customer Details**")
            if 'job' in st.session_state.data.columns:
                jobs = st.session_state.data['job'].unique()[:50]  # Limit for performance
                selected_job = st.selectbox("Job", options=jobs)
                input_values['job'] = selected_job

            input_values['city_population'] = st.number_input("City Population", min_value=1, value=8000000, step=1000)
            input_values['zip'] = st.number_input("ZIP Code", min_value=10000, max_value=99999, value=10001)
            input_values['unix_time'] = st.number_input("Unix Time", value=1325376018)

        # Additional merchant location if available
        if 'merchant_latitude' in st.session_state.data.columns:
            input_values['merchant_latitude'] = st.number_input("Merchant Latitude", value=40.7580, step=0.0001,
                                                                format="%.4f")
            input_values['merchant_longitude'] = st.number_input("Merchant Longitude", value=-73.9855, step=0.0001,
                                                                 format="%.4f")

        # Predict button
        if st.button("🔍 Predict Fraud", type="primary"):
            try:
                # Prepare input data for prediction
                input_df = pd.DataFrame([input_values])

                # Encode categorical variables using stored encoders
                for col, encoder in label_encoders.items():
                    if col in input_df.columns:
                        try:
                            input_df[col + '_encoded'] = encoder.transform(input_df[col].astype(str))
                        except ValueError:
                            # Handle unknown categories
                            input_df[col + '_encoded'] = 0
                            st.warning(f"Unknown value in {col}, using default encoding.")

                # Select features that match training data
                available_features = [col for col in feature_names if
                                      col in input_df.columns or col.replace('_encoded', '') in input_df.columns]

                # Create feature vector
                feature_vector = []
                for feature in feature_names:
                    if feature in input_df.columns:
                        feature_vector.append(input_df[feature].iloc[0])
                    elif feature.endswith('_encoded'):
                        base_feature = feature.replace('_encoded', '')
                        if base_feature in input_df.columns:
                            if base_feature in label_encoders:
                                try:
                                    encoded_val = \
                                        label_encoders[base_feature].transform([str(input_df[base_feature].iloc[0])])[0]
                                    feature_vector.append(encoded_val)
                                except:
                                    feature_vector.append(0)
                            else:
                                feature_vector.append(0)
                        else:
                            feature_vector.append(0)
                    else:
                        feature_vector.append(0)  # Default value for missing features

                # Convert to numpy array and reshape
                feature_array = np.array(feature_vector).reshape(1, -1)

                # Scale features
                feature_scaled = scaler.transform(feature_array)

                # Make prediction
                model = models[selected_model]
                prediction = model.predict(feature_scaled)[0]

                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(feature_scaled)[0]
                    fraud_probability = probabilities[1]
                    confidence = max(probabilities)
                else:
                    fraud_probability = prediction
                    confidence = 1.0

                # Display results
                st.markdown("### 🎯 Prediction Results")

                if prediction == 1:
                    st.markdown(f"""
                    <div class="warning-card">
                        <h3>⚠️ FRAUD DETECTED</h3>
                        <p><strong>Prediction:</strong> This transaction is likely fraudulent</p>
                        <p><strong>Fraud Probability:</strong> {fraud_probability:.4f} ({fraud_probability * 100:.2f}%)</p>
                        <p><strong>Confidence:</strong> {confidence:.4f} ({confidence * 100:.2f}%)</p>
                        <p><strong>Model Used:</strong> {selected_model}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>✅ LEGITIMATE TRANSACTION</h3>
                        <p><strong>Prediction:</strong> This transaction appears legitimate</p>
                        <p><strong>Fraud Probability:</strong> {fraud_probability:.4f} ({fraud_probability * 100:.2f}%)</p>
                        <p><strong>Confidence:</strong> {confidence:.4f} ({confidence * 100:.2f}%)</p>
                        <p><strong>Model Used:</strong> {selected_model}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Risk assessment
                st.markdown("### 📊 Risk Assessment")

                risk_col1, risk_col2, risk_col3 = st.columns(3)

                with risk_col1:
                    if fraud_probability < 0.3:
                        risk_level = "LOW"
                        risk_color = "green"
                    elif fraud_probability < 0.7:
                        risk_level = "MEDIUM"
                        risk_color = "orange"
                    else:
                        risk_level = "HIGH"
                        risk_color = "red"

                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; border: 2px solid {risk_color}; border-radius: 10px;">
                        <h3 style="color: {risk_color};">Risk Level</h3>
                        <h2 style="color: {risk_color};">{risk_level}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with risk_col2:
                    st.metric("Fraud Score", f"{fraud_probability:.4f}")
                    st.metric("Model Confidence", f"{confidence:.4f}")

                with risk_col3:
                    st.metric("Transaction Amount", f"${input_values['amount']:.2f}")
                    if 'category' in input_values:
                        st.write(f"**Category:** {input_values['category']}")

                # Recommendations
                st.markdown("### 💡 Recommendations")

                if prediction == 1:
                    st.markdown("""
                    **Immediate Actions:**
                    - 🚫 Block or hold the transaction for manual review
                    - 📞 Contact the cardholder to verify the transaction
                    - 🔍 Flag the account for additional monitoring
                    - 📊 Investigate similar patterns in recent transactions
                    """)
                else:
                    st.markdown("""
                    **Actions:**
                    - ✅ Transaction appears safe to proceed
                    - 📊 Continue normal monitoring protocols
                    - 🔍 Log transaction for future analysis
                    """)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all required fields are filled correctly.")

        # Batch prediction option
        st.markdown("### 📂 Batch Prediction")
        st.info(
            "For processing multiple transactions, upload a CSV file with the same structure as your training data.")

        batch_file = st.file_uploader(
            "Upload transactions for batch prediction",
            type=['csv'],
            help="CSV file should have the same columns as training data"
        )

        if batch_file is not None and st.button("🚀 Process Batch"):
            try:
                batch_data = pd.read_csv(batch_file)

                # Process batch data (simplified version)
                st.write(f"Loaded {len(batch_data)} transactions for prediction.")

                # Note: In a real implementation, you would process each row
                # similar to the single prediction above
                st.info("Batch processing would be implemented here. For demo purposes, showing first 5 rows:")
                st.dataframe(batch_data.head())

            except Exception as e:
                st.error(f"Error processing batch file: {str(e)}")

    else:
        st.markdown("""
        <div class="warning-card">
            <h3>🔮 Models Not Available</h3>
            <p>Please train models and complete preprocessing before making predictions.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Credit Card Fraud Detection System | Group 10 | University of Ghana</p>
    <p>Built with ❤️ using Streamlit and Scikit-learn</p>
</div>
""", unsafe_allow_html=True)