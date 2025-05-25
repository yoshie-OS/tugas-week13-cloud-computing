import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import bigquery
from google.oauth2 import service_account
import os
from datetime import datetime
import numpy as np

# Set page config
st.set_page_config(
    page_title="Marmara Seismic Analysis",
    page_icon="🌍",
    layout="wide"
)

class SeismicAnalyzer:
    def __init__(self):
        self.client = None
        self.connected = False
        self.project_id = "tugas-week-13"
        self.dataset_id = "seismic_analysis"
        self.table_id = "gempa_marmara"
        self.full_table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
    
    def connect_to_bigquery(self, credentials_path=None):
        """Connect to BigQuery using service account credentials"""
        try:
            if credentials_path and os.path.exists(credentials_path):
                # Use service account file
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                # Try default credentials (for Cloud Run deployment)
                self.client = bigquery.Client(project=self.project_id)
            
            # Test connection
            query = f"SELECT COUNT(*) as total FROM `{self.full_table_id}` LIMIT 1"
            result = self.client.query(query).result()
            
            self.connected = True
            return True, "Successfully connected to BigQuery!"
            
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def execute_query(self, query):
        """Execute BigQuery SQL and return DataFrame"""
        if not self.connected:
            return None, "Not connected to BigQuery"
        
        try:
            result = self.client.query(query)
            df = result.to_dataframe()
            return df, "Success"
        except Exception as e:
            return None, f"Query failed: {str(e)}"
    
    def get_data_summary(self):
        """Get basic summary of earthquake data"""
        query = f"""
        SELECT 
            COUNT(*) as total_earthquakes,
            MIN(Date) as earliest_date,
            MAX(Date) as latest_date,
            MIN(Magnitude_ML) as min_magnitude,
            MAX(Magnitude_ML) as max_magnitude,
            AVG(Magnitude_ML) as avg_magnitude,
            MIN(Depth_km) as min_depth,
            MAX(Depth_km) as max_depth,
            AVG(Depth_km) as avg_depth,
            COUNT(DISTINCT Nearest_Fault) as unique_faults
        FROM `{self.full_table_id}`
        WHERE Magnitude_ML IS NOT NULL 
        AND Depth_km IS NOT NULL
        """
        return self.execute_query(query)
    
    def analyze_high_magnitude_earthquakes(self, magnitude_threshold):
        """Analyze earthquakes above a certain magnitude threshold"""
        query = f"""
        SELECT 
            Nearest_Fault,
            COUNT(*) as earthquake_count,
            AVG(Magnitude_ML) as avg_magnitude,
            MAX(Magnitude_ML) as max_magnitude,
            AVG(Depth_km) as avg_depth,
            AVG(Slip_Rate_mm_per_yr) as avg_slip_rate
        FROM `{self.full_table_id}`
        WHERE Magnitude_ML >= {magnitude_threshold}
        GROUP BY Nearest_Fault
        ORDER BY earthquake_count DESC
        """
        return self.execute_query(query)
    
    def analyze_depth_magnitude_correlation(self):
        """Analyze relationship between depth and magnitude"""
        query = f"""
        SELECT 
            Depth_km,
            Magnitude_ML,
            Latitude,
            Longitude,
            Nearest_Fault,
            Date
        FROM `{self.full_table_id}`
        WHERE Depth_km IS NOT NULL 
        AND Magnitude_ML IS NOT NULL
        AND Latitude IS NOT NULL 
        AND Longitude IS NOT NULL
        ORDER BY Magnitude_ML DESC
        LIMIT 1000
        """
        return self.execute_query(query)
    
    def get_fault_analysis(self):
        """Analyze earthquake distribution by fault"""
        query = f"""
        SELECT 
            Nearest_Fault,
            COUNT(*) as total_earthquakes,
            AVG(Magnitude_ML) as avg_magnitude,
            MAX(Magnitude_ML) as max_magnitude,
            AVG(Depth_km) as avg_depth,
            AVG(Fault_Length_km) as avg_fault_length,
            AVG(Slip_Rate_mm_per_yr) as avg_slip_rate,
            AVG(Recurrence_Interval_yr) as avg_recurrence_interval
        FROM `{self.full_table_id}`
        WHERE Nearest_Fault IS NOT NULL
        GROUP BY Nearest_Fault
        HAVING COUNT(*) >= 10
        ORDER BY total_earthquakes DESC
        """
        return self.execute_query(query)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SeismicAnalyzer()

# Main App
st.title("🌍 Marmara Region Seismic Analysis Dashboard")
st.markdown("**Earthquake Data Analysis from 2000-2025**")
st.markdown("---")

# Sidebar for connection
st.sidebar.header("🔗 BigQuery Connection")

# Check if credentials already exist
if 'credentials_uploaded' not in st.session_state:
    st.session_state.credentials_uploaded = False

# Show connection status
if st.session_state.analyzer.connected:
    st.sidebar.success("✅ Already connected!")
    if st.sidebar.button("Disconnect"):
        st.session_state.analyzer.connected = False
        st.session_state.credentials_uploaded = False
        st.rerun()
else:
    # Credentials file upload
    credentials_file = st.sidebar.file_uploader(
        "Upload Service Account JSON",
        type=['json'],
        help="Upload your GCP service account credentials file"
    )

    if credentials_file is not None:
        # Save uploaded file permanently
        credentials_path = "persistent_credentials.json"
        with open(credentials_path, "wb") as f:
            f.write(credentials_file.getbuffer())
        
        st.session_state.credentials_uploaded = True
        
        if st.sidebar.button("Connect to BigQuery"):
            with st.spinner("Connecting to BigQuery..."):
                success, message = st.session_state.analyzer.connect_to_bigquery(credentials_path)
            
            if success:
                st.sidebar.success(message)
                st.rerun()
            else:
                st.sidebar.error(message)
    
    # Auto-connect if credentials already uploaded
    elif st.session_state.credentials_uploaded and os.path.exists("persistent_credentials.json"):
        if st.sidebar.button("Connect with Saved Credentials"):
            with st.spinner("Connecting to BigQuery..."):
                success, message = st.session_state.analyzer.connect_to_bigquery("persistent_credentials.json")
            
            if success:
                st.sidebar.success(message)
                st.rerun()
            else:
                st.sidebar.error(message)

    # Alternative connection for Cloud Run (no file upload needed)
    if st.sidebar.button("Connect (Cloud Run Mode)"):
        with st.spinner("Connecting to BigQuery..."):
            success, message = st.session_state.analyzer.connect_to_bigquery()
        
        if success:
            st.sidebar.success(message)
            st.rerun()
        else:
            st.sidebar.error(message)

# Main content
if st.session_state.analyzer.connected:
    st.success("✅ Connected to BigQuery!")
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs([
        "📊 Data Overview", 
        "🏔️ High Magnitude Analysis", 
        "🗺️ Spatial Analysis"
    ])
    
    with tab1:
        st.header("Data Overview")
        
        # Debug section
        st.subheader("🔧 Debug Connection")
        if st.button("Test Basic Query"):
            test_query = f"SELECT COUNT(*) as total FROM `{st.session_state.analyzer.full_table_id}` LIMIT 1"
            st.code(test_query)
            
            test_df, test_message = st.session_state.analyzer.execute_query(test_query)
            if test_df is not None:
                st.success(f"✅ Connection works! Found {test_df.iloc[0]['total']} rows")
                st.write("Test result:", test_df)
            else:
                st.error(f"❌ Connection failed: {test_message}")
        
        if st.button("Check Table Schema"):
            schema_query = f"""
            SELECT column_name, data_type 
            FROM `{st.session_state.analyzer.project_id}.{st.session_state.analyzer.dataset_id}.INFORMATION_SCHEMA.COLUMNS` 
            WHERE table_name = '{st.session_state.analyzer.table_id}'
            """
            st.code(schema_query)
            
            schema_df, schema_message = st.session_state.analyzer.execute_query(schema_query)
            if schema_df is not None:
                st.success("✅ Table schema found!")
                st.dataframe(schema_df)
            else:
                st.error(f"❌ Schema check failed: {schema_message}")
        
        st.markdown("---")
        
        if st.button("Load Data Summary"):
            with st.spinner("Loading data summary..."):
                summary_df, message = st.session_state.analyzer.get_data_summary()
            
            if summary_df is not None:
                st.subheader("Database Summary Statistics")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Earthquakes", f"{summary_df.iloc[0]['total_earthquakes']:,}")
                with col2:
                    st.metric("Avg Magnitude", f"{summary_df.iloc[0]['avg_magnitude']:.2f}")
                with col3:
                    st.metric("Max Magnitude", f"{summary_df.iloc[0]['max_magnitude']:.2f}")
                with col4:
                    st.metric("Unique Faults", summary_df.iloc[0]['unique_faults'])
                
                # Display full summary
                st.subheader("Detailed Statistics")
                st.dataframe(summary_df)
                
                # Download option
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary as CSV",
                    data=csv,
                    file_name=f"seismic_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Store for other tabs
                st.session_state.data_summary = summary_df
            else:
                st.error(f"Failed to load summary: {message}")
    
    with tab2:
        st.header("High Magnitude Earthquake Analysis")
        
        magnitude_threshold = st.slider(
            "Minimum Magnitude Threshold", 
            min_value=2.0, 
            max_value=7.0, 
            value=4.0, 
            step=0.1
        )
        
        if st.button("Analyze High Magnitude Earthquakes"):
            with st.spinner("Analyzing high magnitude earthquakes..."):
                analysis_df, message = st.session_state.analyzer.analyze_high_magnitude_earthquakes(magnitude_threshold)
            
            if analysis_df is not None and len(analysis_df) > 0:
                st.subheader(f"Earthquakes with Magnitude ≥ {magnitude_threshold}")
                st.metric("Total High Magnitude Earthquakes", len(analysis_df))
                
                # Display results
                st.dataframe(analysis_df)
                
                # Download option
                csv = analysis_df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis as CSV",
                    data=csv,
                    file_name=f"high_magnitude_analysis_{magnitude_threshold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Bar chart of earthquake count by fault
                top_faults = analysis_df.head(10)
                ax1.bar(range(len(top_faults)), top_faults['earthquake_count'])
                ax1.set_title(f'Top 10 Faults by High Magnitude Earthquakes (≥{magnitude_threshold})')
                ax1.set_xlabel('Fault Index')
                ax1.set_ylabel('Earthquake Count')
                
                # Average magnitude by fault
                ax2.bar(range(len(top_faults)), top_faults['avg_magnitude'])
                ax2.set_title('Average Magnitude by Fault')
                ax2.set_xlabel('Fault Index')
                ax2.set_ylabel('Average Magnitude')
                
                st.pyplot(fig)
                
                # Store results
                st.session_state.high_mag_analysis = analysis_df
            else:
                st.warning("No earthquakes found above the specified magnitude threshold")
    
    with tab3:
        st.header("Spatial and Depth Analysis")
        
        if st.button("Analyze Depth-Magnitude Correlation"):
            with st.spinner("Loading spatial data..."):
                spatial_df, message = st.session_state.analyzer.analyze_depth_magnitude_correlation()
            
            if spatial_df is not None and len(spatial_df) > 0:
                # Scatter plot: Depth vs Magnitude
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(spatial_df['Depth_km'], spatial_df['Magnitude_ML'], 
                                   alpha=0.6, c=spatial_df['Magnitude_ML'], cmap='viridis')
                ax.set_xlabel('Depth (km)')
                ax.set_ylabel('Magnitude (ML)')
                ax.set_title('Earthquake Depth vs Magnitude Correlation')
                plt.colorbar(scatter, label='Magnitude')
                st.pyplot(fig)
                
                # Correlation coefficient
                correlation = spatial_df['Depth_km'].corr(spatial_df['Magnitude_ML'])
                st.metric("Depth-Magnitude Correlation", f"{correlation:.3f}")
                
                # Download option
                csv = spatial_df.to_csv(index=False)
                st.download_button(
                    label="Download Spatial Data as CSV",
                    data=csv,
                    file_name=f"spatial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Interactive map using plotly
                st.subheader("Geographic Distribution of Major Earthquakes")
                
                # Filter for better visualization
                map_data = spatial_df[spatial_df['Magnitude_ML'] >= 3.0].head(500)
                
                fig_map = px.scatter_mapbox(
                    map_data,
                    lat="Latitude",
                    lon="Longitude",
                    color="Magnitude_ML",
                    size="Magnitude_ML",
                    hover_data=['Depth_km', 'Nearest_Fault', 'Date'],
                    color_continuous_scale="Viridis",
                    mapbox_style="open-street-map",
                    zoom=8,
                    height=600,
                    title="Earthquake Distribution in Marmara Region"
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
            
            else:
                st.warning("No spatial data available")
        
        # Fault analysis
        if st.button("Analyze Fault Characteristics"):
            with st.spinner("Analyzing fault data..."):
                fault_df, message = st.session_state.analyzer.get_fault_analysis()
            
            if fault_df is not None and len(fault_df) > 0:
                st.subheader("Fault Analysis Summary")
                st.dataframe(fault_df)
                
                # Download option
                csv = fault_df.to_csv(index=False)
                st.download_button(
                    label="Download Fault Analysis as CSV",
                    data=csv,
                    file_name=f"fault_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Bar chart of earthquakes by fault
                fig, ax = plt.subplots(figsize=(12, 6))
                top_faults = fault_df.head(15)
                bars = ax.bar(range(len(top_faults)), top_faults['total_earthquakes'])
                ax.set_title('Earthquake Count by Fault (Top 15)')
                ax.set_xlabel('Fault Index')
                ax.set_ylabel('Total Earthquakes')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
                
                st.pyplot(fig)
                
                # Most active fault
                most_active_fault = fault_df.iloc[0]['Nearest_Fault']
                st.success(f"Most Active Fault: {most_active_fault} ({fault_df.iloc[0]['total_earthquakes']} earthquakes)")
            
            else:
                st.warning("No fault data available")

else:
    st.info("👈 Please connect to BigQuery using the sidebar")
    st.markdown("""
    **To connect:**
    1. Upload your service account JSON file, OR
    2. Use 'Cloud Run Mode' if running on GCP
    
    **Data Source:** Marmara Region Earthquake Database (2000-2025)
    - **Project:** tugas-week-13
    - **Dataset:** seismic_analysis  
    - **Table:** gempa_marmara
    """)

# Footer
st.markdown("---")
st.markdown("**Marmara Seismic Analysis Dashboard** | Built with Streamlit & BigQuery")
