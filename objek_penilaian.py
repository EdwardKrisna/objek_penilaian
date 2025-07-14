import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import json
import traceback
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
import requests
import math
import asyncio
import re
from agents import Agent, function_tool, Runner, set_default_openai_key
from openai.types.responses import ResponseTextDeltaEvent

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="RHR AI Agent",
    page_icon=None,
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
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #efe;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #fee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e74c3c;
        margin: 1rem 0;
    }
    .agent-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class DatabaseConnection:
    """Handle PostgreSQL database connections"""
    
    def __init__(self):
        self.engine = None
        self.connection_status = False
    
    def connect(self, db_user: str, db_pass: str, db_host: str, 
                db_port: str, db_name: str, schema: str = 'public'):
        """Establish database connection"""
        try:
            connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
            self.engine = create_engine(connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.connection_status = True
            return True, "Connection successful!"
        
        except Exception as e:
            self.connection_status = False
            return False, f"Connection failed: {str(e)}"
    
    def execute_query(self, query: str) -> tuple:
        """Execute SQL query and return results"""
        try:
            if not self.connection_status:
                return None, "No database connection established"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                
                rows = []
                columns = list(result.keys())
                
                for row in result:
                    row_dict = {}
                    for i, value in enumerate(row):
                        row_dict[columns[i]] = value
                    rows.append(row_dict)
                
                if rows:
                    df = pd.DataFrame(rows)
                else:
                    df = pd.DataFrame(columns=columns)
                
                return df, "Query executed successfully"
        
        except Exception as e:
            return None, f"Query execution failed: {str(e)}"

class GeocodeService:
    """Handle geocoding using Google Maps Geocoding API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    def geocode_address(self, address: str) -> tuple:
        """Geocode an address to get latitude and longitude"""
        try:
            params = {
                'address': address,
                'key': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                result = data['results'][0]
                location = result['geometry']['location']
                formatted_address = result['formatted_address']
                
                return (
                    location['lat'],
                    location['lng'],
                    formatted_address
                )
            else:
                return None, None, None
                
        except Exception as e:
            st.error(f"Geocoding error: {str(e)}")
            return None, None, None

# Core Agent Tools
@function_tool
def execute_sql_query(sql_query: str) -> str:
    """Execute SQL query and return formatted results"""
    try:
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is not None and len(result_df) > 0:
            # Store for future reference
            st.session_state.last_query_result = result_df.copy()
            
            # Display results in expandable section
            with st.expander("📊 Query Results", expanded=False):
                st.code(sql_query, language="sql")
                st.dataframe(result_df, use_container_width=True)
            
            # Return formatted summary
            if len(result_df) == 1 and len(result_df.columns) == 1:
                # Single value result (like COUNT)
                value = result_df.iloc[0, 0]
                return f"Query result: {value}"
            elif len(result_df) <= 10:
                # Small result - show all
                return f"Query returned {len(result_df)} rows:\n{result_df.to_string(index=False)}"
            else:
                # Large result - show summary
                return f"Query returned {len(result_df)} rows. Data displayed in expandable section above."
        else:
            return f"No results returned: {query_msg}"
            
    except Exception as e:
        return f"SQL Error: {str(e)}"

@function_tool
def create_map_visualization(sql_query: str, title: str = "Property Locations") -> str:
    """Create a map visualization from SQL query results"""
    try:
        # Execute query
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is None or len(result_df) == 0:
            return f"Error: No data returned from query - {query_msg}"
        
        # Check required columns
        if 'latitude' not in result_df.columns or 'longitude' not in result_df.columns:
            return "Error: Query must include 'latitude' and 'longitude' columns for map visualization"
        
        # Clean and filter coordinates
        map_df = result_df.copy()
        map_df = map_df.dropna(subset=['latitude', 'longitude'])
        map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
        map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
        
        # Filter valid coordinates
        map_df = map_df[
            (map_df['latitude'] >= -90) & (map_df['latitude'] <= 90) &
            (map_df['longitude'] >= -180) & (map_df['longitude'] <= 180) &
            (map_df['latitude'] != 0) & (map_df['longitude'] != 0)
        ]
        
        if len(map_df) == 0:
            return "Error: No valid coordinates found in the data"
        
        # Create map
        fig = go.Figure()
        
        # Create hover text
        hover_text = []
        for idx, row in map_df.iterrows():
            text_parts = []
            for col in ['id', 'nama_objek', 'pemberi_tugas', 'wadmpr', 'wadmkk']:
                if col in row and pd.notna(row[col]):
                    label = {'id': 'ID', 'nama_objek': 'Objek', 'pemberi_tugas': 'Client', 
                            'wadmpr': 'Provinsi', 'wadmkk': 'Kab/Kota'}.get(col, col)
                    text_parts.append(f"{label}: {row[col]}")
            
            if 'distance_km' in row:
                text_parts.append(f"Jarak: {row['distance_km']:.2f} km")
            
            hover_text.append("<br>".join(text_parts))
        
        # Add markers
        fig.add_trace(go.Scattermapbox(
            lat=map_df['latitude'],
            lon=map_df['longitude'],
            mode='markers',
            marker=dict(size=8, color='red'),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name=f'Properties ({len(map_df)})'
        ))
        
        # Calculate center
        center_lat = map_df['latitude'].mean()
        center_lon = map_df['longitude'].mean()
        
        # Map layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=8
            ),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            title=title
        )
        
        # Store the figure in session state for persistence
        st.session_state.last_visualization = {
            "type": "map",
            "figure": fig,
            "title": title
        }

        # Display map
        st.plotly_chart(fig, use_container_width=True)
        
        # Store for future reference
        st.session_state.last_query_result = map_df.copy()
        st.session_state.last_map_data = map_df.copy()
        
        # Show query details
        with st.expander("🗺️ Map Query Details", expanded=False):
            st.code(sql_query, language="sql")
            st.info(f"Mapped {len(map_df)} properties with valid coordinates")
        
        return f"✅ Map successfully created with {len(map_df)} properties"
        
    except Exception as e:
        return f"Error creating map: {str(e)}"

@function_tool
def create_chart_visualization(chart_type: str, sql_query: str, title: str, 
                              x_column: str = None, y_column: str = None, 
                              color_column: str = None) -> str:
    """Create chart visualizations from SQL query results"""
    try:
        # Execute query
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is None or len(result_df) == 0:
            return f"Error: No data returned from query - {query_msg}"
        
        # Auto-detect columns if not provided
        if x_column is None or x_column not in result_df.columns:
            x_column = result_df.columns[0]
        if y_column is None or y_column not in result_df.columns:
            numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
            y_column = numeric_cols[0] if numeric_cols else result_df.columns[1] if len(result_df.columns) > 1 else None
        
        fig = None
        
        # Create chart based on type
        if chart_type == "bar":
            fig = px.bar(result_df, x=x_column, y=y_column, color=color_column, title=title)
            fig.update_layout(xaxis_tickangle=-45)
        elif chart_type == "pie":
            if y_column:
                fig = px.pie(result_df, names=x_column, values=y_column, title=title)
            else:
                pie_data = result_df[x_column].value_counts().reset_index()
                pie_data.columns = [x_column, 'count']
                fig = px.pie(pie_data, names=x_column, values='count', title=title)
        elif chart_type == "line":
            fig = px.line(result_df, x=x_column, y=y_column, color=color_column, title=title, markers=True)
        elif chart_type == "scatter":
            fig = px.scatter(result_df, x=x_column, y=y_column, color=color_column, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(result_df, x=x_column if x_column else y_column, color=color_column, title=title)
        else:
            # Default to bar chart
            fig = px.bar(result_df, x=x_column, y=y_column, color=color_column, title=title)
            fig.update_layout(xaxis_tickangle=-45)
        
        if fig:
            fig.update_layout(
                height=500,
                template="plotly_white",
                title_x=0.5,
                margin=dict(l=50, r=50, t=80, b=100)
            )
        
        if fig:
            # Store the figure in session state for persistence
            st.session_state.last_visualization = {
                "type": "chart",
                "figure": fig,
                "chart_type": chart_type,
                "title": title
            }
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Store for future reference
            st.session_state.last_query_result = result_df.copy()
            
            # Show query details
            with st.expander("📊 Chart Query Details", expanded=False):
                st.code(sql_query, language="sql")
                st.dataframe(result_df, use_container_width=True)
            
            return f"✅ {chart_type.title()} chart successfully created with {len(result_df)} data points"
        else:
            return "Error: Failed to create chart"
            
    except Exception as e:
        return f"Error creating chart: {str(e)}"

@function_tool
def find_nearby_projects(location_name: str, radius_km: float = 1.0, 
                        title: str = None) -> str:
    """Find and map projects near a specific location using geocoding"""
    try:
        if not hasattr(st.session_state, 'geocode_service') or st.session_state.geocode_service is None:
            return "Error: Geocoding service not available. Please add Google Maps API key."
        
        # Set default title
        if title is None:
            title = f"Projects within {radius_km} km from {location_name}"
        
        # Geocode the location
        lat, lng, formatted_address = st.session_state.geocode_service.geocode_address(location_name)
        
        if lat is None or lng is None:
            return f"Error: Could not find coordinates for location '{location_name}'. Try being more specific."
        
        st.success(f"📍 Location found: {formatted_address}")
        st.info(f"Coordinates: {lat:.6f}, {lng:.6f}")
        
        # Get table name
        table_name = st.secrets["database"]["table_name"]
        
        # Query nearby projects using Haversine formula
        sql_query = f"""
        SELECT 
            id,
            nama_objek,
            pemberi_tugas,
            latitude,
            longitude,
            wadmpr,
            wadmkk,
            wadmkc,
            jenis_objek_text,
            status_text,
            cabang_text,
            (6371 * acos(
                cos(radians({lat})) * cos(radians(latitude)) * 
                cos(radians(longitude) - radians({lng})) + 
                sin(radians({lat})) * sin(radians(latitude))
            )) as distance_km
        FROM {table_name}
        WHERE 
            latitude IS NOT NULL 
            AND longitude IS NOT NULL
            AND latitude != 0 
            AND longitude != 0
            AND (6371 * acos(
                cos(radians({lat})) * cos(radians(latitude)) * 
                cos(radians(longitude) - radians({lng})) + 
                sin(radians({lat})) * sin(radians(latitude))
            )) <= {radius_km}
        ORDER BY distance_km ASC
        LIMIT 50
        """
        
        # Execute query
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is not None and len(result_df) > 0:
            # Create enhanced map with reference point
            fig = go.Figure()
            
            # Add reference point (target location)
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lng],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='star'),
                text=[f"📍 {location_name}<br>{formatted_address}"],
                hovertemplate='%{text}<extra></extra>',
                name='Target Location'
            ))
            
            # Add project markers
            hover_text = []
            for idx, row in result_df.iterrows():
                text_parts = [
                    f"ID: {row['id']}",
                    f"Objek: {row['nama_objek']}",
                    f"Client: {row['pemberi_tugas']}",
                    f"Provinsi: {row['wadmpr']}",
                    f"Kab/Kota: {row['wadmkk']}",
                    f"Jarak: {row['distance_km']:.2f} km"
                ]
                hover_text.append("<br>".join(text_parts))
            
            fig.add_trace(go.Scattermapbox(
                lat=result_df['latitude'],
                lon=result_df['longitude'],
                mode='markers',
                marker=dict(size=8, color='red'),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                name=f'Properties ({len(result_df)})'
            ))
            
            # Map layout centered on target location
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=lat, lon=lng),
                    zoom=12
                ),
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                title=title
            )
            
            # Store the figure in session state for persistence
            st.session_state.last_visualization = {
                "type": "nearby_map",
                "figure": fig,
                "title": title,
                "location": location_name,
                "radius": radius_km,
                "count": len(result_df)
            }

            # Display map
            st.plotly_chart(fig, use_container_width=True)

            # Store for future reference
            st.session_state.last_query_result = result_df.copy()
            st.session_state.last_map_data = result_df.copy()
            
            # Show results table
            with st.expander("📊 Nearby Projects Details", expanded=False):
                st.code(sql_query, language="sql")
                st.dataframe(result_df[['id', 'nama_objek', 'pemberi_tugas', 'jenis_objek_text', 
                            'wadmpr', 'wadmkk', 'distance_km']].round(2), 
                use_container_width=True)

            return f"✅ Found {len(result_df)} projects within {radius_km} km from {location_name}. Closest project is {result_df['distance_km'].min():.2f} km away."
        
        else:
            return f"❌ No projects found within {radius_km} km from {location_name}."
        
    except Exception as e:
        return f"Error finding nearby projects: {str(e)}"

def initialize_main_agent():
    """Initialize the single o4-mini agent that handles everything"""
    
    # Set OpenAI API key
    try:
        openai_api_key = st.secrets["openai"]["api_key"]
        set_default_openai_key(openai_api_key)
    except KeyError:
        st.error("OpenAI API key not found in secrets.toml")
        return None
    
    # Get table name from secrets
    try:
        table_name = st.secrets["database"]["table_name"]
    except KeyError:
        st.error("Table name not found in secrets.toml")
        return None
    
    # Single Unified Agent using o4-mini
    main_agent = Agent(
        name="rhr_assistant",
        instructions=f"""You are RHR's comprehensive AI assistant for property appraisal analysis. You handle everything: conversation, SQL queries, visualizations, and business insights.

**DATABASE SCHEMA - TABLE: {table_name}**

**Project Information:**
- sumber (text): Data source (e.g., "kontrak")
- pemberi_tugas (text): Client/Task giver (e.g., "PT Asuransi Jiwa IFG")
- no_kontrak (text): Contract number
- nama_lokasi (text): Location name
- alamat_lokasi (text): Address detail
- id (int8): Unique project identifier (PRIMARY KEY)

**Property Information:**
- objek_penilaian (text): Appraisal object type
- nama_objek (text): Object name (e.g., "Rumah", "Tanah Kosong")
- jenis_objek_text (text): Object type (e.g., "Hotel", "Aset Tak Berwujud")
- kepemilikan (text): Ownership type
- keterangan (text): Additional notes

**Project Management:**
- penilaian_ke (text): Assessment count
- penugasan_text (text): Task type
- tujuan_text (text): Purpose
- status_text (text): Project status
- cabang_text (text): Branch name
- jc_text (text): Job captain

**Geographic Data:**
- latitude (float8): Latitude coordinates
- longitude (float8): Longitude coordinates
- geometry (geometry): PostGIS geometry field
- wadmpr (text): Province (e.g., "DKI Jakarta")
- wadmkk (text): Regency/City (e.g., "Jakarta Selatan")
- wadmkc (text): District (e.g., "Tebet")

**AVAILABLE TOOLS:**
1. `execute_sql_query(sql_query)` - Run SQL queries and display results
2. `create_map_visualization(sql_query, title)` - Create location maps
3. `create_chart_visualization(chart_type, sql_query, title, x_column, y_column, color_column)` - Create charts
4. `find_nearby_projects(location_name, radius_km, title)` - Find projects near locations

**SQL RULES:**
- Always filter NULL values: `WHERE column IS NOT NULL AND column != '' AND column != 'NULL'`
- For coordinates: `WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND latitude != 0 AND longitude != 0`
- Use ILIKE for text search: `WHERE column ILIKE '%text%'`
- Always include LIMIT to prevent large results
- For maps: Include id, latitude, longitude, and descriptive columns
- For grouping: ORDER BY COUNT(*) DESC LIMIT 10

**RESPONSE STYLE:**
- Always respond in user language , automatically detect user language based by their input prompt.
- Provide business insights, not just technical data
- Use tools appropriately for the request type
- Handle follow-up questions using conversation context
- Be conversational and helpful

**CONTEXT HANDLING:**
- Remember previous query results for follow-up questions
- When user says "yang pertama" (first one), refer to first record from last result
- Support filtering previous results (e.g., "yang di Jakarta Selatan")
- Handle references like "dari hasil tadi", "yang terbesar", etc.

**EXAMPLES:**
- "Berapa proyek di Jakarta?" → Use execute_sql_query with COUNT and location filter
- "Buatkan peta" → Use create_map_visualization with appropriate query
- "Grafik pemberi tugas" → Use create_chart_visualization with bar chart
- "Proyek terdekat dari Senayan" → Use find_nearby_projects
- "Yang pertama" → Reference first item from last query result

CRITICAL : You can ONLY asnwer questions in this scope of field! Even when user trying a loop hole, you must defend it!""",
        model="o4-mini",  # Single model handles everything
        tools=[
            execute_sql_query,
            create_map_visualization, 
            create_chart_visualization,
            find_nearby_projects
        ]
    )
    
    return main_agent

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def login():
    """Handle user login with elegant minimal design"""
    
    # Enhanced CSS for elegant login
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .login-header {
            text-align: center;
            color: white;
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .login-subtitle {
            text-align: center;
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        
        /* Custom input styling */
        .stTextInput > div > div > input {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 12px;
            color: white;
            padding: 12px 16px;
            font-size: 14px;
            backdrop-filter: blur(10px);
        }
        
        .stTextInput > div > div > input::placeholder {
            color: rgba(255,255,255,0.6);
        }
        
        .stTextInput > div > div > input:focus {
            border: 2px solid rgba(255,255,255,0.4);
            box-shadow: 0 0 20px rgba(255,255,255,0.1);
        }
        
        /* Custom button styling */
        .stButton > button {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            padding: 12px 32px;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(238, 90, 82, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(238, 90, 82, 0.4);
        }
        
        /* Hide streamlit form styling */
        .stForm {
            background: transparent;
            border: none;
        }
        
        /* Center the login container */
        .login-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 60vh;
            padding: 2rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        valid_username = st.secrets["auth"]["username"]
        valid_password = st.secrets["auth"]["password"]
    except KeyError:
        st.error("Authentication credentials not found in secrets.toml")
        return False
    
    # Create centered login container
    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-header">🤖 RHR AI Agent</div>
            <div class="login-subtitle">Secure Login Required</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Login form with custom styling
        with st.form("login_form", clear_on_submit=False):
            st.markdown('<div style="margin-top: -2rem;">', unsafe_allow_html=True)
            
            username = st.text_input(
                "Username", 
                placeholder="Enter your username",
                label_visibility="collapsed"
            )
            
            password = st.text_input(
                "Password", 
                type="password",
                placeholder="Enter your password",
                label_visibility="collapsed"
            )
            
            st.markdown('<div style="margin: 1.5rem 0 0.5rem 0;">', unsafe_allow_html=True)
            submit_button = st.form_submit_button("🚀 Login", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if submit_button:
                if username == valid_username and password == valid_password:
                    st.session_state.authenticated = True
                    st.success("✅ Login successful!")
                    st.balloons()  # Add celebration effect
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return False

def initialize_database():
    """Initialize database connection"""
    if 'db_connection' not in st.session_state:
        st.session_state.db_connection = DatabaseConnection()
    
    if not st.session_state.db_connection.connection_status:
        try:
            db_user = st.secrets["database"]["user"]
            db_pass = st.secrets["database"]["password"]
            db_host = st.secrets["database"]["host"]
            db_port = st.secrets["database"]["port"]
            db_name = st.secrets["database"]["name"]
            schema = st.secrets["database"]["schema"]
            
            success, message = st.session_state.db_connection.connect(
                db_user, db_pass, db_host, db_port, db_name, schema
            )
            
            if success:
                st.session_state.schema = schema
                return True
            else:
                st.error(f"Database connection failed: {message}")
                return False
                
        except KeyError as e:
            st.error(f"Missing database configuration: {e}")
            return False
    
    return True

def initialize_geocode_service():
    """Initialize geocoding service"""
    try:
        google_api_key = st.secrets["google"]["api_key"]
        if 'geocode_service' not in st.session_state:
            st.session_state.geocode_service = GeocodeService(google_api_key)
        return st.session_state.geocode_service
    except KeyError:
        st.warning("Google Maps API key not found. Location search features unavailable.")
        st.session_state.geocode_service = None
        return None

async def process_user_query(query: str, main_agent: Agent) -> str:
    """Process user query with the unified o4-mini agent"""
    try:
        # Build conversation context
        conversation_context = ""
        if hasattr(st.session_state, 'chat_messages') and len(st.session_state.chat_messages) > 1:
            recent_messages = st.session_state.chat_messages[-4:]
            context_parts = []
            
            for msg in recent_messages:
                if msg['role'] == 'user':
                    context_parts.append(f"User: {msg['content']}")
                elif msg['role'] == 'assistant' and len(msg['content']) < 200:
                    context_parts.append(f"Assistant: {msg['content']}")
            
            if context_parts:
                conversation_context = "\n".join(context_parts)
        
        # Enhanced query with context
        enhanced_query = query
        if conversation_context:
            enhanced_query = f"""CONVERSATION CONTEXT:
{conversation_context}

CURRENT REQUEST: {query}

Use context appropriately for follow-up questions."""
        
        # Streaming response with o4-mini
        response_container = st.empty()
        full_response = ""
        
        # Run agent with streaming
        result = Runner.run_streamed(main_agent, input=enhanced_query)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                full_response += event.data.delta
                response_container.markdown(full_response + "▌")
        
        response_container.markdown(full_response)
        return full_response
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        st.error(error_msg)
        return error_msg

def render_ai_chat():
    """Render the streamlined AI chat interface"""
    # st.markdown('<div class="section-header">RHR AI Agent </div>', unsafe_allow_html=True)
    
    if not initialize_database():
        return
    
    # Initialize services
    geocode_service = initialize_geocode_service()
    main_agent = initialize_main_agent()
    
    if not main_agent:
        return
    
    # Agent status display
    st.markdown("""
    <div class="agent-status">
        🤖 KJPP RHR FIRST AI AGENT - Handling All Tasks
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        welcome_msg = """Halo! Saya RHR AI Agent !

**Kemampuan Saya:**
- 💬 **Percakapan Natural**: Saya berbicara dalam bahasa yang anda gunakan!
- 📊 **Analisis Data**: "Berapa proyek di Jakarta?" ,"Siapa client terbesar?"
- 🗺️ **Visualisasi Peta**: "Buatkan peta semua proyek di Bali"
- 📈 **Grafik & Chart**: "Grafik pemberi tugas per cabang"
- 📍 **Pencarian Lokasi**: "Proyek terdekat dari Mall Taman Anggrek radius 1km"
- 🔄 **Follow-up Kontekstual**: "Yang pertama" • "Detail yang di Jakarta Selatan"

Apa yang ingin Anda ketahui tentang proyek properti RHR hari ini?"""
        
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": welcome_msg
        })
    
    # Display ALL existing chat messages FIRST
    for i, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Redisplay visualization if present
            if message.get("visualization"):
                viz = message["visualization"]
                if viz["type"] in ["map", "nearby_map"]:
                    st.plotly_chart(viz["figure"], use_container_width=True)
                    
                    # Show additional info for nearby maps
                    if viz["type"] == "nearby_map":
                        st.caption(f"📍 {viz['location']} • Radius: {viz['radius']} km • Found: {viz['count']} projects")
                        
                elif viz["type"] == "chart":
                    st.plotly_chart(viz["figure"], use_container_width=True)

    # Handle new user input LAST
    if prompt := st.chat_input("Tanya tentang data properti Anda..."):
        # Add user message to history IMMEDIATELY
        st.session_state.chat_messages.append({
            "role": "user", 
            "content": prompt, 
            "visualization": None
        })
        
        # Set a flag to process response on next run
        st.session_state.pending_response = prompt
        
        # Refresh immediately to show user input at bottom
        st.rerun()

    # Process pending response if exists
    if hasattr(st.session_state, 'pending_response') and st.session_state.pending_response:
        prompt = st.session_state.pending_response
        
        # Clear the pending flag
        del st.session_state.pending_response
        
        # Clear any previous visualization
        if 'last_visualization' in st.session_state:
            del st.session_state.last_visualization
        
        # Process assistant response
        with st.spinner("🤖 Processing..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                response = loop.run_until_complete(
                    process_user_query(prompt, main_agent)
                )
                
                loop.close()
                
                # Check if a visualization was created
                viz_data = st.session_state.get('last_visualization', None)
                
                # Add to chat history with visualization
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response,
                    "visualization": viz_data
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "visualization": None
                })
        
        # Refresh again to show response
        st.rerun()
    
    if st.button("📊 Show Last Data", use_container_width=True):
            if hasattr(st.session_state, 'last_query_result') and st.session_state.last_query_result is not None:
                with st.expander("Last Query Results", expanded=True):
                    st.dataframe(st.session_state.last_query_result, use_container_width=True)
            else:
                st.info("No previous query results available")

    # Chat controls
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            if 'last_query_result' in st.session_state:
                del st.session_state.last_query_result
            if 'last_map_data' in st.session_state:
                del st.session_state.last_map_data
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset Context", use_container_width=True):
            if 'last_query_result' in st.session_state:
                del st.session_state.last_query_result
            if 'last_map_data' in st.session_state:
                del st.session_state.last_map_data
            st.success("Context cleared!")
    
    with col3:
        if st.button("💾 Export Chat", use_container_width=True):
            chat_export = {
                "timestamp": datetime.now().isoformat(),
                "agent_model": "o4-mini",
                "chat_messages": st.session_state.chat_messages
            }
            
            st.download_button(
                label="Download",
                data=json.dumps(chat_export, indent=2, ensure_ascii=False),
                file_name=f"rhr_o4_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

def main():
    """Main application"""
    st.markdown('<h1 class="main-header"> RHR AI Agent </h1>', unsafe_allow_html=True)
    
    # Check authentication
    if not check_authentication():
        login()
        return
    
    # Sidebar navigation
    st.sidebar.title("RHR AI Agent")
    st.sidebar.success(f"Logged in as: {st.secrets['auth']['username']}")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    
    # Handle example query injection
    if hasattr(st.session_state, 'example_query'):
        st.info(f"Running example: {st.session_state.example_query}")
        # Add to chat messages
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        
        st.session_state.chat_messages.append({
            "role": "user", 
            "content": st.session_state.example_query
        })
        
        # Clear the example query
        del st.session_state.example_query
        st.rerun()
    
    render_ai_chat()
    
    # Sidebar system status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 System Status**")
    
    # Database status
    if hasattr(st.session_state, 'db_connection') and st.session_state.db_connection.connection_status:
        st.sidebar.success("🗄️ Database Connected")
    else:
        st.sidebar.error("❌ Database Disconnected")
    
    # Geocoding service status
    try:
        google_api_key = st.secrets["google"]["api_key"]
        st.sidebar.success("🌍 Geocoding Available")
    except KeyError:
        st.sidebar.warning("⚠️ Geocoding Unavailable")
    
    # Chat status
    if hasattr(st.session_state, 'chat_messages'):
        st.sidebar.info(f"💬 Messages: {len(st.session_state.chat_messages)}")


if __name__ == "__main__":
    main()
