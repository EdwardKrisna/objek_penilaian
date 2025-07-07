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
from agents import Agent, function_tool, Runner, set_default_openai_key

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="RHR AI Agents App",
    page_icon="🤖",
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
        """
        Geocode an address to get latitude and longitude
        Returns: (latitude, longitude, formatted_address) or (None, None, None) if failed
        """
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

# Agent Tools
@function_tool
def create_map_visualization(sql_query: str, title: str = "Property Locations") -> str:
    """Create a map visualization of properties from database query results"""
    try:
        # Execute query
        result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
        
        if result_df is None or len(result_df) == 0:
            return f"Error: No data returned from query - {query_msg}"
        
        # Check if data has required columns
        if 'latitude' not in result_df.columns or 'longitude' not in result_df.columns:
            return "Error: Query results must include 'latitude' and 'longitude' columns"
        
        # Clean coordinates
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
            if 'id' in row:
                text_parts.append(f"ID: {row['id']}")
            if 'nama_objek' in row:
                text_parts.append(f"Objek: {row['nama_objek']}")
            if 'pemberi_tugas' in row:
                text_parts.append(f"Client: {row['pemberi_tugas']}")
            if 'wadmpr' in row:
                text_parts.append(f"Provinsi: {row['wadmpr']}")
            if 'wadmkk' in row:
                text_parts.append(f"Kab/Kota: {row['wadmkk']}")
            
            hover_text.append("<br>".join(text_parts))
        
        # Add markers
        fig.add_trace(go.Scattermapbox(
            lat=map_df['latitude'],
            lon=map_df['longitude'],
            mode='markers',
            marker=dict(size=8, color='red'),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Properties'
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
        
        # Display map in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Store for future reference
        st.session_state.last_query_result = map_df.copy()
        st.session_state.last_map_data = map_df.copy()
        
        return f"✅ Map successfully created with {len(map_df)} properties"
        
    except Exception as e:
        return f"Error creating map: {str(e)}"

@function_tool
def create_chart_visualization(chart_type: str, sql_query: str, title: str, 
                              x_column: str = None, y_column: str = None, 
                              color_column: str = None) -> str:
    """Create chart visualizations from database query results"""
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
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Store for future reference
            st.session_state.last_query_result = result_df.copy()
            
            return f"✅ {chart_type.title()} chart successfully created with {len(result_df)} data points"
        else:
            return "Error: Failed to create chart"
            
    except Exception as e:
        return f"Error creating chart: {str(e)}"

@function_tool
def find_nearby_projects(location_name: str, radius_km: float = 1.0, 
                        title: str = "Nearby Projects") -> str:
    """Find and map projects near a specific location"""
    try:
        if not hasattr(st.session_state, 'geocode_service') or st.session_state.geocode_service is None:
            return "Error: Geocoding service not available. Please add Google Maps API key."
        
        # Geocode the location
        lat, lng, formatted_address = st.session_state.geocode_service.geocode_address(location_name)
        
        if lat is None or lng is None:
            return f"Error: Could not find coordinates for location '{location_name}'"
        
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
                name='Properties'
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
                title=f"{title} - {len(result_df)} projects within {radius_km} km from {location_name}"
            )
            
            # Display map
            st.plotly_chart(fig, use_container_width=True)
            
            # Store for future reference
            st.session_state.last_query_result = result_df.copy()
            st.session_state.last_map_data = result_df.copy()
            
            return f"✅ Found {len(result_df)} projects within {radius_km} km from {location_name}. Closest project is {result_df['distance_km'].min():.2f} km away."
        
        else:
            return f"❌ No projects found within {radius_km} km from {location_name}."
        
    except Exception as e:
        return f"Error finding nearby projects: {str(e)}"

def initialize_agents():
    """Initialize the agents"""
    
    # Set OpenAI API key for agents
    try:
        openai_api_key = st.secrets["openai"]["api_key"]
        set_default_openai_key(openai_api_key)
    except KeyError:
        st.error("OpenAI API key not found in secrets.toml")
        return None, None
    
    # Get table name from secrets
    try:
        table_name = st.secrets["database"]["table_name"]
    except KeyError:
        st.error("Table name not found in secrets.toml")
        return None, None
    
    # SQL Agent
    sql_agent = Agent(
        name="sql_agent",
        instructions=f"""You are a PostgreSQL expert for the RHR property appraisal database.

TABLE: {table_name}  

IMPORTANT: The table name is {table_name} - NEVER use "properties" or any other table name!

COLUMNS:
- id (int8): Unique project identifier - PRIMARY KEY
- sumber (text): Data source (e.g., "kontrak")
- pemberi_tugas (text): Client name (e.g., "PT Asuransi Jiwa IFG")
- no_kontrak (text): Contract number
- nama_lokasi (text): Location name
- alamat_lokasi (text): Address detail
- objek_penilaian (text): Appraisal object type
- nama_objek (text): Object name (e.g., "Rumah", "Hotel")
- jenis_objek_text (text): Object type
- kepemilikan (text): Ownership type
- keterangan (text): Additional notes
- penilaian_ke (text): Assessment sequence
- penugasan_text (text): Task type
- tujuan_text (text): Purpose
- status_text (text): Project status
- cabang_text (text): Branch name
- jc_text (text): Job captain
- latitude (float8): Latitude coordinates
- longitude (float8): Longitude coordinates
- geometry (geometry): PostGIS geometry
- wadmpr (text): Province
- wadmkk (text): Regency/City
- wadmkc (text): District

LOCATION MAPPING RULES:
- "bandung" -> wadmkk ILIKE '%bandung%'
- "jakarta" -> (wadmpr ILIKE '%jakarta%' OR wadmkk ILIKE '%jakarta%')
- "surabaya" -> wadmkk ILIKE '%surabaya%'

QUERY RULES:
1. ALWAYS use table name: {table_name}
2. Filter NULL values: column IS NOT NULL AND column != '' AND column != 'NULL'
3. For maps: SELECT id, latitude, longitude, nama_objek, pemberi_tugas, wadmpr, wadmkk FROM {table_name} WHERE...
4. For counting: SELECT COUNT(*) FROM {table_name} WHERE...
5. Location search: WHERE (wadmpr ILIKE '%location%' OR wadmkk ILIKE '%location%' OR wadmkc ILIKE '%location%')
6. Always include valid coordinates for maps: latitude IS NOT NULL AND longitude IS NOT NULL AND latitude != 0 AND longitude != 0
7. Use LIMIT to prevent large results

EXAMPLES:
- "berapa proyek di bandung" -> SELECT COUNT(*) FROM {table_name} WHERE wadmkk ILIKE '%bandung%'
- "peta proyek bandung" -> SELECT id, latitude, longitude, nama_objek, pemberi_tugas, wadmpr, wadmkk FROM {table_name} WHERE wadmkk ILIKE '%bandung%' AND latitude IS NOT NULL AND longitude IS NOT NULL

Return ONLY the PostgreSQL query, no explanations.""",
        model="o4-mini"
    )
    
    # Orchestrator Agent
    orchestrator_agent = Agent(
        name="orchestrator_agent",
        instructions="""You are RHR's AI assistant for property appraisal data analysis.

You help users with:
- Data queries and analysis
- Map visualizations 
- Chart creation
- Finding nearby projects

DECISION LOGIC - VERY IMPORTANT:
1. If user asks for "peta" or "map" (like "buatkan peta", "tampilkan peta", "peta proyek bandung"):
   -> ALWAYS use create_map_visualization tool with appropriate SQL query
   
2. If user asks for "grafik" or "chart":
   -> use create_chart_visualization tool

3. If user asks for "terdekat" or "nearby":
   -> use find_nearby_projects tool

4. For simple data questions (berapa, siapa, apa):
   -> use sql_query_builder tool

CONTEXT HANDLING:
- Remember previous conversations in this session
- If user says "buatkan petanya" after asking "berapa proyek di bandung", 
  create map for Bandung projects
- Connect follow-up requests to previous context

RESPONSE STYLE:
- Always respond in friendly Bahasa Indonesia
- Provide business insights, not just technical details
- Be conversational and helpful
- When creating maps/charts, explain what you're showing

EXAMPLE FLOWS:
User: "berapa proyek di bandung" 
-> Use sql_query_builder: "SELECT COUNT(*) FROM table WHERE wadmkk ILIKE '%bandung%'"

User: "buatkan petanya"
-> Use create_map_visualization with SQL for Bandung projects

CRITICAL: When user asks for maps, ALWAYS use the create_map_visualization tool, not sql_query_builder.""",
        model="gpt-4.1-mini",
        tools=[
            sql_agent.as_tool(
                tool_name="sql_query_builder",
                tool_description="Build SQL queries to retrieve data from the RHR property database. Use for counting, listing, or getting raw data."
            ),
            create_map_visualization,
            create_chart_visualization,
            find_nearby_projects
        ]
    )
    
    return sql_agent, orchestrator_agent

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def login():
    """Handle user login"""
    st.markdown('<div class="section-header">Login</div>', unsafe_allow_html=True)
    
    try:
        valid_username = st.secrets["auth"]["username"]
        valid_password = st.secrets["auth"]["password"]
    except KeyError:
        st.error("Authentication credentials not found in secrets.toml")
        return False
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username == valid_username and password == valid_password:
                st.session_state.authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
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

async def process_user_query(query: str, orchestrator_agent: Agent) -> str:
    """Process user query using the orchestrator agent"""
    try:
        result = await Runner.run(orchestrator_agent, input=query)
        return result.final_output
    except Exception as e:
        return f"Error processing query: {str(e)}"

def render_ai_chat():
    """Render AI chat interface using agents"""
    st.markdown('<div class="section-header">AI Chat with Agents</div>', unsafe_allow_html=True)
    
    if not initialize_database():
        return
    
    # Initialize geocoding service
    geocode_service = initialize_geocode_service()
    
    # Initialize agents
    sql_agent, orchestrator_agent = initialize_agents()
    if not orchestrator_agent:
        return
    
    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        welcome_msg = """Halo! Saya asisten AI RHR menggunakan sistem agents 🤖

Saya dapat membantu dengan:
- **Analisis Data**: "Berapa proyek di Jakarta?" 
- **Visualisasi Peta**: "Tampilkan peta semua proyek"
- **Grafik**: "Buatkan grafik pemberi tugas"
- **Pencarian Lokasi**: "Proyek terdekat dari Mall Taman Anggrek"

Apa yang ingin Anda ketahui?"""
        
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": welcome_msg
        })
    
    # Status indicators
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.db_connection.connection_status:
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Disconnected")
    
    with col2:
        if geocode_service:
            st.success("✅ Location Service Active")
        else:
            st.warning("⚠️ Location Service Inactive")
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your property data..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response using agents
        with st.chat_message("assistant"):
            with st.spinner("Processing with AI agents..."):
                # Run the async function
                try:
                    # Create a new event loop for this request
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    response = loop.run_until_complete(
                        process_user_query(prompt, orchestrator_agent)
                    )
                    
                    loop.close()
                    
                    st.markdown(response)
                    
                    # Add to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Chat controls
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            if 'last_query_result' in st.session_state:
                del st.session_state.last_query_result
            if 'last_map_data' in st.session_state:
                del st.session_state.last_map_data
            st.rerun()
    
    with col2:
        if st.button("Show Last Query", use_container_width=True):
            if hasattr(st.session_state, 'last_query_result') and st.session_state.last_query_result is not None:
                st.dataframe(st.session_state.last_query_result, use_container_width=True)
            else:
                st.info("No previous query results available")
    
    with col3:
        if st.button("Export Chat", use_container_width=True):
            chat_export = {
                "timestamp": datetime.now().isoformat(),
                "chat_messages": st.session_state.chat_messages
            }
            
            st.download_button(
                label="Download",
                data=json.dumps(chat_export, indent=2, ensure_ascii=False),
                file_name=f"agents_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">RHR AI Agents Assistant</h1>', unsafe_allow_html=True)
    
    # Check authentication
    if not check_authentication():
        login()
        return
    
    # Show current user
    st.sidebar.title("RHR AI Agents")
    st.sidebar.success(f"Logged in as: {st.secrets['auth']['username']}")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # Render AI chat
    render_ai_chat()
    
    # Sidebar system status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    
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
    
    # Agent info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**AI Agents**")
    st.sidebar.info("🤖 SQL Agent: Query Builder")
    st.sidebar.info("🎯 Orchestrator: Main Assistant")
    st.sidebar.info("🛠️ Tools: Map, Chart, Location")

if __name__ == "__main__":
    main()