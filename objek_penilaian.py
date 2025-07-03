import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from openai import OpenAI
import json
import traceback
from datetime import datetime
import warnings
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="RHR AI Query App",
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
            
            # Use connection context manager to avoid SQLAlchemy issues
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                
                # Convert result to list of dictionaries first
                rows = []
                columns = list(result.keys())
                
                for row in result:
                    # Convert each row to a regular dict
                    row_dict = {}
                    for i, value in enumerate(row):
                        row_dict[columns[i]] = value
                    rows.append(row_dict)
                
                # Create DataFrame from the cleaned data
                if rows:
                    df = pd.DataFrame(rows)
                else:
                    # Create empty DataFrame with proper columns
                    df = pd.DataFrame(columns=columns)
                
                return df, "Query executed successfully"
        
        except Exception as e:
            return None, f"Query execution failed: {str(e)}"
    
    def get_unique_geographic_values(self, column, parent_filter=None, table_name=None):
        """Get unique values for geographic columns with optional parent filtering"""
        try:
            if not table_name:
                return []
            
            base_query = f"SELECT DISTINCT {column} FROM {table_name} WHERE {column} IS NOT NULL"
            
            if parent_filter:
                if column == 'wadmkk' and 'wadmpr' in parent_filter:
                    provinces = parent_filter['wadmpr']
                    escaped_provinces = [p.replace("'", "''") for p in provinces]
                    province_list = "', '".join(escaped_provinces)
                    base_query += f" AND wadmpr IN ('{province_list}')"
                elif column == 'wadmkc' and 'wadmkk' in parent_filter:
                    regencies = parent_filter['wadmkk']
                    escaped_regencies = [r.replace("'", "''") for r in regencies]
                    regency_list = "', '".join(escaped_regencies)
                    base_query += f" AND wadmkk IN ('{regency_list}')"
            
            base_query += f" ORDER BY {column}"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(base_query))
                # Convert to list properly
                values = [row[0] for row in result.fetchall()]
                return values
        except Exception as e:
            st.error(f"Failed to load {column} options: {str(e)}")
            return []

class RHRAIChat:
    """AI chatbot for database queries using o4-mini for queries and GPT-4.1-mini for responses"""
    
    def __init__(self, api_key: str, table_name: str):
        # Single OpenAI client for both models
        self.client = OpenAI(api_key=api_key)
        self.table_name = table_name
    
    def generate_query(self, user_question: str, geographic_context: str = "") -> str:
        """Use o4-mini to generate SQL query with function calling support"""
        try:
            system_prompt = f"""You are a SQL query generator for RHR property appraisal database.

TABLE: {self.table_name}

DETAILED COLUMN INFORMATION:

Project Information:
- sumber (text): Data source (e.g., "kontrak" = contract-based projects)
- pemberi_tugas (text): Client/Task giver (e.g., "PT Asuransi Jiwa IFG", "PT Perkebunan Nusantara II")
- no_kontrak (text): Contract number (e.g., "RHR00C1P0623111.0")
- nama_lokasi (text): Location name (e.g., "Lokasi 20", "Lokasi 3")
- id (integer): Unique project identifier (e.g., 16316, 17122) - PRIMARY KEY

Property Information:
- objek_penilaian (text): Appraisal object type (e.g., "real properti")
- nama_objek (text): Object name (e.g., "Rumah", "Tanah Kosong")
- jenis_objek (integer): Object type code that joins with master_jenis_objek table for readable names
- kepemilikan (text): Ownership type (e.g., "tunggal" = single ownership)
- keterangan (text): Additional notes (e.g., "Luas Tanah : 1.148", may contain NULL)

Status & Management:
- status (integer): Project status code (4, 5, etc.)
- cabang (integer): Branch office code (0, 1, etc.)

Geographic Data:
- latitude (decimal): Latitude coordinates (e.g., -6.236507782741299)
- longitude (decimal): Longitude coordinates (e.g., 106.86356067983168)
- geometry (text): PostGIS geometry field (binary spatial data)
- wadmpr (text): Province (e.g., "DKI Jakarta", "Sumatera Utara")
- wadmkk (text): Regency/City (e.g., "Kota Administrasi Jakarta Selatan", "Deli Serdang")
- wadmkc (text): District (e.g., "Tebet", "Labuhan Deli")

CRITICAL SQL RULES:
1. For counting: SELECT COUNT(*) FROM {self.table_name} WHERE...
2. For samples: SELECT id, [columns] FROM {self.table_name} WHERE... ORDER BY id DESC LIMIT 5
3. For grouping: SELECT [column], COUNT(*) FROM {self.table_name} t LEFT JOIN master_jenis_objek m ON t.jenis_objek = m.id WHERE [column] IS NOT NULL GROUP BY [column] ORDER BY COUNT(*) DESC LIMIT 10
4. For object type analysis: Always use m.name instead of t.jenis_objek for readable results
5. Always handle NULLs: Use "WHERE column IS NOT NULL" when querying specific columns
6. Text search: Use "ILIKE '%text%'" for case-insensitive search
7. Geographic search: "(wadmpr ILIKE '%location%' OR wadmkk ILIKE '%location%' OR wadmkc ILIKE '%location%')"
8. Always add LIMIT to prevent large result sets
9. For map visualization: ALWAYS include id, latitude, longitude, and descriptive columns (nama_objek, pemberi_tugas, wadmpr, wadmkk)
10. For readable object types: Always JOIN with master_jenis_objek to get names instead of codes
11. Use LEFT JOIN master_jenis_objek m ON t.jenis_objek = m.id
12. Select m.name as jenis_objek_name for readable output

SAMPLE DATA EXAMPLES:
Row 1: id=16316, pemberi_tugas="PT Asuransi Jiwa IFG", nama_objek="Rumah", jenis_objek=13, wadmpr="DKI Jakarta", wadmkk="Kota Administrasi Jakarta Selatan", wadmkc="Tebet"
Row 2: id=17122, pemberi_tugas="PT Perkebunan Nusantara II", nama_objek="Tanah Kosong", jenis_objek=1, wadmpr="Sumatera Utara", wadmkk="Deli Serdang", wadmkc="Labuhan Deli"

LOOKUP TABLES:
- master_jenis_objek: Contains id and name for jenis_objek codes
  Join: {self.table_name}.jenis_objek = master_jenis_objek.id

SQL JOIN EXAMPLES:
- Simple query with object type names: 
  SELECT t.id, t.nama_objek, m.name as jenis_objek_name 
  FROM {self.table_name} t 
  LEFT JOIN master_jenis_objek m ON t.jenis_objek = m.id

- Grouping by object type:
  SELECT m.name as object_type, COUNT(*) as count
  FROM {self.table_name} t 
  LEFT JOIN master_jenis_objek m ON t.jenis_objek = m.id
  WHERE m.name IS NOT NULL
  GROUP BY m.name ORDER BY count DESC

Generate ONLY the PostgreSQL query, no explanations."""

            # Define tools for function calling
            tools = [{
                "type": "function",
                "name": "create_map_visualization",
                "description": "Create map visualization from property data with coordinates. Use this when user asks for map, peta, visualisasi lokasi, or wants to see property locations on map.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "SQL query to fetch property data with coordinates. Must include id, latitude, longitude and descriptive columns."
                        },
                        "title": {
                            "type": "string",
                            "description": "Title for the map visualization"
                        }
                    },
                    "required": ["sql_query", "title"],
                    "additionalProperties": False
                },
                "strict": True
            }]

            prompt = f"""User question: {user_question}

{geographic_context}

If user is asking for map/peta/visualisasi lokasi, use create_map_visualization function.
Otherwise, generate PostgreSQL query for this question."""

            response = self.client.responses.create(
                model="o4-mini",
                reasoning={"effort": "low"},
                input=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                tools=tools,
                max_output_tokens=500
            )
            
            return response
            
        except Exception as e:
            st.error(f"Error generating query: {str(e)}")
            return None
    
    def format_response(self, user_question: str, query_results: pd.DataFrame, sql_query: str) -> str:
        """Use GPT-4.1-mini to format response in Bahasa Indonesia"""
        try:
            prompt = f"""User asked: {user_question}

SQL Query executed: {sql_query}
Results: {query_results.to_dict('records') if len(query_results) > 0 else 'No results found'}

Provide clear answer in Bahasa Indonesia. Focus on business insights, not technical details."""

            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                stream=True,
                messages=[
                    {
                        "role": "system", 
                        "content": "You interpret database results for business users. Always respond in Bahasa Indonesia with clear, actionable insights."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            full_response = ""
            response_container = st.empty()
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
            return full_response
            
        except Exception as e:
            return f"Maaf, terjadi kesalahan dalam memproses hasil: {str(e)}"
    
    def create_map_visualization(self, query_data: pd.DataFrame, title: str = "Property Locations") -> str:
        """Create map visualization from query data"""
        try:
            # Check if data has required columns
            if 'latitude' not in query_data.columns or 'longitude' not in query_data.columns:
                return "Error: Data tidak memiliki kolom latitude dan longitude untuk visualisasi peta."
            
            # Clean coordinates
            map_df = query_data.copy()
            map_df = map_df.dropna(subset=['latitude', 'longitude'])
            
            # Convert to numeric
            map_df['latitude'] = pd.to_numeric(map_df['latitude'], errors='coerce')
            map_df['longitude'] = pd.to_numeric(map_df['longitude'], errors='coerce')
            
            # Filter valid coordinates
            map_df = map_df[
                (map_df['latitude'] >= -90) & (map_df['latitude'] <= 90) &
                (map_df['longitude'] >= -180) & (map_df['longitude'] <= 180)
            ]
            
            if len(map_df) == 0:
                return "Error: Tidak ada data dengan koordinat yang valid untuk visualisasi peta."
            
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
            
            return f"✅ Peta berhasil ditampilkan dengan {len(map_df)} properti."
            
        except Exception as e:
            return f"Error membuat visualisasi peta: {str(e)}"

    def direct_chat(self, user_question: str) -> str:   
        """Direct chat using GPT-4.1-mini for non-query questions"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                stream=True,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are RHR assistant. Always respond in Bahasa Indonesia."
                    },
                    {
                        "role": "user", 
                        "content": user_question
                    }
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            full_response = ""
            response_container = st.empty()
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
            return full_response
            
        except Exception as e:
            return f"Maaf, terjadi kesalahan: {str(e)}"
    
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

def render_geographic_filter():
    """Render geographic filtering interface"""
    st.markdown('<div class="section-header">Geographic Filter</div>', unsafe_allow_html=True)
    
    if not initialize_database():
        return
    
    try:
        table_name = st.secrets["database"]["table_name"]
    except KeyError:
        st.error("Table name not found in secrets.toml")
        return
    
    st.markdown("Select geographic areas to help AI focus on specific regions (optional)")
    
    # Geographic Filters Section
    st.markdown("#### Geographic Selection")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Province (wadmpr)**")
        if st.button("Load Provinces", key="load_provinces"):
            with st.spinner("Loading province list..."):
                province_options = st.session_state.db_connection.get_unique_geographic_values(
                    'wadmpr', table_name=table_name
                )
                st.session_state.province_options = province_options
        
        if 'province_options' in st.session_state:
            selected_provinces = st.multiselect(
                "Select Provinces",
                st.session_state.province_options,
                key="custom_provinces",
                help="Choose one or more provinces"
            )
        else:
            selected_provinces = []
            st.info("Click 'Load Provinces' to see available options")
    
    with col2:
        st.markdown("**Regency/City (wadmkk)**")
        if selected_provinces and st.button("Load Regencies", key="load_regencies"):
            with st.spinner("Loading regency list..."):
                regency_options = st.session_state.db_connection.get_unique_geographic_values(
                    'wadmkk',
                    {'wadmpr': selected_provinces},
                    table_name=table_name
                )
                st.session_state.regency_options = regency_options
        
        if 'regency_options' in st.session_state and selected_provinces:
            selected_regencies = st.multiselect(
                "Select Regencies/Cities",
                st.session_state.regency_options,
                key="custom_regencies",
                help="Choose regencies within selected provinces"
            )
        else:
            selected_regencies = []
            if not selected_provinces:
                st.info("Select provinces first")
            else:
                st.info("Click 'Load Regencies' to see options")
    
    with col3:
        st.markdown("**District (wadmkc)**")
        if selected_regencies and st.button("Load Districts", key="load_districts"):
            with st.spinner("Loading district list..."):
                district_options = st.session_state.db_connection.get_unique_geographic_values(
                    'wadmkc',
                    {'wadmkk': selected_regencies},
                    table_name=table_name
                )
                st.session_state.district_options = district_options
        
        if 'district_options' in st.session_state and selected_regencies:
            selected_districts = st.multiselect(
                "Select Districts",
                st.session_state.district_options,
                key="custom_districts",
                help="Choose districts within selected regencies"
            )
        else:
            selected_districts = []
            if not selected_regencies:
                st.info("Select regencies first")
            else:
                st.info("Click 'Load Districts' to see options")
    
    # Store geographic filters in session state
    st.session_state.geographic_filters = {
        'wadmpr': selected_provinces,
        'wadmkk': selected_regencies,
        'wadmkc': selected_districts
    }
    
    # Show current selection
    if any([selected_provinces, selected_regencies, selected_districts]):
        st.markdown("**Current Geographic Selection:**")
        if selected_provinces:
            st.write(f"Provinces: {', '.join(selected_provinces)}")
        if selected_regencies:
            st.write(f"Regencies: {', '.join(selected_regencies)}")
        if selected_districts:
            st.write(f"Districts: {', '.join(selected_districts)}")
        
        # Clear filters button
        if st.button("Clear All Filters"):
            for key in ['province_options', 'regency_options', 'district_options']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.geographic_filters = {}
            st.rerun()
    else:
        st.info("No geographic filters applied. AI will search across all locations.")

def render_ai_chat():
    """Render AI chat interface"""
    st.markdown('<div class="section-header">AI Chat</div>', unsafe_allow_html=True)
    
    if not initialize_database():
        return
    
    # Get API key
    try:
        api_key = st.secrets["openai"]["api_key"]
    except KeyError:
        st.error("OpenAI API key not found in secrets.toml")
        return
    
    try:
        table_name = st.secrets["database"]["table_name"]
    except KeyError:
        st.error("Table name not found in secrets.toml")
        return
    
    # Initialize AI chat
    if 'ai_chat' not in st.session_state:
        st.session_state.ai_chat = RHRAIChat(api_key, table_name)
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        # Add welcome message
        welcome_msg = """Hello! I'm your RHR AI assistant. I can help you analyze your property appraisal projects.

You can ask me questions like:
- "How many projects do we have in Jakarta?"
- "Who are our top 5 clients?"
- "Show me some recent land appraisals"
- "What types of properties do we appraise most?"

What would you like to know about your projects?"""
        
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    
    # Display geographic context if available
    if hasattr(st.session_state, 'geographic_filters') and any(st.session_state.geographic_filters.values()):
        st.markdown("**Current Geographic Context:**")
        filters = st.session_state.geographic_filters
        context_parts = []
        if filters.get('wadmpr'):
            context_parts.append(f"Provinces: {', '.join(filters['wadmpr'])}")
        if filters.get('wadmkk'):
            context_parts.append(f"Regencies: {', '.join(filters['wadmkk'])}")
        if filters.get('wadmkc'):
            context_parts.append(f"Districts: {', '.join(filters['wadmkc'])}")
        
        st.info(" | ".join(context_parts))
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your projects..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            try:
                # Build geographic context
                geo_context = ""
                if hasattr(st.session_state, 'geographic_filters') and any(st.session_state.geographic_filters.values()):
                    filters = st.session_state.geographic_filters
                    context_parts = []
                    if filters.get('wadmpr'):
                        context_parts.append(f"Provinces: {filters['wadmpr']}")
                    if filters.get('wadmkk'):
                        context_parts.append(f"Regencies: {filters['wadmkk']}")
                    if filters.get('wadmkc'):
                        context_parts.append(f"Districts: {filters['wadmkc']}")
                    
                    geo_context = "Geographic context: " + " | ".join(context_parts)
                
                # Step 1: Generate SQL query or function call using o4-mini
                ai_response = st.session_state.ai_chat.generate_query(prompt, geo_context)
                
                if ai_response and hasattr(ai_response, 'output') and ai_response.output:
                    # Check if AI called a function
                    function_called = False
                    for output_item in ai_response.output:
                        if hasattr(output_item, 'type') and output_item.type == "function_call":
                            function_called = True
                            
                            if output_item.name == "create_map_visualization":
                                # Parse function arguments
                                args = json.loads(output_item.arguments)
                                sql_query = args.get("sql_query")
                                map_title = args.get("title", "Property Locations")
                                
                                # Execute the SQL query
                                result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
                                
                                if result_df is not None and len(result_df) > 0:
                                    # Create map visualization
                                    map_result = st.session_state.ai_chat.create_map_visualization(result_df, map_title)
                                    
                                    # Show query details in expandable section
                                    with st.expander("📊 Query Details", expanded=False):
                                        st.code(sql_query, language="sql")
                                        st.dataframe(result_df, use_container_width=True)
                                    
                                    # Generate response about the map
                                    map_response = f"""Saya telah membuat visualisasi peta untuk permintaan Anda.

{map_result}

Peta menampilkan lokasi properti berdasarkan data yang tersedia dengan koordinat latitude dan longitude."""
                                    
                                    st.markdown("---")
                                    st.markdown(map_response)
                                    final_response = map_response
                                else:
                                    error_msg = f"Tidak dapat membuat peta: {query_msg}"
                                    st.error(error_msg)
                                    final_response = error_msg
                            break
                    
                    # If no function was called, treat as regular SQL query
                    if not function_called and hasattr(ai_response, 'output_text'):
                        sql_query = ai_response.output_text.strip()
                        
                        if sql_query and "SELECT" in sql_query.upper():
                            try:
                                # Step 2: Execute the query
                                result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
                                
                                if result_df is not None:
                                    # Show query results in expandable section
                                    with st.expander("📊 Query Results", expanded=False):
                                        st.code(sql_query, language="sql")
                                        st.dataframe(result_df, use_container_width=True)
                                    
                                    # Step 3: Format response using GPT-4.1-mini
                                    formatted_response = st.session_state.ai_chat.format_response(
                                        prompt, result_df, sql_query
                                    )
                                    
                                    # # Display the formatted response
                                    # st.markdown("---")
                                    # st.markdown(formatted_response)
                                    final_response = formatted_response
                                    
                                else:
                                    error_msg = f"Query gagal dieksekusi: {query_msg}"
                                    st.error(error_msg)
                                    final_response = error_msg
                            
                            except Exception as e:
                                error_msg = f"Error menjalankan query: {str(e)}"
                                st.error(error_msg)
                                final_response = error_msg
                        else:
                            # If no valid SQL generated, use GPT-4.1-mini directly
                            direct_response = st.session_state.ai_chat.direct_chat(prompt)
                            st.markdown(direct_response)
                            final_response = direct_response
                
                # Add assistant response to history
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": final_response
                })
                
            except Exception as e:
                error_msg = f"Maaf, terjadi kesalahan: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("**Quick Questions:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Project Count", use_container_width=True):
            quick_prompt = "How many total projects do we have?"
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()
    
    with col2:
        if st.button("Top Clients", use_container_width=True):
            quick_prompt = "Who are our top 5 clients by number of projects?"
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()
    
    with col3:
        if st.button("Property Types", use_container_width=True):
            quick_prompt = "What are the most common property types we appraise?"
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()
    
    with col4:
        if st.button("Show Map", use_container_width=True):
            quick_prompt = "Buatkan peta untuk menampilkan lokasi semua properti"
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()
    
    # Chat management
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("Export Chat", use_container_width=True):
            chat_export = {
                "timestamp": datetime.now().isoformat(),
                "geographic_filters": st.session_state.get('geographic_filters', {}),
                "chat_messages": st.session_state.chat_messages
            }
            
            st.download_button(
                label="Download Chat History",
                data=json.dumps(chat_export, indent=2),
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">RHR AI Query Assistant</h1>', unsafe_allow_html=True)
    
    # Check authentication
    if not check_authentication():
        login()
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Geographic Filter", "AI Chat"])
    
    # Show current user
    st.sidebar.markdown("---")
    st.sidebar.success(f"Logged in as: {st.secrets['auth']['username']}")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # Render selected page
    if page == "Geographic Filter":
        render_geographic_filter()
    elif page == "AI Chat":
        render_ai_chat()
    
    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    
    # Database status
    if hasattr(st.session_state, 'db_connection') and st.session_state.db_connection.connection_status:
        st.sidebar.success("Database Connected")
    else:
        st.sidebar.error("Database Disconnected")
    
    # Geographic filters status
    if hasattr(st.session_state, 'geographic_filters') and any(st.session_state.geographic_filters.values()):
        filters = st.session_state.geographic_filters
        filter_count = sum(len(v) for v in filters.values() if v)
        st.sidebar.success(f"Geographic Filters: {filter_count} selected")
    else:
        st.sidebar.info("No Geographic Filters")
    
    # Chat status
    if hasattr(st.session_state, 'chat_messages'):
        st.sidebar.info(f"Chat Messages: {len(st.session_state.chat_messages)}")

if __name__ == "__main__":
    main()