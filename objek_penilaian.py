import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from openai import OpenAI
import json
import traceback
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
import re
import requests
import math

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
    
    def __init__(self, api_key: str, table_name: str, geocode_service: GeocodeService = None):
        # Single OpenAI client for both models
        self.client = OpenAI(api_key=api_key)
        self.table_name = table_name
        self.geocode_service = geocode_service
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth (in kilometers)
        using the Haversine formula
        """
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in kilometers
        r = 6371
        
        return c * r
    
    def extract_location_and_radius(self, user_question: str) -> tuple:
        """
        Extract location name and radius from user question
        Returns: (location_name, radius_km)
        """
        # Pattern to extract location and radius
        # Examples: "setiabudi one (radius 1km)", "mall taman anggrek radius 2km", "terdekat dari plaza indonesia (500m)"
        
        # Extract radius
        radius_patterns = [
            r'radius\s*(\d+(?:\.\d+)?)\s*km',
            r'radius\s*(\d+(?:\.\d+)?)\s*meter',
            r'radius\s*(\d+(?:\.\d+)?)\s*m\b',
            r'\((\d+(?:\.\d+)?)\s*km\)',
            r'\((\d+(?:\.\d+)?)\s*meter\)',
            r'\((\d+(?:\.\d+)?)\s*m\)'
        ]
        
        radius_km = 1.0  # default radius
        for pattern in radius_patterns:
            match = re.search(pattern, user_question, re.IGNORECASE)
            if match:
                radius_value = float(match.group(1))
                # Convert to km if needed
                if 'meter' in pattern or r'\s*m\b' in pattern or r'\s*m\)' in pattern:
                    radius_km = radius_value / 1000
                else:
                    radius_km = radius_value
                break
        
        # Extract location name
        # Remove common phrases and extract the location
        cleaned_question = user_question.lower()
        
        # Enhanced removal of common phrases for Indonesian language
        remove_phrases = [
            'ada proyek apa saja di', 'ada proyek apa di', 'ada apa di',
            'buatkan map', 'buatkan peta', 'tampilkan map', 'tampilkan peta',
            'proyek terdekat dari', 'terdekat dari', 'proyek sekitar', 'proyek dekat',
            'di sekitar', 'di dekat', 'sekitar', 'dekat', 'map', 'near', 'nearby',
            'what projects are', 'show me projects', 'find projects'
        ]
        
        for phrase in remove_phrases:
            cleaned_question = cleaned_question.replace(phrase, '')
        
        # Remove radius information
        for pattern in radius_patterns:
            cleaned_question = re.sub(pattern, '', cleaned_question, flags=re.IGNORECASE)
        
        # Clean up and extract location
        location_name = cleaned_question.strip()
        
        # Remove extra parentheses, question marks, and common words
        location_name = re.sub(r'\(.*?\)', '', location_name)
        location_name = location_name.replace('radius', '').replace('km', '').replace('meter', '').replace('m', '')
        location_name = location_name.replace('?', '').replace('!', '').replace(',', '')
        location_name = location_name.strip()
        
        return location_name, radius_km
    
    def generate_query(self, user_question: str, geographic_context: str = "") -> str:
        #create_chart_visualization(chart_type: string, sql_query: string, title: string, x_column: string, color_column y_column: string: string)
        system_prompt = f"""
You are a strict SQL‐only assistant for the RHR property appraisal database.
You have three helper functions:

  create_map_visualization(sql_query: string, title: string)
    → Returns a map of properties when called.
    
  find_nearby_projects(location_name: string, radius_km: float, title: string)
    → Finds and maps projects near a specific location within given radius.
    
  create_chart_visualization(chart_type: string, sql_query: string, title: string, x_column: string, color_column ,y_column: string: string)
    → Creates various charts (bar, pie, line, scatter, histogram) from data.

**RULES**  
- If the user asks for charts/graphs ("grafik", "chart", "barchart", "pie", etc.), use `create_chart_visualization` function.
- If the user asks for projects near a specific location, use `find_nearby_projects` function.
- If the user asks for a general map, use `create_map_visualization` function.  
- Otherwise return *only* a PostgreSQL query (no explanations).

TABLE: {self.table_name}

DETAILED COLUMN INFORMATION:

Project Information:
- sumber (text): Data source (e.g., "kontrak" = contract-based projects)
- pemberi_tugas (text): Client/Task giver (e.g., "PT Asuransi Jiwa IFG", "PT Perkebunan Nusantara II")
- no_kontrak (text): Contract number (e.g., "RHR00C1P0623111.0")
- nama_lokasi (text): Location name (e.g., "Lokasi 20", "Lokasi 3")
- id (int8): Unique project identifier (e.g., 16316, 17122) - PRIMARY KEY

Property Information:
- objek_penilaian (text): Appraisal object type (e.g., "real properti")
- nama_objek (text): Object name (e.g., "Rumah", "Tanah Kosong")
- jenis_objek (float8): Object type code that joins with master_jenis_objek table for readable names
- kepemilikan (text): Ownership type (e.g., "tunggal" = single ownership)
- keterangan (text): Additional notes (e.g., "Luas Tanah : 1.148", may contain NULL)

Status & Management:
- status (float8): Project status code that joins with master_status_objek table for readable names  
- cabang (float8): Branch office code that joins with master_cabang table for readable names
- jc_text (text): Job captain or 'jc' (e.g., "IMW","FHM")

Geographic Data:
- latitude (float8): Latitude coordinates (e.g., -6.236507782741299)
- longitude (float8): Longitude coordinates (e.g., 106.86356067983168)
- geometry (geometry): PostGIS geometry field (binary spatial data)
- wadmpr (text): Province (e.g., "DKI Jakarta", "Sumatera Utara")
- wadmkk (text): Regency/City (e.g., "Kota Administrasi Jakarta Selatan", "Deli Serdang")
- wadmkc (text): District (e.g., "Tebet", "Labuhan Deli")

CRITICAL SQL RULES:
1. For counting: SELECT COUNT(*) FROM {self.table_name} WHERE...
2. For samples: SELECT id, [columns] FROM {self.table_name} WHERE... ORDER BY id DESC LIMIT 5
3. For grouping with lookups: 
   - Object types: SELECT mj.name as object_type, COUNT(*) FROM {self.table_name} t LEFT JOIN master_jenis_objek mj ON t.jenis_objek = mj.id GROUP BY mj.name ORDER BY COUNT(*) DESC LIMIT 10
   - Status: SELECT ms.name as status_name, COUNT(*) FROM {self.table_name} t LEFT JOIN master_status_objek ms ON t.status = ms.id GROUP BY ms.name ORDER BY COUNT(*) DESC LIMIT 10  
   - Branch: SELECT mc.name as branch_name, COUNT(*) FROM {self.table_name} t LEFT JOIN master_cabang mc ON t.cabang = mc.id GROUP BY mc.name ORDER BY COUNT(*) DESC LIMIT 10
   - Other columns: SELECT [column], COUNT(*) FROM {self.table_name} t WHERE [column] IS NOT NULL GROUP BY [column] ORDER BY COUNT(*) DESC LIMIT 10
4. Always use readable names: mj.name (not jenis_objek), ms.name (not status), mc.name (not cabang)
5. Always handle NULLs: Use "WHERE column IS NOT NULL" when querying specific columns
6. Text search: Use "ILIKE '%text%'" for case-insensitive search
7. Geographic search: "(wadmpr ILIKE '%location%' OR wadmkk ILIKE '%location%' OR wadmkc ILIKE '%location%')"
8. Always add LIMIT to prevent large result sets
9. For map visualization: ALWAYS include id, latitude, longitude, and descriptive columns (nama_objek, pemberi_tugas, wadmpr, wadmkk)
10. For readable names: Always JOIN with lookup tables to get names instead of codes
11. Use aliases: mj (master_jenis_objek), ms (master_status_objek), mc (master_cabang)
12. Standard JOINs:
   - LEFT JOIN master_jenis_objek mj ON t.jenis_objek = mj.id
   - LEFT JOIN master_status_objek ms ON t.status = ms.id  
   - LEFT JOIN master_cabang mc ON t.cabang = mc.id
13. Select readable columns: mj.name as jenis_objek_name, ms.name as status_name, mc.name as cabang_name

Generate ONLY the PostgreSQL query, no explanations."""

        # Check for chart/graph requests
        is_chart_request = bool(re.search(r"\b(grafik|chart|barchart|pie|line|scatter|histogram|graph|visualisasi data)\b", user_question, re.I))
        is_nearby_request = bool(re.search(r"\b(terdekat|sekitar|dekat|nearby|near)\b", user_question, re.I))
        is_map_request = bool(re.search(r"\b(map|peta|visualisasi lokasi)\b", user_question, re.I))

        tools = [
            {
                "type": "function",
                "name": "create_map_visualization",
                "description": "Create a map of properties. Use when the user requests general location visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "SQL query including id, latitude, longitude, nama_objek, pemberi_tugas, wadmpr, wadmkk"
                        },
                        "title": { "type": "string" }
                    },
                    "required": ["sql_query", "title"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "find_nearby_projects",
                "description": "Find and map projects near a specific location. Use when user asks for projects near a place.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_name": {
                            "type": "string",
                            "description": "Name of the location to search near (e.g., 'Setiabudi One', 'Mall Taman Anggrek')"
                        },
                        "radius_km": {
                            "type": "number",
                            "description": "Search radius in kilometers (default: 1.0)"
                        },
                        "title": { "type": "string" }
                    },
                    "required": ["location_name", "radius_km", "title"],
                    "additionalProperties": False
                },
                "strict": True
            },
            {
                "type": "function",
                "name": "create_chart_visualization",
                "description": "Create charts from data. Use when user requests graphs, charts, or data visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "pie", "line", "scatter", "histogram", "auto"],
                            "description": "Type of chart to create"
                        },
                        "sql_query": {
                            "type": "string",
                            "description": "SQL query to get data for the chart"
                        },
                        "title": { "type": "string" },
                        "x_column": {
                            "type": "string",
                            "description": "Column name for x-axis (optional, can be auto-detected)"
                        },
                        "y_column": {
                            "type": "string", 
                            "description": "Column name for y-axis (optional, can be auto-detected)"
                        },
                        "color_column": {
                            "type": "string",
                            "description": "Column name for color grouping (optional)"
                        }
                    },
                    "required": ["chart_type", "sql_query", "title",
                                  "x_column", "y_column",
                                    "color_column"
                                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        ]

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        if geographic_context:
            messages.append({"role": "user", "content": geographic_context})
        messages.append({"role": "user", "content": user_question})

        # Determine which function to use
        tool_choice = "auto"
        if is_chart_request:
            tool_choice = {"type": "function", "name": "create_chart_visualization"}
        elif is_nearby_request and is_map_request:
            tool_choice = {"type": "function", "name": "find_nearby_projects"}
        elif is_map_request and not is_nearby_request:
            tool_choice = {"type": "function", "name": "create_map_visualization"}

        response = self.client.responses.create(
            model="o4-mini",
            reasoning={"effort": "low"},
            input=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_output_tokens=500
        )

        return response
    
    def format_response(self, user_question: str, query_results: pd.DataFrame, sql_query: str) -> str:
        """Use GPT-4.1-mini to format response in Bahasa Indonesia"""
        try:
            prompt = f"""User asked: {user_question}

SQL Query executed: {sql_query}
Results: {query_results.to_dict('records') if len(query_results) > 0 else 'No results found'}

Provide clear answer in Bahasa Indonesia. Focus on business insights, not technical details.
In every end of your answer, add 'wiwokdetok'
"""

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
    
    def determine_chart_type_and_data(self, user_question: str, last_query_result: pd.DataFrame = None) -> tuple:
        """
        Determine the best chart type and prepare data based on user request
        Returns: (chart_type, x_column, y_column, color_column, suggested_sql)
        """
        user_question_lower = user_question.lower()
        
        # Detect specific chart type requests
        chart_type = "auto"
        if any(word in user_question_lower for word in ['bar', 'barchart', 'batang']):
            chart_type = "bar"
        elif any(word in user_question_lower for word in ['pie', 'donut', 'lingkaran']):
            chart_type = "pie"
        elif any(word in user_question_lower for word in ['line', 'trend', 'garis']):
            chart_type = "line"
        elif any(word in user_question_lower for word in ['scatter', 'titik', 'korelasi']):
            chart_type = "scatter"
        elif any(word in user_question_lower for word in ['histogram', 'distribusi']):
            chart_type = "histogram"
        
        # If user refers to previous data
        if any(phrase in user_question_lower for phrase in ['data tadi', 'berdasarkan data', 'dari hasil']):
            if last_query_result is not None and len(last_query_result) > 0:
                return self._suggest_chart_from_dataframe(last_query_result, chart_type)
        
        # Generate appropriate SQL based on chart type and common requests
        suggested_sql = self._generate_chart_sql(user_question_lower, chart_type)
        
        return chart_type, None, None, None, suggested_sql
    
    def _suggest_chart_from_dataframe(self, df: pd.DataFrame, preferred_chart: str = "auto") -> tuple:
        """Suggest best chart configuration from existing dataframe"""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove system columns
        system_cols = ['id', 'latitude', 'longitude', 'geometry']
        numeric_cols = [col for col in numeric_cols if col not in system_cols]
        
        x_col, y_col, color_col = None, None, None
        chart_type = preferred_chart
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            if preferred_chart == "auto":
                if 'count' in numeric_cols[0].lower() or len(df) < 50:
                    chart_type = "bar"
                else:
                    chart_type = "scatter"
            
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
            color_col = categorical_cols[1] if len(categorical_cols) > 1 else None
            
        elif len(categorical_cols) > 1:
            chart_type = "pie" if preferred_chart == "auto" else preferred_chart
            x_col = categorical_cols[0]
            
        elif len(numeric_cols) > 1:
            chart_type = "scatter" if preferred_chart == "auto" else preferred_chart
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        
        return chart_type, x_col, y_col, color_col, None
    
    def _generate_chart_sql(self, user_question: str, chart_type: str) -> str:
        """Generate SQL query for chart based on user request"""
        
        # Common chart SQL patterns
        if any(word in user_question for word in ['client', 'pemberi tugas', 'klien']):
            return f"""
            SELECT pemberi_tugas, COUNT(*) as jumlah_proyek
            FROM {self.table_name} t
            WHERE pemberi_tugas IS NOT NULL
            GROUP BY pemberi_tugas 
            ORDER BY jumlah_proyek DESC 
            LIMIT 10
            """
        
        elif any(word in user_question for word in ['provinsi', 'province', 'wilayah']):
            return f"""
            SELECT wadmpr as provinsi, COUNT(*) as jumlah_proyek
            FROM {self.table_name} t
            WHERE wadmpr IS NOT NULL
            GROUP BY wadmpr 
            ORDER BY jumlah_proyek DESC 
            LIMIT 15
            """
        
        elif any(word in user_question for word in ['jenis', 'tipe', 'type', 'objek']):
            return f"""
            SELECT mj.name as jenis_objek, COUNT(*) as jumlah_proyek
            FROM {self.table_name} t
            LEFT JOIN master_jenis_objek mj ON t.jenis_objek = mj.id
            WHERE mj.name IS NOT NULL
            GROUP BY mj.name 
            ORDER BY jumlah_proyek DESC 
            LIMIT 10
            """
        
        elif any(word in user_question for word in ['status', 'kondisi']):
            return f"""
            SELECT ms.name as status_proyek, COUNT(*) as jumlah_proyek
            FROM {self.table_name} t
            LEFT JOIN master_status_objek ms ON t.status = ms.id
            WHERE ms.name IS NOT NULL
            GROUP BY ms.name 
            ORDER BY jumlah_proyek DESC
            """
        
        elif any(word in user_question for word in ['cabang', 'branch', 'kantor']):
            return f"""
            SELECT mc.name as cabang, COUNT(*) as jumlah_proyek
            FROM {self.table_name} t
            LEFT JOIN master_cabang mc ON t.cabang = mc.id
            WHERE mc.name IS NOT NULL
            GROUP BY mc.name 
            ORDER BY jumlah_proyek DESC 
            LIMIT 10
            """
        
        else:
            # Default: top clients
            return f"""
            SELECT pemberi_tugas, COUNT(*) as jumlah_proyek
            FROM {self.table_name} t
            WHERE pemberi_tugas IS NOT NULL
            GROUP BY pemberi_tugas 
            ORDER BY jumlah_proyek DESC 
            LIMIT 10
            """
    
    def create_chart_visualization(self, data: pd.DataFrame, chart_type: str, title: str, 
                                 x_col: str = None, y_col: str = None, color_col: str = None) -> str:
        """Create chart visualization using Plotly Express"""
        try:
            if data is None or len(data) == 0:
                return "Error: Tidak ada data untuk membuat grafik."
            
            # Auto-detect columns if not provided
            if x_col is None or y_col is None:
                chart_type, x_col, y_col, color_col, _ = self._suggest_chart_from_dataframe(data, chart_type)
            
            # Ensure columns exist in dataframe
            available_cols = data.columns.tolist()
            if x_col and x_col not in available_cols:
                x_col = available_cols[0] if available_cols else None
            if y_col and y_col not in available_cols:
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                y_col = numeric_cols[0] if numeric_cols else available_cols[1] if len(available_cols) > 1 else None
            
            fig = None
            
            # Create chart based on type
            if chart_type == "bar":
                fig = px.bar(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    title=title,
                    labels={x_col: x_col.replace('_', ' ').title(), 
                           y_col: y_col.replace('_', ' ').title() if y_col else 'Count'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                
            elif chart_type == "pie":
                # For pie charts, use count data or first numeric column
                if y_col:
                    fig = px.pie(
                        data, 
                        names=x_col, 
                        values=y_col, 
                        title=title
                    )
                else:
                    # Count occurrences
                    pie_data = data[x_col].value_counts().reset_index()
                    pie_data.columns = [x_col, 'count']
                    fig = px.pie(
                        pie_data, 
                        names=x_col, 
                        values='count', 
                        title=title
                    )
                    
            elif chart_type == "line":
                fig = px.line(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    title=title,
                    markers=True
                )
                
            elif chart_type == "scatter":
                fig = px.scatter(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    title=title,
                    size_max=60
                )
                
            elif chart_type == "histogram":
                fig = px.histogram(
                    data, 
                    x=x_col if x_col else y_col, 
                    color=color_col,
                    title=title,
                    nbins=20
                )
            
            else:
                # Default to bar chart
                fig = px.bar(
                    data, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col,
                    title=title
                )
                fig.update_layout(xaxis_tickangle=-45)
            
            if fig:
                # Improve chart appearance
                fig.update_layout(
                    height=500,
                    showlegend=True if color_col else False,
                    template="plotly_white",
                    title_x=0.5,
                    margin=dict(l=50, r=50, t=80, b=100)
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                return f"✅ Grafik {chart_type} berhasil ditampilkan dengan {len(data)} data points."
            else:
                return "Error: Gagal membuat grafik."
                
        except Exception as e:
            return f"Error membuat grafik: {str(e)}"
        
    def find_nearby_projects(self, location_name: str, radius_km: float, title: str, db_connection) -> str:
        """Find projects near a specific location using geocoding"""
        try:
            if not self.geocode_service:
                return "Error: Layanan geocoding tidak tersedia. Silakan tambahkan Google Maps API key."
            
            # Geocode the location
            with st.spinner(f"Mencari koordinat untuk '{location_name}'..."):
                lat, lng, formatted_address = self.geocode_service.geocode_address(location_name)
            
            if lat is None or lng is None:
                return f"Error: Tidak dapat menemukan koordinat untuk lokasi '{location_name}'. Silakan coba dengan nama lokasi yang lebih spesifik."
            
            st.success(f"📍 Lokasi ditemukan: {formatted_address}")
            st.info(f"Koordinat: {lat:.6f}, {lng:.6f}")
            
            # Query nearby projects using Haversine formula
            sql_query = f"""
            SELECT 
                t.id,
                t.nama_objek,
                t.pemberi_tugas,
                t.latitude,
                t.longitude,
                t.wadmpr,
                t.wadmkk,
                t.wadmkc,
                mj.name as jenis_objek_name,
                ms.name as status_name,
                mc.name as cabang_name,
                (6371 * acos(
                    cos(radians({lat})) * cos(radians(t.latitude)) * 
                    cos(radians(t.longitude) - radians({lng})) + 
                    sin(radians({lat})) * sin(radians(t.latitude))
                )) as distance_km
            FROM {self.table_name} t
            LEFT JOIN master_jenis_objek mj ON t.jenis_objek = mj.id
            LEFT JOIN master_status_objek ms ON t.status = ms.id
            LEFT JOIN master_cabang mc ON t.cabang = mc.id
            WHERE 
                t.latitude IS NOT NULL 
                AND t.longitude IS NOT NULL
                AND (6371 * acos(
                    cos(radians({lat})) * cos(radians(t.latitude)) * 
                    cos(radians(t.longitude) - radians({lng})) + 
                    sin(radians({lat})) * sin(radians(t.latitude))
                )) <= {radius_km}
            ORDER BY distance_km ASC
            LIMIT 50
            """
            
            # Execute query
            with st.spinner(f"Mencari proyek dalam radius {radius_km} km..."):
                result_df, query_msg = db_connection.execute_query(sql_query)
            
            if result_df is not None and len(result_df) > 0:
                # Add reference point to map
                reference_point = pd.DataFrame({
                    'latitude': [lat],
                    'longitude': [lng],
                    'id': ['REF'],
                    'nama_objek': [location_name],
                    'pemberi_tugas': ['Reference Point'],
                    'wadmpr': [''],
                    'wadmkk': [''],
                    'distance_km': [0.0]
                })
                
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
                    title=f"{title} - {len(result_df)} proyek dalam radius {radius_km} km dari {location_name}"
                )
                
                # Display map
                st.plotly_chart(fig, use_container_width=True)
                
                # Show results table
                with st.expander("📊 Detail Proyek Terdekat", expanded=False):
                    st.dataframe(result_df[['id', 'nama_objek', 'pemberi_tugas', 'jenis_objek_name', 
                                          'wadmpr', 'wadmkk', 'distance_km']].round(2), 
                               use_container_width=True)
                
                return f"✅ Ditemukan {len(result_df)} proyek dalam radius {radius_km} km dari {location_name}. Proyek terdekat berjarak {result_df['distance_km'].min():.2f} km."
            
            else:
                return f"❌ Tidak ada proyek yang ditemukan dalam radius {radius_km} km dari {location_name}."
            
        except Exception as e:
            return f"Error mencari proyek terdekat: {str(e)}"

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

def initialize_geocode_service():
    """Initialize geocoding service"""
    try:
        google_api_key = st.secrets["google"]["api_key"]
        if 'geocode_service' not in st.session_state:
            st.session_state.geocode_service = GeocodeService(google_api_key)
        return st.session_state.geocode_service
    except KeyError:
        st.warning("Google Maps API key tidak ditemukan. Fitur pencarian lokasi tidak tersedia.")
        return None

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
    
    # Initialize geocoding service
    geocode_service = initialize_geocode_service()
    
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
    
    # Initialize AI chat with geocoding service
    if 'ai_chat' not in st.session_state:
        st.session_state.ai_chat = RHRAIChat(api_key, table_name, geocode_service)
    
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
        # Add welcome message
        welcome_msg = """Halo! Saya asisten AI RHR Anda. Saya dapat membantu Anda menganalisis proyek penilaian properti Anda.

Anda dapat menanyakan hal-hal seperti:
- "Berapa banyak proyek yang kita miliki di Jakarta?"
- "Siapa 5 klien utama kita?"
- "Jenis properti apa yang paling sering kita nilai?"
- "Buatkan peta proyek terdekat dari Setiabudi One dengan radius 1 km"
- "Tampilkan proyek sekitar Mall Taman Anggrek dalam radius 500 m"
- "Buatkan grafik pemberi tugas di tiap cabang, saya ingin lihat pada tiap cabang pemberi tugasnya siapa saja dan berapa jumlahnya."

Apa yang ingin Anda ketahui tentang proyek Anda?"""
        
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    
    # Display geocoding service status
    if geocode_service:
        st.success("🌍 Layanan pencarian lokasi aktif")
    else:
        st.warning("⚠️ Layanan pencarian lokasi tidak aktif - tambahkan Google Maps API key untuk menggunakan fitur pencarian terdekat")
    
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
                            
                            elif output_item.name == "create_chart_visualization":
                                # Parse function arguments
                                args = json.loads(output_item.arguments)
                                chart_type = args.get("chart_type", "auto")
                                sql_query = args.get("sql_query")
                                chart_title = args.get("title", "Data Visualization")
                                x_col = args.get("x_column")
                                y_col = args.get("y_column") 
                                color_col = args.get("color_column")
                                
                                # Execute the SQL query
                                result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
                                
                                if result_df is not None and len(result_df) > 0:
                                    # Create chart visualization
                                    chart_result = st.session_state.ai_chat.create_chart_visualization(
                                        result_df, chart_type, chart_title, x_col, y_col,
                                          color_col
                                    )
                                    
                                    # Show query details in expandable section
                                    with st.expander("📊 Data & Query Details", expanded=False):
                                        st.code(sql_query, language="sql")
                                        st.dataframe(result_df, use_container_width=True)
                                    
                                    # Store last query result for future reference
                                    st.session_state.last_query_result = result_df
                                    
                                    # Generate response about the chart
                                    chart_response = f"""Saya telah membuat visualisasi grafik untuk permintaan Anda.

{chart_result}

Grafik menampilkan data berdasarkan query yang dijalankan."""
                                    
                                    st.markdown("---")
                                    st.markdown(chart_response)
                                    final_response = chart_response
                                else:
                                    error_msg = f"Tidak dapat membuat grafik: {query_msg}"
                                    st.error(error_msg)
                                    final_response = error_msg
                            
                            elif output_item.name == "find_nearby_projects":
                                # Parse function arguments
                                args = json.loads(output_item.arguments)
                                location_name = args.get("location_name")
                                radius_km = args.get("radius_km", 1.0)
                                map_title = args.get("title", f"Proyek Terdekat dari {location_name}")
                                
                                # Find nearby projects
                                nearby_result = st.session_state.ai_chat.find_nearby_projects(
                                    location_name, radius_km, map_title, st.session_state.db_connection
                                )
                                
                                st.markdown("---")
                                st.markdown(nearby_result)
                                final_response = nearby_result
                            
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
                                    
                                    # Store last query result for future reference
                                    st.session_state.last_query_result = result_df
                                    
                                    # Step 3: Format response using GPT-4.1-mini
                                    formatted_response = st.session_state.ai_chat.format_response(
                                        prompt, result_df, sql_query
                                    )
                                    
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
    
    # # Quick action buttons
    # st.markdown("---")
    # st.markdown("**Quick Questions:**")
    
    # col1, col2, col3, col4 = st.columns(4)
    
    # with col1:
    #     if st.button("Project Count", use_container_width=True):
    #         quick_prompt = "How many total projects do we have?"
    #         st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #         st.rerun()
    
    # with col2:
    #     if st.button("Top Clients", use_container_width=True):
    #         quick_prompt = "Who are our top 5 clients by number of projects?"
    #         st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #         st.rerun()
    
    # with col3:
    #     if st.button("Property Types", use_container_width=True):
    #         quick_prompt = "What are the most common property types we appraise?"
    #         st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #         st.rerun()
    
    # with col4:
    #     if st.button("Show Chart", use_container_width=True):
    #         quick_prompt = "Buatkan grafik pemberi tugas terbanyak"
    #         st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #         st.rerun()
    
    # # NEW: Chart-based quick actions
    # st.markdown("**📊 Chart Quick Actions:**")
    # col1, col2, col3, col4 = st.columns(4)
    
    # with col1:
    #     if st.button("Client Chart", use_container_width=True):
    #         quick_prompt = "Tampilkan barchart klien terbanyak"
    #         st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #         st.rerun()
    
    # with col2:
    #     if st.button("Province Pie", use_container_width=True):
    #         quick_prompt = "Buatkan pie chart berdasarkan provinsi"
    #         st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #         st.rerun()
    
    # with col3:
    #     if st.button("Property Types", use_container_width=True):
    #         quick_prompt = "Tampilkan grafik jenis properti"
    #         st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #         st.rerun()
    
    # with col4:
    #     if st.button("Status Chart", use_container_width=True):
    #         quick_prompt = "Buatkan barchart status proyek"
    #         st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #         st.rerun()
    
    # # NEW: Location-based quick actions
    # if geocode_service:
    #     st.markdown("**📍 Location-based Quick Actions:**")
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         if st.button("Near Plaza Indonesia", use_container_width=True):
    #             quick_prompt = "Buatkan map proyek terdekat dari Plaza Indonesia radius 2km"
    #             st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #             st.rerun()
        
    #     with col2:
    #         if st.button("Near Senayan City", use_container_width=True):
    #             quick_prompt = "Tampilkan proyek sekitar Senayan City dalam radius 1.5km"
    #             st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #             st.rerun()
        
    #     with col3:
    #         if st.button("Near Thamrin", use_container_width=True):
    #             quick_prompt = "Cari proyek terdekat dari Thamrin Jakarta radius 1km"
    #             st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
    #             st.rerun()
    
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
    
    # Geocoding service status
    try:
        google_api_key = st.secrets["google"]["api_key"]
        st.sidebar.success("🌍 Geocoding Service Available")
    except KeyError:
        st.sidebar.warning("⚠️ Geocoding Service Unavailable")
    
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