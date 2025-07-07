import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import json
import traceback
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
import re
import requests
import math
from typing import Any, Optional, List, Dict
from dataclasses import dataclass
import asyncio

# OpenAI Agents framework imports
from agents import Agent, Runner, function_tool, handoff, RunContextWrapper
from agents.models.openai_responses import OpenAIResponsesModel
from pydantic import BaseModel
import openai  # Add this import

# Configure OpenAI API key globally and for environment
try:
    import os
    api_key = st.secrets["openai"]["api_key"]
    
    # Set for openai library
    openai.api_key = api_key
    
    # Set environment variable for agents framework tracing
    os.environ["OPENAI_API_KEY"] = api_key
    
except KeyError:
    st.error("OpenAI API key not found in secrets.toml. Please add your API key.")
    st.stop()

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="RHR AI Query App with Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
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

# Context for agents
@dataclass
class RHRContext:
    """Context shared across all agents"""
    db_connection: Any = None
    geocode_service: Any = None
    table_name: str = ""
    geographic_filters: Dict = None
    last_query_result: pd.DataFrame = None
    last_map_data: pd.DataFrame = None
    chat_history: List[Dict] = None

# Pydantic models for structured data
class SQLQueryRequest(BaseModel):
    query: str
    query_type: str  # "data", "map", "chart", "nearby"
    context: str

class QueryResponse(BaseModel):
    success: bool
    data: Optional[str] = None
    error: Optional[str] = None
    query_executed: Optional[str] = None

class VisualizationRequest(BaseModel):
    chart_type: str
    title: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None

class LocationRequest(BaseModel):
    location_name: str
    radius_km: float
    title: str

# Keep existing classes with minor modifications
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
    
    def get_unique_geographic_values(self, column: str, filters: Dict = None, table_name: str = "objek_penilaian") -> List[str]:
        """Get unique values for geographic columns with optional filters"""
        try:
            if not self.connection_status:
                return []
            
            # Build base query
            query = f"SELECT DISTINCT {column} FROM {table_name} WHERE {column} IS NOT NULL AND {column} != '' AND {column} != 'NULL'"
            
            # Add filters if provided
            if filters:
                for filter_col, filter_values in filters.items():
                    if filter_values:
                        values_str = "', '".join(filter_values)
                        query += f" AND {filter_col} IN ('{values_str}')"
            
            query += f" ORDER BY {column}"
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                return [row[0] for row in result if row[0]]
        
        except Exception as e:
            st.error(f"Error fetching {column} values: {str(e)}")
            return []

# Agent tool functions
@function_tool
async def execute_sql_query(ctx: RunContextWrapper[RHRContext], request: SQLQueryRequest) -> str:
    """Execute SQL query and return results as JSON string"""
    try:
        db_conn = ctx.context.db_connection
        if not db_conn or not db_conn.connection_status:
            return json.dumps({"error": "Database connection not available"})
        
        result_df, message = db_conn.execute_query(request.query)
        
        if result_df is not None:
            # Store in context for future reference
            ctx.context.last_query_result = result_df
            
            # Return structured data
            response = {
                "success": True,
                "data": result_df.to_dict('records'),
                "query_executed": request.query,
                "row_count": len(result_df),
                "columns": list(result_df.columns)
            }
            return json.dumps(response)
        else:
            return json.dumps({
                "success": False,
                "error": message,
                "query_executed": request.query
            })
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "query_executed": request.query
        })

@function_tool
async def create_map_visualization(ctx: RunContextWrapper[RHRContext], data_json: str, title: str = "Property Locations") -> str:
    """Create map visualization from query data"""
    try:
        data_dict = json.loads(data_json)
        if not data_dict.get("success"):
            return f"Cannot create map: {data_dict.get('error', 'Unknown error')}"
        
        df = pd.DataFrame(data_dict["data"])
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return "Error: Data does not have latitude and longitude columns for map visualization."
        
        # Clean and filter coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        df = df[
            (df['latitude'] >= -90) & (df['latitude'] <= 90) &
            (df['longitude'] >= -180) & (df['longitude'] <= 180) &
            (df['latitude'] != 0) & (df['longitude'] != 0)
        ]
        
        if len(df) == 0:
            return "Error: No valid coordinates found for map visualization."
        
        # Enhanced tool configuration with limits
        try:
            tool_config = get_tool_config()
            max_results = tool_config["max_map_results"]
            
            # Limit data if necessary
            if len(df) > max_results:
                st.warning(f"Showing first {max_results} results out of {len(df)} total. Use filters to narrow down results.")
                df = df.head(max_results)
        except:
            # Fallback if get_tool_config fails
            max_results = 50
            if len(df) > max_results:
                st.warning(f"Showing first {max_results} results out of {len(df)} total.")
                df = df.head(max_results)
        
        # Create map
        fig = go.Figure()
        
        # Create hover text
        hover_text = []
        for idx, row in df.iterrows():
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
            lat=df['latitude'],
            lon=df['longitude'],
            mode='markers',
            marker=dict(size=8, color='red'),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Properties'
        ))
        
        # Calculate center
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
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
        
        # Store map data in context
        ctx.context.last_map_data = df.copy()
        
        return f"✅ Map successfully displayed with {len(df)} properties."
        
    except Exception as e:
        return f"Error creating map visualization: {str(e)}"

@function_tool
async def create_chart_visualization(ctx: RunContextWrapper[RHRContext], data_json: str, request: VisualizationRequest) -> str:
    """Create chart visualization from query data"""
    try:
        data_dict = json.loads(data_json)
        if not data_dict.get("success"):
            return f"Cannot create chart: {data_dict.get('error', 'Unknown error')}"
        
        df = pd.DataFrame(data_dict["data"])
        
        if len(df) == 0:
            return "Error: No data available for chart visualization."
        
        # Apply data limits for performance
        try:
            tool_config = get_tool_config()
            max_data_points = tool_config["max_chart_data_points"]
            
            if len(df) > max_data_points:
                st.warning(f"Large dataset detected. Showing first {max_data_points} records out of {len(df)} for performance.")
                df = df.head(max_data_points)
        except:
            # Fallback if get_tool_config fails
            max_data_points = 1000
            if len(df) > max_data_points:
                st.warning(f"Large dataset detected. Showing first {max_data_points} records.")
                df = df.head(max_data_points)
        
        # Auto-detect columns if not provided
        chart_type = request.chart_type
        x_col = request.x_column
        y_col = request.y_column
        color_col = request.color_column
        
        # Auto-detection logic
        if not x_col or not y_col:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Remove system columns
            system_cols = ['id', 'latitude', 'longitude', 'geometry']
            numeric_cols = [col for col in numeric_cols if col not in system_cols]
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                x_col = x_col or categorical_cols[0]
                y_col = y_col or numeric_cols[0]
            elif len(categorical_cols) > 1:
                x_col = x_col or categorical_cols[0]
            elif len(numeric_cols) > 1:
                x_col = x_col or numeric_cols[0]
                y_col = y_col or numeric_cols[1]
        
        fig = None
        
        # Create chart based on type
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=request.title)
            fig.update_layout(xaxis_tickangle=-45)
        elif chart_type == "pie":
            if y_col:
                fig = px.pie(df, names=x_col, values=y_col, title=request.title)
            else:
                pie_data = df[x_col].value_counts().reset_index()
                pie_data.columns = [x_col, 'count']
                fig = px.pie(pie_data, names=x_col, values='count', title=request.title)
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=request.title, markers=True)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=request.title)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_col or y_col, color=color_col, title=request.title, nbins=20)
        else:
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=request.title)
        
        if fig:
            fig.update_layout(
                height=500,
                showlegend=True if color_col else False,
                template="plotly_white",
                title_x=0.5,
                margin=dict(l=50, r=50, t=80, b=100)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            return f"✅ Chart ({chart_type}) successfully displayed with {len(df)} data points."
        else:
            return "Error: Failed to create chart."
            
    except Exception as e:
        return f"Error creating chart visualization: {str(e)}"

@function_tool
async def find_nearby_projects(ctx: RunContextWrapper[RHRContext], request: LocationRequest) -> str:
    """Find projects near a specific location using geocoding"""
    try:
        geocode_service = ctx.context.geocode_service
        if not geocode_service:
            return "Error: Geocoding service not available. Please add Google Maps API key."
        
        # Geocode the location
        lat, lng, formatted_address = geocode_service.geocode_address(request.location_name)
        
        if lat is None or lng is None:
            return f"Error: Could not find coordinates for location '{request.location_name}'. Please try with a more specific location name."
        
        st.success(f"📍 Location found: {formatted_address}")
        st.info(f"Coordinates: {lat:.6f}, {lng:.6f}")
        
        # Query nearby projects using Haversine formula with configurable limit
        table_name = ctx.context.table_name
        try:
            tool_config = get_tool_config()
            max_results = tool_config["max_map_results"]
        except:
            max_results = 50
        
        sql_query = f"""
        SELECT 
            id, nama_objek, pemberi_tugas, latitude, longitude,
            wadmpr, wadmkk, wadmkc, jenis_objek_text, status_text, cabang_text,
            (6371 * acos(
                cos(radians({lat})) * cos(radians(latitude)) * 
                cos(radians(longitude) - radians({lng})) + 
                sin(radians({lat})) * sin(radians(latitude))
            )) as distance_km
        FROM {table_name}
        WHERE 
            latitude IS NOT NULL AND longitude IS NOT NULL
            AND latitude != 0 AND longitude != 0
            AND (6371 * acos(
                cos(radians({lat})) * cos(radians(latitude)) * 
                cos(radians(longitude) - radians({lng})) + 
                sin(radians({lat})) * sin(radians(latitude))
            )) <= {request.radius_km}
        ORDER BY distance_km ASC
        LIMIT {max_results}
        """
        
        # Execute query
        db_conn = ctx.context.db_connection
        result_df, query_msg = db_conn.execute_query(sql_query)
        
        if result_df is not None and len(result_df) > 0:
            # Create enhanced map with reference point
            fig = go.Figure()
            
            # Add reference point (target location)
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lng],
                mode='markers',
                marker=dict(size=15, color='blue', symbol='star'),
                text=[f"📍 {request.location_name}<br>{formatted_address}"],
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
                title=f"{request.title} - {len(result_df)} projects within {request.radius_km} km from {request.location_name}"
            )
            
            # Display map
            st.plotly_chart(fig, use_container_width=True)
            
            # Store results in context
            ctx.context.last_map_data = result_df.copy()
            ctx.context.last_query_result = result_df.copy()
            
            # Show results table
            with st.expander("📊 Detail Nearby Projects", expanded=False):
                st.dataframe(result_df[['id', 'nama_objek', 'pemberi_tugas', 'jenis_objek_text', 
                            'wadmpr', 'wadmkk', 'distance_km']].round(2), 
                use_container_width=True)
            
            return f"✅ Found {len(result_df)} projects within {request.radius_km} km from {request.location_name}. Closest project is {result_df['distance_km'].min():.2f} km away."
        
        else:
            return f"❌ No projects found within {request.radius_km} km from {request.location_name}."
            
    except Exception as e:
        return f"Error finding nearby projects: {str(e)}"

# Configuration helper functions
def get_model_config(agent_type: str) -> str:
    """Get model configuration from secrets"""
    try:
        return st.secrets["agents"][f"{agent_type}_model"]
    except KeyError:
        # Fallback to default models
        defaults = {
            "manager": "o4-mini",
            "sql": "o4-mini", 
            "visualization": "o4-mini",
            "explanation": "gpt-4.1-mini"
        }
        return defaults.get(agent_type, "o4-mini")

def get_agent_settings() -> dict:
    """Get agent behavior settings from secrets"""
    try:
        return {
            "max_turns": st.secrets["agents"]["max_turns"],
            "temperature": st.secrets["agents"]["temperature"]
        }
    except KeyError:
        # Fallback defaults
        return {
            "max_turns": 10,
            "temperature": 0.3
        }

def get_tool_config() -> dict:
    """Get tool configuration from secrets"""
    try:
        return {
            "max_map_results": st.secrets["tools"]["max_map_results"],
            "max_chart_data_points": st.secrets["tools"]["max_chart_data_points"],
            "default_radius_km": st.secrets["tools"]["default_radius_km"]
        }
    except KeyError:
        # Fallback defaults
        return {
            "max_map_results": 50,
            "max_chart_data_points": 1000,
            "default_radius_km": 1.0
        }

# Agent definitions with secrets integration
def create_sql_agent() -> Agent[RHRContext]:
    """Create SQL specialist agent with configurable model"""
    model = get_model_config("sql")
    settings = get_agent_settings()
    
    return Agent[RHRContext](
        name="SQL Agent",
        instructions="""You are an expert SQL agent for the RHR property appraisal database.
        
        Your role is to:
        1. Convert natural language queries into optimized PostgreSQL queries
        2. Handle geographic filtering and location-based searches
        3. Generate appropriate queries for data retrieval, counting, grouping, and analysis
        4. Return structured SQLQueryRequest objects
        
        TABLE: The main table contains property appraisal project data with these key columns:
        - id (int8): Primary key
        - sumber, pemberi_tugas, no_kontrak: Project info
        - nama_lokasi, alamat_lokasi: Location details
        - objek_penilaian, nama_objek, jenis_objek_text: Property details
        - status_text, cabang_text, jc_text: Status and management
        - latitude, longitude: Coordinates
        - wadmpr, wadmkk, wadmkc: Administrative regions (Province, Regency, District)
        
        CRITICAL SQL RULES:
        1. Always filter out NULL, empty strings, and 'NULL' text values
        2. Use ILIKE for case-insensitive text searches
        3. For geographic queries: Use coordinate filters and administrative region filters
        4. Always include LIMIT to prevent large result sets
        5. For maps: Include id, latitude, longitude, and descriptive columns
        6. Handle reference queries (first, last, biggest, etc.) from previous results
        
        Always respond with a SQLQueryRequest object containing the optimized query.""",
        model=model,
        model_settings={
            "temperature": settings["temperature"]  # Only temperature, no api_key
        },
        tools=[execute_sql_query],
        output_type=SQLQueryRequest
    )

def create_visualization_agent() -> Agent[RHRContext]:
    """Create visualization specialist agent with configurable model"""
    model = get_model_config("visualization")
    settings = get_agent_settings()
    
    return Agent[RHRContext](
        name="Visualization Agent",
        instructions="""You are a data visualization expert for the RHR system.
        
        Your role is to:
        1. Create maps, charts, and other visualizations from query data
        2. Handle location-based visualizations and nearby project searches
        3. Choose appropriate visualization types based on data characteristics
        4. Generate engaging and informative visual presentations
        
        Available tools:
        - create_map_visualization: For property location maps
        - create_chart_visualization: For bar, pie, line, scatter, histogram charts
        - find_nearby_projects: For location-based proximity searches
        
        Always choose the most appropriate visualization type based on:
        - Data characteristics (numeric vs categorical)
        - User intent (comparison, distribution, location, trends)
        - Data volume and complexity
        
        Provide clear, informative visualizations with proper titles and labeling.""",
        model=model,
        model_settings={
            "temperature": settings["temperature"]
        },
        tools=[create_map_visualization, create_chart_visualization, find_nearby_projects]
    )

def create_explanation_agent() -> Agent[RHRContext]:
    """Create explanation and chat agent with configurable model"""
    model = get_model_config("explanation")
    settings = get_agent_settings()
    
    return Agent[RHRContext](
        name="Explanation Agent",
        instructions="""You are the explanation and communication expert for the RHR system.
        
        Your role is to:
        1. Interpret data and query results for business users
        2. Provide clear explanations in Bahasa Indonesia
        3. Generate insights and actionable recommendations
        4. Handle conversational interactions and follow-up questions
        5. Summarize findings and provide business context
        
        Always respond in Bahasa Indonesia with:
        - Clear, non-technical language
        - Business insights and implications
        - Actionable recommendations when appropriate
        - Professional and helpful tone
        
        Focus on business value and practical insights rather than technical details.""",
        model=model,
        model_settings={
            "temperature": settings["temperature"]
        }
    )

def create_manager_agent() -> Agent[RHRContext]:
    """Create the main manager agent that orchestrates everything"""
    model = get_model_config("manager")
    settings = get_agent_settings()
    
    sql_agent = create_sql_agent()
    visualization_agent = create_visualization_agent()
    explanation_agent = create_explanation_agent()
    
    return Agent[RHRContext](
        name="Manager Agent",
        instructions="""You are the Manager Agent for the RHR AI Query system.
        
        Your role is to:
        1. Understand user requests and determine the best approach
        2. Orchestrate SQL Agent for data queries
        3. Orchestrate Visualization Agent for maps and charts
        4. Orchestrate Explanation Agent for final responses
        5. Maintain conversation context and memory
        6. Handle complex multi-step workflows
        
        WORKFLOW DECISION TREE:
        1. For data queries → Hand off to SQL Agent
        2. For maps/charts/visualizations → Hand off to Visualization Agent  
        3. For explanations and conversations → Hand off to Explanation Agent
        4. For complex requests → Coordinate multiple agents in sequence
        
        IMPORTANT: 
        - Always maintain context between agents
        - Ensure proper handoffs with relevant data
        - Coordinate the complete workflow from request to final response
        - Remember previous interactions and results
        
        You are the orchestrator and decision maker.""",
        model=model,
        model_settings={
            "temperature": settings["temperature"]
        },
        handoffs=[
            handoff(sql_agent, tool_name_override="query_database"),
            handoff(visualization_agent, tool_name_override="create_visualization"), 
            handoff(explanation_agent, tool_name_override="explain_results")
        ]
    )

# Initialize system functions
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

def initialize_context() -> RHRContext:
    """Initialize the shared context for all agents with secrets"""
    if 'rhr_context' not in st.session_state:
        st.session_state.rhr_context = RHRContext()
    
    context = st.session_state.rhr_context
    
    # Initialize database connection
    if context.db_connection is None:
        context.db_connection = DatabaseConnection()
        
        try:
            # Load database config from secrets
            db_user = st.secrets["database"]["user"]
            db_pass = st.secrets["database"]["password"]
            db_host = st.secrets["database"]["host"]
            db_port = st.secrets["database"]["port"]
            db_name = st.secrets["database"]["name"]
            context.table_name = st.secrets["database"]["table_name"]
            
            success, message = context.db_connection.connect(
                db_user, db_pass, db_host, db_port, db_name
            )
            
            if not success:
                st.error(f"Database connection failed: {message}")
        except KeyError as e:
            st.error(f"Missing database configuration in secrets: {e}")
    
    # Initialize geocoding service
    if context.geocode_service is None:
        try:
            google_api_key = st.secrets["google"]["api_key"]
            context.geocode_service = GeocodeService(google_api_key)
        except KeyError:
            st.warning("Google Maps API key not found in secrets. Location search features will be unavailable.")
    
    # Initialize other context variables
    if context.geographic_filters is None:
        context.geographic_filters = {}
    
    if context.chat_history is None:
        context.chat_history = []
    
    return context

async def run_agent_query(user_input: str, context: RHRContext):
    """Run the agent system with user input and configurable settings"""
    try:
        settings = get_agent_settings()
        manager_agent = create_manager_agent()

        # Call Runner.run with correct parameter structure
        result = await Runner.run(
            agent=manager_agent,          # Use keyword argument
            input=user_input,             # Use keyword argument
            context=context,              # Context as keyword
            max_turns=settings["max_turns"]
        )
        return result.final_output

    except Exception as e:
        return f"Error running agent system: {str(e)}"


def render_agent_chat():
    """Render the new agent-based chat interface"""
    st.markdown('<div class="section-header">RHR AI Agents</div>', unsafe_allow_html=True)
    
    # Initialize context
    context = initialize_context()
    
    # Check if database is connected
    if not context.db_connection or not context.db_connection.connection_status:
        st.error("Database connection is required. Please check your configuration.")
        return
    
    # Display system status
    col1, col2, col3 = st.columns(3)
    with col1:
        if context.db_connection.connection_status:
            st.success("🗄️ Database Connected")
        else:
            st.error("🗄️ Database Disconnected")
    
    with col2:
        if context.geocode_service:
            st.success("🌍 Geocoding Available")
        else:
            st.warning("🌍 Geocoding Unavailable")
    
    with col3:
        if context.geographic_filters:
            filter_count = sum(len(v) for v in context.geographic_filters.values() if v)
            st.info(f"📍 Filters: {filter_count}")
        else:
            st.info("📍 No Filters")
    
    # Initialize chat history
    if 'agent_chat_messages' not in st.session_state:
        st.session_state.agent_chat_messages = []
        # Add welcome message
        welcome_msg = """Halo! Saya adalah sistem AI Agent RHR yang baru dan lebih canggih.

Saya terdiri dari beberapa agent specialist:
- **Manager Agent**: Mengatur dan mengoordinasi semua agent
- **SQL Agent**: Specialist dalam query database dan analisis data
- **Visualization Agent**: Expert dalam pembuatan peta dan grafik
- **Explanation Agent**: Ahli dalam menjelaskan hasil dan insights

Anda dapat menanyakan hal-hal seperti:
- "Berapa banyak proyek yang kita miliki di Jakarta?"
- "Buatkan peta proyek terdekat dari Setiabudi One dengan radius 1 km"
- "Buatkan grafik pemberi tugas di tiap cabang"
- "Siapa 5 klien utama kita dan bagaimana distribusinya?"

Apa yang ingin Anda ketahui tentang proyek Anda?"""
        
        st.session_state.agent_chat_messages.append({
            "role": "assistant",
            "content": welcome_msg
        })
    
    # Display chat history
    for message in st.session_state.agent_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your projects..."):
        # Add user message
        st.session_state.agent_chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            try:
                # Update context with latest session state data
                context.geographic_filters = st.session_state.get('geographic_filters', {})
                context.last_query_result = st.session_state.get('last_query_result', None)
                context.last_map_data = st.session_state.get('last_map_data', None)
                
                # Show thinking indicator
                with st.spinner("🤖 Agent system is thinking..."):
                    # Run async agent system
                    response = asyncio.run(run_agent_query(prompt, context))
                
                # Display response
                st.markdown(response)
                
                # Update session state with context changes
                st.session_state.last_query_result = context.last_query_result
                st.session_state.last_map_data = context.last_map_data
                
                # Add assistant response to history
                st.session_state.agent_chat_messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"Maaf, terjadi kesalahan dalam sistem agent: {str(e)}"
                st.error(error_msg)
                st.session_state.agent_chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
    
    # Chat management
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.agent_chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("Show Context", use_container_width=True):
            with st.expander("📋 Current Context", expanded=True):
                st.json({
                    "geographic_filters": context.geographic_filters,
                    "has_last_query": context.last_query_result is not None,
                    "has_map_data": context.last_map_data is not None,
                    "table_name": context.table_name,
                    "chat_history_length": len(st.session_state.agent_chat_messages)
                })
    
    with col3:
        if st.button("Export Chat", use_container_width=True):
            chat_export = {
                "timestamp": datetime.now().isoformat(),
                "geographic_filters": context.geographic_filters,
                "chat_messages": st.session_state.agent_chat_messages,
                "system_info": {
                    "framework": "OpenAI Agents",
                    "agents": ["Manager", "SQL", "Visualization", "Explanation"]
                }
            }
            
            st.download_button(
                label="Download Agent Chat History",
                data=json.dumps(chat_export, indent=2),
                file_name=f"agent_chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )

def render_geographic_filter():
    """Render geographic filtering interface"""
    st.markdown('<div class="section-header">Geographic Filter</div>', unsafe_allow_html=True)
    
    context = initialize_context()
    
    if not context.db_connection or not context.db_connection.connection_status:
        st.error("Database connection is required for geographic filtering.")
        return
    
    st.markdown("Select geographic areas to help AI agents focus on specific regions (optional)")
    
    # Geographic Filters Section
    st.markdown("#### Geographic Selection")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Province (wadmpr)**")
        if st.button("Load Provinces", key="load_provinces"):
            with st.spinner("Loading province list..."):
                province_options = context.db_connection.get_unique_geographic_values(
                    'wadmpr', table_name=context.table_name
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
                regency_options = context.db_connection.get_unique_geographic_values(
                    'wadmkk',
                    {'wadmpr': selected_provinces},
                    table_name=context.table_name
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
                district_options = context.db_connection.get_unique_geographic_values(
                    'wadmkc',
                    {'wadmkk': selected_regencies},
                    table_name=context.table_name
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
        st.info("No geographic filters applied. AI agents will search across all locations.")

def main():
    """Main application"""
    st.markdown('<h1 class="main-header">RHR AI Query Assistant - Agent Framework</h1>', unsafe_allow_html=True)
    
    # Check authentication
    if not check_authentication():
        login()
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Geographic Filter", "AI Agents"])
    
    # Show current user
    st.sidebar.markdown("---")
    try:
        username = st.secrets['auth']['username']
        st.sidebar.success(f"Logged in as: {username}")
    except KeyError:
        st.sidebar.success("Logged in as: User")

    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    # Framework info with configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🤖 Agent Framework**")

    # Show current model configuration
    try:
        st.sidebar.info(f"""
        **Current Models:**
        - Manager: {get_model_config("manager")}
        - SQL: {get_model_config("sql")}
        - Visualization: {get_model_config("visualization")}
        - Explanation: {get_model_config("explanation")}
        
        **Settings:**
        - Max Turns: {get_agent_settings()["max_turns"]}
        - Temperature: {get_agent_settings()["temperature"]}
        """)
    except Exception:
        st.sidebar.info("""
        This app uses OpenAI Agents framework:
        - Manager Agent (Orchestrator)
        - SQL Agent (Database Specialist) 
        - Visualization Agent (Maps & Charts)
        - Explanation Agent (Communication)
        """)

    
    # Configuration status
    st.sidebar.markdown("**⚙️ Configuration Status**")
    try:
        # Check if all required secrets are present
        missing_configs = []
        
        # Check OpenAI API key
        try:
            api_key = st.secrets["openai"]["api_key"]
            if api_key and len(api_key) > 10:
                st.sidebar.success("✅ OpenAI API Key")
            else:
                missing_configs.append("OpenAI API Key")
                st.sidebar.error("❌ Invalid OpenAI API Key")
        except KeyError:
            missing_configs.append("OpenAI API Key")
            st.sidebar.error("❌ OpenAI API Key Missing")

        
        # Check database config
        try:
            db_user = st.secrets["database"]["user"]
            table_name = st.secrets["database"]["table_name"]
            if db_user and table_name:
                st.sidebar.success("✅ Database Config")
            else:
                missing_configs.append("Database Config")
                st.sidebar.error("❌ Invalid Database Config")
        except KeyError:
            missing_configs.append("Database Config")
            st.sidebar.error("❌ Database Config Missing")
        
        # Check Google Maps API (optional)
        try:
            google_key = st.secrets["google"]["api_key"]
            if google_key and len(google_key) > 10:
                st.sidebar.success("✅ Google Maps API")
            else:
                st.sidebar.warning("⚠️ Invalid Google Maps API")
        except KeyError:
            st.sidebar.warning("⚠️ Google Maps API (Optional)")
        
        # Check agent configuration
        try:
            manager_model = st.secrets["agents"]["manager_model"]
            if manager_model:
                st.sidebar.success("✅ Agent Models Config")
            else:
                st.sidebar.info("ℹ️ Using Default Agent Models")
        except KeyError:
            st.sidebar.info("ℹ️ Using Default Agent Models")
        
        if missing_configs:
            st.sidebar.error(f"Missing: {', '.join(missing_configs)}")
            
    except Exception as e:
        st.sidebar.error("❌ Configuration Error")
    
    # Render selected page
    if page == "Geographic Filter":
        render_geographic_filter()
    elif page == "AI Agents":
        render_agent_chat()
    
    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status**")
    
    # Initialize context for status display
    try:
        context = initialize_context()
        
        # Database status
        if context.db_connection and context.db_connection.connection_status:
            st.sidebar.success("🗄️ Database Connected")
        else:
            st.sidebar.error("🗄️ Database Disconnected")
        
        # Geocoding service status
        if context.geocode_service:
            st.sidebar.success("🌍 Geocoding Available")
        else:
            st.sidebar.warning("🌍 Geocoding Unavailable")
        
        # Geographic filters status
        if hasattr(st.session_state, 'geographic_filters') and any(st.session_state.geographic_filters.values()):
            filters = st.session_state.geographic_filters
            filter_count = sum(len(v) for v in filters.values() if v)
            st.sidebar.success(f"📍 Geographic Filters: {filter_count}")
        else:
            st.sidebar.info("📍 No Geographic Filters")
        
        # Chat status
        if hasattr(st.session_state, 'agent_chat_messages'):
            st.sidebar.info(f"💬 Chat Messages: {len(st.session_state.agent_chat_messages)}")
    
    except Exception as e:
        st.sidebar.error(f"⚠️ System Error: {str(e)}")

if __name__ == "__main__":
    main()