import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import traceback
from datetime import datetime
import warnings

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
    """AI chatbot for database queries"""
    
    def __init__(self, api_key: str, table_name: str):
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=api_key,
            temperature=0.3,
            max_tokens=2000
        )
        self.table_name = table_name
        self.system_prompt = self.create_system_prompt()
    
    def create_system_prompt(self):
        return f"""
You are a database query assistant for RHR property appraisal company. You help users analyze their project data by generating and executing SQL queries.

TABLE NAME: {self.table_name}

COLUMN ANALYSIS:

Project Information:
- sumber = Data source (e.g., "kontrak" = contract-based projects)
- pemberi_tugas = Client/Task giver (e.g., "PT Asuransi Jiwa IFG", "PT Perkebunan Nusantara II")
- no_kontrak = Contract number (e.g., "RHR00C1P0623111.0")
- nama_lokasi = Location name (e.g., "Lokasi 20", "Lokasi 3")
- id = Unique project identifier (16316, 17122) [ALWAYS INCLUDE IN QUERIES]

Property Information:
- objek_penilaian = Appraisal object type (e.g., "real properti")
- nama_objek = Object name (e.g., "Rumah", "Tanah Kosong")
- jenis_objek = Object type code (13=House?, 1=Land?) [Codes will be provided]
- kepemilikan = Ownership type (e.g., "tunggal" = single ownership)
- keterangan = Additional notes (e.g., "Luas Tanah : 1.148")

Status & Management:
- status = Project status code (5, 4) [Codes will be provided]
- cabang = Branch office code (0, 1) [Codes will be provided]

Geographic Data:
- latitude = Latitude coordinates (-6.236, 3.662)
- longitude = Longitude coordinates (106.863, 98.656)
- geometry = PostGIS geometry field (binary spatial data)
- wadmpr = Province (e.g., "DKI Jakarta", "Sumatera Utara")
- wadmkk = Regency/City (e.g., "Kota Administrasi Jakarta Selatan", "Deli Serdang")
- wadmkc = District (e.g., "Tebet", "Labuhan Deli")

QUERY INSTRUCTIONS:

1. COUNTING/EXISTENCE QUERIES
For questions like: "Do I have projects in Jakarta?", "How many projects from this client?"
Template:
SELECT COUNT(*) as total_count
FROM {self.table_name}
WHERE [condition] AND [column] IS NOT NULL;

Alternative for existence check:
SELECT EXISTS(
    SELECT 1 FROM {self.table_name} 
    WHERE [condition] AND [column] IS NOT NULL
) as has_projects;

2. SUMMARY/GROUPING QUERIES
For questions like: "Top clients", "Projects by region", "Most common property types"
Template:
SELECT [grouping_column], COUNT(*) as count
FROM {self.table_name}
WHERE [column] IS NOT NULL
GROUP BY [grouping_column]
ORDER BY count DESC
LIMIT [reasonable_limit];

3. SAMPLE/EXAMPLE QUERIES
For questions like: "Show me some projects", "What kind of properties", "Examples of contracts"
Template:
SELECT id, [relevant_columns]
FROM {self.table_name}
WHERE [condition] AND [main_column] IS NOT NULL
ORDER BY id DESC
LIMIT [small_number];

4. GEOGRAPHIC QUERIES
For questions like: "Projects in specific locations", "Regional distribution"
Template:
SELECT wadmpr, wadmkk, wadmkc, COUNT(*) as count
FROM {self.table_name}
WHERE (wadmpr ILIKE '%[location]%' OR wadmkk ILIKE '%[location]%' OR wadmkc ILIKE '%[location]%')
AND wadmpr IS NOT NULL
GROUP BY wadmpr, wadmkk, wadmkc
ORDER BY count DESC;

5. CLIENT ANALYSIS QUERIES
For questions like: "Client performance", "Contract analysis"
Template:
SELECT pemberi_tugas, COUNT(*) as total_contracts, 
       COUNT(DISTINCT no_kontrak) as unique_contracts
FROM {self.table_name}
WHERE pemberi_tugas IS NOT NULL
GROUP BY pemberi_tugas
ORDER BY total_contracts DESC
LIMIT 10;

6. PROPERTY TYPE QUERIES
For questions like: "Property type distribution", "Object analysis"
Template:
SELECT jenis_objek, nama_objek, COUNT(*) as count
FROM {self.table_name}
WHERE jenis_objek IS NOT NULL AND nama_objek IS NOT NULL
GROUP BY jenis_objek, nama_objek
ORDER BY count DESC
LIMIT 15;

7. STATUS/WORKFLOW QUERIES
For questions like: "Project status", "Completion rate", "Branch performance"
Template:
SELECT status, cabang, COUNT(*) as count
FROM {self.table_name}
WHERE status IS NOT NULL AND cabang IS NOT NULL
GROUP BY status, cabang
ORDER BY count DESC;

CRITICAL RULES:
1. ALWAYS include id column in SELECT clause for reference
2. Handle NULL values with IS NOT NULL or COALESCE()
3. Use ILIKE for case-insensitive text matching
4. Add LIMIT to prevent overwhelming results
5. Use meaningful aliases (e.g., COUNT(*) as total_projects)
6. Group by relevant columns for summaries
7. Never use SELECT * unless specifically needed
8. Don't return massive datasets without LIMIT

RESPONSE FORMAT:
When executing queries, always provide:
1. Clear answer to user's question
2. Key numbers and insights
3. Reference IDs for important findings
4. Context about data quality (if many NULLs found)
5. Suggestions for follow-up questions

Generate SQL queries based on user questions and provide clear, actionable insights.

IMPORTANT: Always respond in Bahasa Indonesia. Provide natural, conversational answers in Indonesian language.
"""

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
        welcome_msg = """Halo! Saya RHR AI. Saya akan membantu Anda menganalisis proyek penilaian properti Anda.

Anda dapat bertanya hal-hal seperti:
- Berapa banyak proyek yang kita miliki di Jakarta?
- Siapa saja 5 klien terbesar kita?
- Tampilkan beberapa penilaian tanah terbaru.
- Jenis properti apa yang paling sering kita nilai?

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
                # Build message history
                messages = [SystemMessage(content=st.session_state.ai_chat.system_prompt)]
                
                # Add geographic context if available
                if hasattr(st.session_state, 'geographic_filters') and any(st.session_state.geographic_filters.values()):
                    geo_context = "Current geographic filters: "
                    filters = st.session_state.geographic_filters
                    if filters.get('wadmpr'):
                        geo_context += f"Provinces: {filters['wadmpr']} "
                    if filters.get('wadmkk'):
                        geo_context += f"Regencies: {filters['wadmkk']} "
                    if filters.get('wadmkc'):
                        geo_context += f"Districts: {filters['wadmkc']} "
                    
                    messages.append(SystemMessage(content=geo_context))
                
                # Add conversation history
                for msg in st.session_state.chat_messages:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                
                # Get AI response
                response = st.session_state.ai_chat.llm.invoke(messages)
                
                # If response contains a SQL query, execute it
                if "SELECT" in response.content.upper():
                    # Extract SQL query (simple extraction)
                    lines = response.content.split('\n')
                    sql_query = None
                    in_sql_block = False
                    
                    for line in lines:
                        if '```sql' in line.lower() or '```' in line and 'SELECT' in line.upper():
                            in_sql_block = True
                            continue
                        elif '```' in line and in_sql_block:
                            break
                        elif in_sql_block and line.strip():
                            if sql_query is None:
                                sql_query = line.strip()
                            else:
                                sql_query += " " + line.strip()
                    
                    # If no SQL block found, look for SELECT statements
                    if not sql_query:
                        for line in lines:
                            if line.strip().upper().startswith('SELECT'):
                                sql_query = line.strip()
                                break
                    
                    if sql_query:
                        try:
                            # Execute the query
                            result_df, query_msg = st.session_state.db_connection.execute_query(sql_query)
                            
                            if result_df is not None and len(result_df) > 0:
                                # Show query results in expandable section
                                with st.expander("📊 Query Results", expanded=False):
                                    st.code(sql_query, language="sql")
                                    st.dataframe(result_df, use_container_width=True)
                                
                                # Create a follow-up response with the actual results
                                follow_up_prompt = f"""
Based on the query results, provide a clear and conversational answer to the user's question.

Query executed: {sql_query}
Results: {result_df.to_dict('records')}

Please give a natural, helpful response that directly answers what the user asked, without showing technical details or raw data. Focus on the key insights and numbers that matter to the user.
"""
                                
                                # Get AI interpretation of results
                                interpretation_messages = [
                                    SystemMessage(content="You are a helpful assistant that interprets database query results for business users. Provide clear, conversational answers in Bahasa Indonesia without technical jargon."),
                                    HumanMessage(content=follow_up_prompt)
                                ]
                                
                                interpretation_response = st.session_state.ai_chat.llm.invoke(interpretation_messages)
                                
                                # Display the interpreted response
                                st.markdown("---")
                                st.markdown(interpretation_response.content)
                                
                                # Update the response content to include interpretation
                                response.content = interpretation_response.content
                                
                            else:
                                st.error(f"Query failed: {query_msg}")
                                response.content += f"\n\nNote: Query execution failed - {query_msg}"
                        
                        except Exception as e:
                            st.error(f"Error executing query: {str(e)}")
                            response.content += f"\n\nNote: Error executing query - {str(e)}"
                
                # # Display the response
                # st.markdown(response.content)
                
                # Add assistant response to history
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
            except Exception as e:
                error_msg = f"I encountered an error: {str(e)}\n\nPlease try rephrasing your question."
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
        if st.button("Geographic Distribution", use_container_width=True):
            quick_prompt = "Show me the distribution of projects by province"
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