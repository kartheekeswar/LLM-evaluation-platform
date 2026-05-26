import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
import mlflow
import mlflow.pyfunc
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
try:
    mlflow.set_experiment("llm-comparison-experiment")
except:
    pass

# Available models
MODELS = {
    "llama-3.1-8b-instant": "Llama-3.1-8B",
    "llama-3.3-70b-versatile": "Llama-3.3-70B",
    "openai/gpt-oss-120b": "GPT-OSS-120B",
    "openai/gpt-oss-20b": "GPT-OSS-20B",
    "qwen/qwen3-32b": "Qwen3-32B",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama-4-Scout-17B"
}

# PERSISTENT STORAGE FILE
HISTORY_FILE = "comparison_history.csv"

# Load/Save Functions
def load_history():
    """Load comparison history from CSV"""
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        except:
            return pd.DataFrame(columns=['Prompt', 'Model', 'Response', 'Response Time (s)', 'Tokens', 'Timestamp'])
    else:
        return pd.DataFrame(columns=['Prompt', 'Model', 'Response', 'Response Time (s)', 'Tokens', 'Timestamp'])

def save_to_history(prompt, results):
    """Save comparison to CSV file"""
    new_data = []
    for result in results:
        new_data.append({
            'Prompt': prompt,
            'Model': MODELS[result['model']],
            'Response': result['response'][:500] if len(result['response']) > 500 else result['response'],
            'Response Time (s)': result['response_time'],
            'Tokens': result['tokens'],
            'Timestamp': result['timestamp']
        })
    
    new_df = pd.DataFrame(new_data)
    
    if os.path.exists(HISTORY_FILE):
        existing_df = pd.read_csv(HISTORY_FILE)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(HISTORY_FILE, index=False)
    else:
        new_df.to_csv(HISTORY_FILE, index=False)

# Page config
st.set_page_config(
    page_title="LLM Comparison Platform",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 LLM Comparison Platform")
st.markdown("Compare responses from multiple Large Language Models")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Select Models to Compare")
    
    model1 = st.selectbox(
        "Model 1:",
        options=list(MODELS.keys()),
        format_func=lambda x: MODELS[x],
        index=0,
        key="model1"
    )
    
    model2 = st.selectbox(
        "Model 2:",
        options=list(MODELS.keys()),
        format_func=lambda x: MODELS[x],
        index=1,
        key="model2"
    )
    
    model3 = st.selectbox(
        "Model 3:",
        options=list(MODELS.keys()),
        format_func=lambda x: MODELS[x],
        index=2,
        key="model3"
    )
    
    st.divider()
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=4096,
        value=1024,
        step=100
    )
    
    st.divider()
    
    page = st.radio("Navigation", ["Compare Models", "Analytics", "Export"])

# Initialize session state for current comparison
if 'current_comparison' not in st.session_state:
    st.session_state.current_comparison = None

# Function to get LLM response
def get_llm_response(model, prompt, temperature, max_tokens):
    """Get response from Groq LLM"""
    try:
        start_time = datetime.now()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        return {
            "response": response.choices[0].message.content,
            "model": model,
            "response_time": response_time,
            "tokens": response.usage.total_tokens,
            "timestamp": datetime.now()
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "model": model,
            "response_time": 0,
            "tokens": 0,
            "timestamp": datetime.now()
        }

# Function to log to MLflow
def log_comparison_to_mlflow(prompt, results):
    """Log comparison to MLflow"""
    with mlflow.start_run():
        mlflow.log_param("prompt", prompt)
        mlflow.log_param("temperature", temperature)
        mlflow.log_param("max_tokens", max_tokens)
        
        for i, result in enumerate(results, 1):
            mlflow.log_metric(f"model_{i}_response_time", result['response_time'])
            mlflow.log_metric(f"model_{i}_tokens", result['tokens'])
            mlflow.log_param(f"model_{i}_name", result['model'])

# PAGE 1: COMPARE MODELS
if page == "Compare Models":
    st.header("🔄 Compare LLM Responses")
    
    selected_models = [model1, model2, model3]
    
    if len(set(selected_models)) != 3:
        st.warning("⚠️ Please select 3 different models for comparison")
    
    st.info(f"**Selected Models:** {MODELS[model1]} | {MODELS[model2]} | {MODELS[model3]}")
    
    prompt = st.text_area(
        "Enter your prompt:",
        height=150,
        placeholder="Ask anything..."
    )
    
    compare_button = st.button("🚀 Compare Models", type="primary", use_container_width=True)
    
    if compare_button and prompt:
        if len(set(selected_models)) != 3:
            st.error("❌ Please select 3 different models!")
        else:
            with st.spinner("Generating responses..."):
                cols = st.columns(3)
                results = []
                
                for idx, (col, model_id) in enumerate(zip(cols, selected_models)):
                    with col:
                        st.subheader(f"🤖 {MODELS[model_id]}")
                        
                        result = get_llm_response(model_id, prompt, temperature, max_tokens)
                        results.append(result)
                        
                        st.markdown("**Response:**")
                        st.write(result['response'])
                        
                        st.divider()
                        metric_col1, metric_col2 = st.columns(2)
                        with metric_col1:
                            st.metric("⏱️ Time", f"{result['response_time']:.2f}s")
                        with metric_col2:
                            st.metric("🎫 Tokens", result['tokens'])
                
                # Save current comparison for export
                st.session_state.current_comparison = {
                    'prompt': prompt,
                    'results': results,
                    'timestamp': datetime.now()
                }
                
                # Save to history
                save_to_history(prompt, results)
                
                # Log to MLflow
                log_comparison_to_mlflow(prompt, results)
                
                st.success("✅ Comparison complete and saved!")
    
    elif compare_button and not prompt:
        st.warning("⚠️ Please enter a prompt first!")

# PAGE 2: ANALYTICS
elif page == "Analytics":
    st.header("📊 Analytics Dashboard")
    
    df = load_history()
    
    if df.empty:
        st.info("No comparison data yet. Run some comparisons first!")
    else:
        total_comparisons = len(df['Prompt'].unique())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Comparisons", total_comparisons)
        with col2:
            st.metric("Avg Response Time", f"{df['Response Time (s)'].mean():.2f}s")
        with col3:
            st.metric("Total Tokens Used", f"{df['Tokens'].sum():,}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.box(
                df,
                x='Model',
                y='Response Time (s)',
                title='Response Time Distribution by Model',
                color='Model'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                df.groupby('Model')['Tokens'].mean().reset_index(),
                x='Model',
                y='Tokens',
                title='Average Token Usage by Model',
                color='Model'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("📋 Detailed Results")
        st.dataframe(df, use_container_width=True)
        
        st.info(f"📊 Community Stats: {total_comparisons} comparisons from all users")
        
        with st.expander("🔧 Admin Actions"):
            if st.button("🗑️ Clear All Data"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.success("✅ All data cleared!")
                st.rerun()

# PAGE 3: EXPORT
elif page == "Export":
    st.header("📥 Export Current Comparison")
    
    if st.session_state.current_comparison is None:
        st.info("No comparison to export yet. Run a comparison first!")
    else:
        comp = st.session_state.current_comparison
        
        st.write(f"**Prompt:** {comp['prompt']}")
        st.write(f"**Models Compared:** {len(comp['results'])}")
        st.write(f"**Date:** {comp['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.button("📄 Generate PDF Report"):
            try:
                import re
                
                # ========== HELPER FUNCTION ==========
                def clean_response_for_pdf(text):
                    """Clean markdown and special characters from response"""
                    text = re.sub(r'#{1,6}\s+', '', text)
                    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                    
                    lines = text.split('\n')
                    cleaned_lines = []
                    in_table = False
                    
                    for line in lines:
                        if re.match(r'^\s*\|[-\s|:]+\|\s*$', line):
                            in_table = True
                            continue
                        
                        if '|' in line and in_table:
                            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                            if cells:
                                cleaned_lines.append('  • ' + ' | '.join(cells))
                        elif '|' in line:
                            cleaned_lines.append(line.replace('|', ' - '))
                        else:
                            in_table = False
                            cleaned_lines.append(line)
                    
                    text = '\n'.join(cleaned_lines)
                    text = text.replace('■', '-')
                    text = text.replace('─', '-')
                    text = text.replace('│', '|')
                    text = re.sub(r'^[-_]{3,}$', '', text, flags=re.MULTILINE)
                    text = re.sub(r' {2,}', ' ', text)
                    text = re.sub(r'^\d+\.\s+', '• ', text, flags=re.MULTILINE)
                    
                    return text
                
                # ========== CREATE PDF ==========
                pdf_filename = "comparison_report.pdf"
                doc = SimpleDocTemplate(
                    pdf_filename, 
                    pagesize=letter,
                    rightMargin=50,
                    leftMargin=50,
                    topMargin=50,
                    bottomMargin=30
                )
                elements = []
                styles = getSampleStyleSheet()
                
                # Styles
                title_style = ParagraphStyle(
                    'Title',
                    parent=styles['Heading1'],
                    fontSize=22,
                    textColor=colors.HexColor('#1a1a1a'),
                    spaceAfter=20,
                    alignment=1,
                    fontName='Helvetica-Bold'
                )
                
                section_header = ParagraphStyle(
                    'SectionHeader',
                    parent=styles['Heading2'],
                    fontSize=14,
                    textColor=colors.HexColor('#2c3e50'),
                    spaceAfter=10,
                    spaceBefore=15,
                    fontName='Helvetica-Bold'
                )
                
                model_header = ParagraphStyle(
                    'ModelHeader',
                    parent=styles['Normal'],
                    fontSize=12,
                    textColor=colors.white,
                    spaceAfter=8,
                    spaceBefore=12,
                    fontName='Helvetica-Bold',
                    backColor=colors.HexColor('#3498db'),
                    borderPadding=6,
                    leftIndent=8
                )
                
                body_style = ParagraphStyle(
                    'BodyText',
                    parent=styles['Normal'],
                    fontSize=9,
                    leading=13,
                    textColor=colors.HexColor('#2c3e50'),
                    spaceAfter=8
                )
                
                # Title
                title = Paragraph("LLM Comparison Report", title_style)
                elements.append(title)
                
                date_style = ParagraphStyle(
                    'Date',
                    parent=styles['Normal'],
                    fontSize=9,
                    textColor=colors.grey,
                    alignment=1
                )
                date_text = Paragraph(
                    f"Generated: {comp['timestamp'].strftime('%B %d, %Y at %I:%M %p')}",
                    date_style
                )
                elements.append(date_text)
                elements.append(Spacer(1, 15))
                
                # Summary
                summary_header = Paragraph("<b>Comparison Summary</b>", section_header)
                elements.append(summary_header)
                
                total_tokens = sum([r['tokens'] for r in comp['results']])
                avg_time = sum([r['response_time'] for r in comp['results']]) / len(comp['results'])
                
                # FIXED SUMMARY DATA
                prompt_display = comp['prompt'][:80] + '...' if len(comp['prompt']) > 80 else comp['prompt']
                models_display = ', '.join([MODELS[r['model']] for r in comp['results']])
                
                summary_data = [
                    ['Prompt:', prompt_display],
                    ['Models:', models_display],
                    ['Temperature:', str(temperature)],
                    ['Max Tokens:', str(max_tokens)],
                    ['Total Tokens:', f"{total_tokens:,}"],
                    ['Avg Time:', f"{avg_time:.2f}s"]
                ]
                
                summary_table = Table(summary_data, colWidths=[1.3*inch, 5.2*inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ]))
                
                elements.append(summary_table)
                elements.append(Spacer(1, 20))
                
                # Model Responses
                responses_header = Paragraph("<b>Model Responses</b>", section_header)
                elements.append(responses_header)
                elements.append(Spacer(1, 8))
                
                for idx, result in enumerate(comp['results'], 1):
                    # Model header
                    model_name = Paragraph(
                        f"  Model {idx}: {MODELS[result['model']]}  ",
                        model_header
                    )
                    elements.append(model_name)
                    elements.append(Spacer(1, 6))
                    
                    # CLEAN and format response
                    raw_response = result['response']
                    cleaned_response = clean_response_for_pdf(raw_response)
                    response_text = cleaned_response.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
                    response_para = Paragraph(response_text, body_style)
                    elements.append(response_para)
                    elements.append(Spacer(1, 10))
                    
                    # Metrics
                    metrics_data = [
                        ['Response Time', 'Tokens Used'],
                        [f"{result['response_time']:.2f}s", f"{result['tokens']:,}"]
                    ]
                    
                    metrics_table = Table(metrics_data, colWidths=[3.25*inch, 3.25*inch])
                    metrics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ecf0f1')),
                        ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#2c3e50')),
                        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
                        ('FONTSIZE', (0, 1), (-1, 1), 9),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    
                    elements.append(metrics_table)
                    
                    if idx < len(comp['results']):
                        elements.append(Spacer(1, 15))
                        line_data = [['_' * 100]]
                        line_table = Table(line_data, colWidths=[6.5*inch])
                        line_table.setStyle(TableStyle([
                            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#bdc3c7')),
                            ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ]))
                        elements.append(line_table)
                        elements.append(Spacer(1, 15))
                
                doc.build(elements)
                
                st.success("✅ PDF generated successfully!")
                
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_file,
                        file_name="llm_comparison_report.pdf",
                        mime="application/pdf"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error generating PDF: {str(e)}")