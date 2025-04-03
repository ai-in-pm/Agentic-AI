"""
Streamlit Frontend for Agentic AI Demo
This module provides the user interface for interacting with the agent.
"""
import streamlit as st
import requests
import json
import time
import os
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API endpoint
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

def start_frontend():
    """Start the Streamlit frontend."""
    import os
    port = os.getenv("FRONTEND_PORT", "8501")
    os.system(f"streamlit run frontend/app.py --server.port={port}")

# Set page config
st.set_page_config(
    page_title="Agentic AI Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("🤖 Agentic AI Demo")
st.markdown("""
This application demonstrates the core capabilities of an Agentic AI system through a step-by-step implementation.
The agent processes input, reasons about it, plans actions, and executes them in real-time.
""")

# Sidebar with module information
with st.sidebar:
    st.header("Agent Modules")

    st.subheader("1. Perception")
    st.info("Processes input data and converts it into structured representations.")

    st.subheader("2. Reasoning")
    st.info("Analyzes processed data using logical inference and domain knowledge.")

    st.subheader("3. Planning")
    st.info("Generates and evaluates action plans based on goals and available information.")

    st.subheader("4. Action")
    st.info("Executes planned actions and monitors outcomes.")

    st.divider()
    st.markdown("### Settings")
    show_details = st.checkbox("Show detailed module outputs", value=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_process' not in st.session_state:
    st.session_state.current_process = None

# Function to process user input
def process_input(user_input: str) -> Dict[str, Any]:
    """Send user input to the API and get the response."""
    try:
        response = requests.post(
            f"{API_URL}/process",
            json={"text": user_input}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error communicating with the API: {e}")
        return {
            "perception_result": {},
            "reasoning_result": {},
            "planning_result": {},
            "action_result": {},
            "final_response": "Error processing your request."
        }

# Create the main layout
col1, col2 = st.columns([2, 3])

# Input area
with col1:
    st.header("Input")
    user_input = st.text_area("Enter your message:", height=100)
    submit_button = st.button("Submit")

    # Chat history
    st.header("Chat History")
    chat_container = st.container(height=400)

# Visualization area
with col2:
    st.header("Agent Process Visualization")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Perception", "Exact Recall", "Reasoning", "Planning", "Action"])

    with tab1:
        perception_container = st.container(height=500)

    with tab2:
        recall_container = st.container(height=500)

    with tab3:
        reasoning_container = st.container(height=500)

    with tab4:
        planning_container = st.container(height=500)

    with tab5:
        action_container = st.container(height=500)

# Process user input when submit button is clicked
if submit_button and user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Show processing indicator
    with chat_container:
        with st.spinner("Agent is processing..."):
            # Process input through the agent pipeline
            start_time = time.time()

            # Process the input through the API
            try:
                # Call the API to process the input
                api_response = process_input(user_input)

                # Extract results from API response
                perception_result = api_response.get("perception_result", {})
                recall_result = api_response.get("recall_result", {"memories": []})
                reasoning_result = api_response.get("reasoning_result", {})
                planning_result = api_response.get("planning_result", {})
                action_result = api_response.get("action_result", {})
                final_response = api_response.get("final_response", "No response generated")
            except Exception as e:
                st.error(f"Error processing input: {e}")
                perception_result = {
                    "tokens": user_input.split(),
                    "entities": [],
                    "sentiment": {"label": "neutral", "score": 0.5},
                    "intent": {"type": "unknown", "confidence": 0.5}
                }

            with perception_container:
                st.subheader("Input Processing")
                st.json(perception_result)

                # Visualization
                st.subheader("Confidence Scores")
                # Extract confidence scores from perception result
                intent_confidence = perception_result.get("intent", {}).get("confidence", 0.5)
                entity_confidence = 0.7  # Default value if not available
                sentiment_score = perception_result.get("sentiment", {}).get("score", 0.5)

                df = pd.DataFrame({
                    'Metric': ['Intent Recognition', 'Entity Detection', 'Sentiment Analysis'],
                    'Confidence': [intent_confidence, entity_confidence, sentiment_score]
                })
                fig = go.Figure(go.Bar(
                    x=df['Confidence'],
                    y=df['Metric'],
                    orientation='h',
                    marker_color='lightblue'
                ))
                fig.update_layout(title="Perception Confidence Metrics")
                st.plotly_chart(fig, use_container_width=True)

            with recall_container:
                st.subheader("Memory Recall")
                memories = recall_result.get("memories", [])
                query = recall_result.get("query", "")

                # Always show the query being processed
                st.write(f"**Query:** {query}")

                # Display memory system status
                st.json({
                    "query": query,
                    "memories_found": len(memories),
                    "timestamp": recall_result.get("timestamp", "")
                })

                if memories:
                    st.write(f"Found {len(memories)} relevant memories:")

                    # Display memories with similarity scores
                    for i, memory in enumerate(memories):
                        similarity = memory.get("similarity", 0)
                        content = memory.get("content", "")
                        created_at = memory.get("created_at", "Unknown time")

                        with st.expander(f"Memory {i+1} - Similarity: {similarity:.2f}"):
                            st.text(content)
                            st.caption(f"Created: {created_at}")

                    # Visualization of memory similarities
                    st.subheader("Memory Relevance")
                    memory_df = pd.DataFrame({
                        'Memory': [f"Memory {i+1}" for i in range(len(memories))],
                        'Similarity': [memory.get("similarity", 0) for memory in memories]
                    })

                    fig = go.Figure(go.Bar(
                        x=memory_df['Similarity'],
                        y=memory_df['Memory'],
                        orientation='h',
                        marker_color='lightgreen'
                    ))
                    fig.update_layout(title="Memory Similarity Scores")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No relevant memories found for this query.")

                    # Add a placeholder visualization for empty state
                    st.subheader("Memory System Status")
                    status_df = pd.DataFrame({
                        'Status': ['System Ready', 'Memories Available', 'Query Processed'],
                        'Value': [1.0, 0.0, 1.0 if query else 0.0]
                    })

                    fig = go.Figure(go.Bar(
                        x=status_df['Value'],
                        y=status_df['Status'],
                        orientation='h',
                        marker_color=['green', 'gray', 'blue' if query else 'gray']
                    ))
                    fig.update_layout(title="Memory System Status")
                    st.plotly_chart(fig, use_container_width=True)

            # Step 2: Reasoning is already loaded from API response

            with reasoning_container:
                st.subheader("Reasoning Process")
                st.json(reasoning_result)

                # Visualization
                st.subheader("Reasoning Analysis")
                # Extract confidence and uncertainty from reasoning result
                confidence = reasoning_result.get("confidence", 0.5)
                uncertainty = reasoning_result.get("uncertainty", 0.5)

                fig = go.Figure(go.Pie(
                    labels=['Confidence', 'Uncertainty'],
                    values=[confidence, uncertainty],
                    hole=.3
                ))
                fig.update_layout(title="Confidence vs. Uncertainty")
                st.plotly_chart(fig, use_container_width=True)

            # Step 3: Planning is already loaded from API response

            with planning_container:
                st.subheader("Action Plan")
                st.json(planning_result)

                # Visualization
                st.subheader("Action Sequence")
                # Extract actions from planning result
                actions = planning_result.get("primary_plan", {}).get("actions", [])
                # Convert actions to DataFrame format if needed
                action_data = []
                for i, action in enumerate(actions):
                    action_data.append({
                        "step": i + 1,
                        "action": action.get("action_id", ""),
                        "priority": action.get("priority", "medium")
                    })
                df = pd.DataFrame(action_data) if action_data else pd.DataFrame({"step": [], "action": [], "priority": []})
                fig = go.Figure()

                for i, row in df.iterrows():
                    color = 'red' if row['priority'] == 'high' else 'orange' if row['priority'] == 'medium' else 'green'
                    fig.add_trace(go.Scatter(
                        x=[row['step'], row['step']],
                        y=[0, 1],
                        mode='lines+markers',
                        name=row['action'],
                        line=dict(color=color, width=10),
                        marker=dict(size=15, symbol='circle')
                    ))

                fig.update_layout(
                    title="Action Sequence Plan",
                    xaxis_title="Step Number",
                    yaxis=dict(showticklabels=False),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

            # Step 4: Action is already loaded from API response

            with action_container:
                st.subheader("Action Execution")
                st.json(action_result)

                # Visualization
                st.subheader("Action Performance")
                # Extract executed actions from action result
                executed_actions = action_result.get("executed_actions", [])
                df = pd.DataFrame(executed_actions) if executed_actions else pd.DataFrame({"action": [], "status": [], "time_ms": []})
                fig = go.Figure(go.Bar(
                    x=df['action'],
                    y=df['time_ms'],
                    marker_color='lightgreen'
                ))
                fig.update_layout(title="Action Execution Time (ms)")
                st.plotly_chart(fig, use_container_width=True)

            # Calculate total processing time
            processing_time = time.time() - start_time

            # Add agent response to chat history
            st.session_state.chat_history.append({
                "role": "agent",
                "content": final_response,
                "processing_time": processing_time
            })

    # Clear the input area
    st.rerun()

# Display chat history
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Agent:** {message['content']}")
            if "processing_time" in message:
                st.caption(f"Processing time: {message['processing_time']:.2f} seconds")
        st.divider()

if __name__ == "__main__":
    pass  # The app is run by Streamlit directly
