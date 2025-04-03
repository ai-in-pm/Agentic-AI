# Agentic AI

This application demonstrates the core capabilities of an Agentic AI system through a step-by-step implementation. The agent processes input, reasons about it, plans actions, and executes them in real-time.

## Features

The agent consists of four main modules:

1. **Perception Module**
   - Processes real-time input data from text
   - Implements pattern recognition and feature extraction
   - Converts raw inputs into structured representations

2. **Reasoning Module**
   - Analyzes processed data using logical inference
   - Applies domain knowledge and rules
   - Generates hypotheses and conclusions based on available information
   - Handles uncertainty and probabilistic reasoning

3. **Planning Module**
   - Defines clear goals and success criteria
   - Generates multiple possible action sequences
   - Evaluates options using cost-benefit analysis
   - Creates optimal execution plans with contingencies
   - Adapts plans based on changing conditions

4. **Action Module**
   - Executes planned actions in real-time
   - Monitors action outcomes and effectiveness
   - Provides feedback mechanisms
   - Implements error handling and recovery

## Technical Implementation

- **Backend**: Python with FastAPI
- **Frontend**: Streamlit for interactive visualization
- **Architecture**: Modular design for easy component updates
- **Visualization**: Real-time visualization of the agent's decision-making process
- **Monitoring**: Comprehensive logging and performance metrics

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd agentic-ai-demo
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Run the main application:
```
python main.py
```

This will start both the FastAPI backend server and the Streamlit frontend.

- The API server will be available at: http://localhost:8000
- The Streamlit frontend will be available at: http://localhost:8501

## Usage

1. Open the Streamlit frontend in your web browser
2. Enter a message in the input field
3. Submit your message to see the agent process it in real-time
4. Observe how each module (Perception, Reasoning, Planning, Action) processes the input
5. View the visualizations to understand the agent's decision-making process

![AgenticAI](https://github.com/user-attachments/assets/61829457-9a0f-4547-a801-206ba7e1a18f)


## Architecture

The application follows a modular architecture:

```
agentic-ai-demo/
├── agent/                  # Agent core modules
│   ├── perception/         # Perception module
│   ├── reasoning/          # Reasoning module
│   ├── planning/           # Planning module
│   └── action/             # Action module
├── api/                    # FastAPI backend
├── frontend/               # Streamlit frontend
├── main.py                 # Main entry point
└── requirements.txt        # Dependencies
```

## Extending the Application

The modular architecture makes it easy to extend the application:

- Add new perception capabilities (e.g., image processing, audio analysis)
- Enhance the reasoning module with more sophisticated inference mechanisms
- Implement additional planning strategies
- Add new action capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is a demonstration of Agentic AI concepts
- Inspired by research in cognitive architectures and autonomous agents
