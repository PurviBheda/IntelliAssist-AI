# 🤖 IntelliAssist – AI-Powered Intelligent Assistant

**IntelliAssist** is an AI-powered intelligent assistant designed to help users interact with documents, retrieve relevant information, generate summaries, and receive context-aware answers through natural language conversations.

It combines Google's Gemini AI with Retrieval-Augmented Generation (RAG), vector search, and document processing to deliver accurate, relevant, and context-aware responses.

---

## 🚀 Key Features

* **AI-Powered Chatbot:** Interactive chatbot supporting natural language conversations.
* **Document Intelligence:** Upload and process PDF, DOCX, and TXT documents.
* **RAG-Based Question Answering:** Retrieve relevant document information using FAISS and embeddings.
* **Context-Aware Responses:** Generate answers grounded in document content with citations.
* **Smart Summarization:** Generate concise and detailed document summaries.
* **Conversation History:** Store and manage previous conversations using SQLite.
* **Sentiment Analysis:** Identify the sentiment of user-provided text.
* **Intent Detection:** Understand user queries and route them to appropriate processing functions.
* **Multi-Turn Conversations:** Maintain conversational context across multiple interactions.
* **Fallback Mechanisms:** Provide extractive summarization when AI-based summarization is unavailable.

---

## 🛠️ Technology Stack

| Category               | Technologies                   |
| ---------------------- | ------------------------------ |
| Programming Language   | Python 3.12.4                  |
| AI Model               | Google Gemini 2.5 Flash        |
| AI Integration         | Google Gen AI SDK              |
| RAG Architecture       | Retrieval-Augmented Generation |
| Vector Database        | FAISS                          |
| Embeddings             | Text Embeddings                |
| Document Processing    | PDF, DOCX, TXT                 |
| Database               | SQLite                         |
| Environment Management | Python Virtual Environment     |
| Configuration          | Environment Variables          |

---

## 🏗️ System Architecture

IntelliAssist follows a modular architecture:

1. **User Input:** Accepts questions, text, and document uploads.
2. **Document Processing:** Extracts text and metadata from uploaded documents.
3. **Text Chunking:** Divides extracted content into smaller chunks.
4. **Embedding Generation:** Converts text chunks into numerical vector representations.
5. **Vector Search:** Uses FAISS to retrieve relevant document chunks.
6. **AI Processing:** Sends relevant context to Google Gemini.
7. **Response Generation:** Generates context-aware answers with citations.
8. **Data Storage:** Stores conversation history using SQLite.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd IntelliAssist
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file in the project root directory.

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Alternatively, the application supports `GOOGLE_API_KEY` for API key configuration.

### 7. Run the Application

Run the project's main entry-point file using Python.

```bash
python main.py
```

*Note: Replace `main.py` with the actual entry-point filename if your project uses a different one.*

---

## 🔑 Environment Configuration

| Variable       | Description                         | Default          |
| -------------- | ----------------------------------- | ---------------- |
| GEMINI_API_KEY | Google Gemini API key               | Required         |
| GOOGLE_API_KEY | Alternative Google API key variable | Optional         |
| GEMINI_MODEL   | Gemini model selection              | gemini-2.5-flash |
| Timeout        | API request timeout                 | 30,000 ms        |

---

## 📂 Project Capabilities

| Module                | Functionality                                |
| --------------------- | -------------------------------------------- |
| Document Intelligence | Extracts and processes document content      |
| RAG Engine            | Retrieves relevant information using FAISS   |
| AI Chat Engine        | Generates intelligent responses using Gemini |
| Summarization Engine  | Produces concise and detailed summaries      |
| Sentiment Analysis    | Analyzes the emotional tone of text          |
| Intent Detection      | Identifies user query intentions             |
| Conversation Manager  | Maintains multi-turn conversations           |
| Database Manager      | Stores conversation history in SQLite        |

---

## 🧪 Testing

The backend has undergone regression testing.

* **Total Tests Passed:** 252
* **Failed Tests:** 0
* **Result:** All regression tests passed.

Live Gemini summarization was also successfully tested.

---

## 🔮 Future Enhancements

* Advanced conversational memory.
* Support for additional document formats.
* Improved semantic search and retrieval.
* Enhanced multilingual capabilities.
* User authentication and personalized chat history.
* Improved AI response evaluation.

---

## 🔐 Security

* Store API keys in environment variables.
* Never commit `.env` files or API credentials.
* Use `.gitignore` to exclude virtual environments and sensitive configuration files.

---

## 👩‍💻 Developer

**Purvi Bheda**

AI | Python | Software Development

---

## ⭐ Acknowledgements

* Google Gemini for AI-powered language processing.
* FAISS for efficient vector similarity search.
* Python ecosystem for document processing and backend development.

---

**IntelliAssist – Making Information Intelligent.**
