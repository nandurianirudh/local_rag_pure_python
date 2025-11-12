# 🧠 Local RAG (Retrieval-Augmented Generation) in Pure Python

A fully local implementation of a **Retrieval-Augmented Generation (RAG)** pipeline using **Weaviate**, **Streamlit**, and **Open Source Embedding Models** — all running locally without cloud dependencies.

---

## 🚀 Features

- Local **Weaviate** instance (Dockerized)
- Pure Python pipeline for:
  - Document loading and chunking
  - Embedding creation
  - Vector storage and retrieval
- Streamlit UI for querying your local knowledge base
- Modular design — easy to extend with other models or data sources

---

## 🧩 Repository Structure

```
local_rag_pure_python/
│
├── app.py                       # Streamlit frontend
├── Code_Dev_Notebook.ipynb      # Development notebook for embeddings & testing
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker setup for Weaviate
├── .collections/                # (Auto-created) Persistent Weaviate data
└── README.md                    # You're here
```

---

## ⚙️ Setup Instructions

Follow these steps to run the app locally.

### 1. Clone the Repository

```bash
git clone https://github.com/nandurianirudh/local_rag_pure_python.git
cd local_rag_pure_python
```

---

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

---

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

### 4. Build and Run Weaviate Locally via Docker

Start the Weaviate vector database:

```bash
docker compose up -d
```

Check that Weaviate is running properly:

```bash
docker ps
```

You should see a container named **weaviate** running on port `8080`.

---

### 5. Create Embeddings and Collections

Run the notebook `Code_Dev_Notebook.ipynb` to:
- Load your text or document dataset
- Generate embeddings (using models such as `bge-base-en-v1.5` or `all-MiniLM-L6-v2`)
- Push the embeddings into the Weaviate database

> 💡 You can modify the notebook to use your own documents or data sources.

---

### 6. Launch the Streamlit App

Once embeddings are ready, start the frontend:

```bash
streamlit run app.py
```

The app will launch in your browser at:
```
http://localhost:8501
```

---

## 🧠 How It Works

1. **Document Ingestion**  
   Raw text or documents are chunked and embedded using a local sentence transformer model.

2. **Vector Storage (Weaviate)**  
   Embeddings and metadata are stored locally in Weaviate.

3. **Retrieval + Augmentation**  
   When a user enters a query in the Streamlit app, similar chunks are retrieved from Weaviate and combined with the query.

4. **Response Generation**  
   The model uses retrieved context to generate a final answer — grounded in your local data.

---

## 🧰 Useful Commands

To stop the Docker container:

```bash
docker compose down
```

To clear all stored data (optional):

```bash
docker compose down -v
```

---

## 🧪 Troubleshooting

- **Weaviate not starting?**  
  Ensure Docker is running and port `8080` is free.

- **Embedding errors?**  
  Check that the correct model is downloaded and accessible in your notebook.

- **Streamlit not opening?**  
  Try `streamlit run app.py --server.port 8501` to specify a port manually.

---

## 🧑‍💻 Contributing

Pull requests are welcome!  
If you’d like to extend the functionality (e.g., add FAISS support or custom LLMs), feel free to fork the repo and submit changes.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Acknowledgements

- [Weaviate](https://weaviate.io/) for local vector search
- [Streamlit](https://streamlit.io/) for fast app prototyping
- [Sentence Transformers](https://www.sbert.net/) for embeddings

---

**Happy Building! 🚀**
