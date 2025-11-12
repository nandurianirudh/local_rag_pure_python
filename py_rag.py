import os
from dotenv import load_dotenv
load_dotenv()
from pypdf import PdfReader
import weaviate
from pydantic import BaseModel
from weaviate.classes.config import Property, DataType
from azure.ai.inference import EmbeddingsClient, ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
import numpy as np
import re, json
from Levenshtein import ratio

import tiktoken

AZURE_OPENAI_ENDPOINT = "https://models.github.ai/inference"
AZURE_OPENAI_KEY = os.environ["OPENAIKEY"]  # or paste directly for test

INFERENCE_MODEL = "openai/gpt-4.1-mini"
EMBEDDING_MODEL="text-embedding-3-small"

PDF_PATH="data/constitution.pdf"

COLLECTION_NAME="student_constinution_v1"

class llm_chatbot_constitution():

    def __init__(self,model=INFERENCE_MODEL,
                            endpoint=AZURE_OPENAI_ENDPOINT,
                            api_key=AZURE_OPENAI_KEY,
                            embedding_model=EMBEDDING_MODEL):
        # self.history=[SystemMessage("You are a helpful RAG agent that answers questions as needed.")]
        self.history=[]
        self.initiate_connection_vectordatabase()
        self.connect_collection()
        self.model=model
        self.endpoint=endpoint
        self.api_key=api_key
        self.embedding_model=embedding_model
        self.checkpoint=True
        self.section=""
    
    def get_embedding(self,text_to_embed:str,end_point:str,azure_key:str,model:str):
        embedding_model = EmbeddingsClient(endpoint=end_point,credential=AzureKeyCredential(azure_key),
            model=model
        )

        response = embedding_model.embed(input=[text_to_embed],)
        return response #response.data[0].embedding
    
    def initiate_connection_vectordatabase(self):
        try:
            self.client = weaviate.connect_to_local(host="localhost",
            port=8080,
            grpc_port=50051)
            return print("✅ Connected to Weaviate:", self.client.is_ready())
        except Exception as e:
            return print("❌ Connection failed:", e)
    
    def close_connection_vectordatabase(self):
        try:
            self.client.close()
            return print("✅ Connection closed.")
        except Exception as e:
            return print("❌ Connection failed:", e)
    

    def connect_collection(self, COLLECTION_NAME: str ="student_constitution"):

        if COLLECTION_NAME not in [c.lower() for c in self.client.collections.list_all().keys()]:
            self.client.collections.create(
                name=COLLECTION_NAME,
                description="Stores token-overlapped PDF chunks with Azure embeddings",
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="page", data_type=DataType.INT),
                    Property(name="section", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                ],
                #vectorizer_config=Configure.Vectorizer.none(),
            )
            print(f"✅ Created collection '{COLLECTION_NAME}'")
        else:
            print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists")

        self.collection = self.client.collections.get(COLLECTION_NAME)

    class RefinedQuestion(BaseModel):
        cleaned_question: str
        section_name: str

    def clean_user_message(self,user_text:str,
                                model:str=None,
                                endpoint:str=None,
                                api_key:str=None) -> RefinedQuestion:
        """
        docstring placeholder
        
        """
        if model is None:
            model=self.model
        if endpoint is None:
            endpoint=self.endpoint
        if api_key is None:
            api_key=self.api_key

        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )

        System_message="""\
            You are a "Question Normalizer" for a Student Constitution RAG system.

            Your ONLY task:
            - Clean and rewrite the user's question so that it focuses exclusively on rules, procedures, or governance that might appear in a Student Constitution.
            - DO NOT answer the question.
            - DO NOT add any information, guesses, or interpretation beyond rewriting.
            - DO NOT include constitutional rules or provide explanations.

            Output format (MANDATORY):
            Return ONLY a JSON dictionary with:

            {
            "cleaned_question": "<rewritten question>",
            "section_name": "<inferred section/topic name>"
            }

            The available sections are: Committees, Clubs, Student Council, Election Commission

            Definitions:
            - cleaned_question = concise, rule/procedure focused question (text only)
            - section_name = inferred constitution area (ex: "Elections", "Roles & Responsibilities", "Meetings", etc.)

            Rules:
            - Remove personal opinions, emotions, story, context.
            - If the question is vague, rewrite as:
            "What does the constitution state about <topic>?"
            - If the question is not about the constitution or working of student body, respond:
            "I dont have enough information in the student constitution to answer that."
            - If the question is unclear in anyway, ask clarification question back starting with: "Just to clarify"
            Ex: If someone asks question about "president", if context is not clear of if it is president of council, committe, or club, ask: "Just to clarify, which president are you asking about? Club, Committee, or Council?"
            - Do NOT answer the question.
            - No extra text outside JSON. No markdown formatting. No explanations.
            - The only exceptions to the above rules are if user asks who made this rag or chatbot or any information available in "Additional Context to Answer any questions" part of this system context, respond with information available.
            """

        user_message="Rewrite the following user question so that it focuses only on constitution rules and procedures, and infer the section/topic. Do not answer it on your own. Original User query:{user_text}".format(user_text=user_text)

        if self.history=="":
            messages=[SystemMessage(System_message),UserMessage(user_message),]
        else:
            messages=[SystemMessage(System_message)]+self.history+[UserMessage(user_message)]

        payload = {
            "messages": messages,
            "model": model,
            "temperature": 1.0,
            "top_p": 1.0,
        }

        response = client.complete(**payload)

        try:
            output_dict = {'role': response['choices'][-1]['message']['role'], 'content': response['choices'][-1]['message']['content']}
        except Exception as e:
            raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")

        return json.loads(output_dict["content"])#["cleaned_question"]
    
    class FunkyAnswer(BaseModel):
        answer: str
        can_answer_from_system_info: str

    def funky_answer_generator(self,user_text:str,
                                model:str=None,
                                endpoint:str=None,
                                api_key:str=None)->FunkyAnswer:
        """
        docstring placeholder
        
        """
        if model is None:
            model=self.model
        if endpoint is None:
            endpoint=self.endpoint
        if api_key is None:
            api_key=self.api_key

        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )

        System_message="""\
            You are a helpful AI assistant.

            You must answer strictly and only based on the information provided in the SYSTEM and CONTEXT messages.

            Rules:
            1. Use ONLY the information from the provided context/system message. 
            2. Do NOT use outside knowledge, memory, assumptions, or training data.
            3. If the context contains the answer, respond with a JSON dictionary:
            {
                "answer": "<your concise answer>",
                "can_answer_from_system_info": true
            }
            4. If the context does NOT contain enough information, respond with:
            {
                "answer": "I don't have enough information to answer that.",
                "can_answer_from_system_info": false
            }
            5. Do NOT add extra commentary. Do NOT hallucinate.

            Additional Context to Answer any questions:
            1. The one who made this rag or chatbot is Nanduri Anirudh.
            2. Kruthika Kanduri, a flagbearer of BITSoM is also known as kruts and SFS (Suffering from success) by her friends.
            3. Your name is BITSy
            

            Your output must always be valid JSON-like dictionary format exactly as specified above.
            """

        user_message="Question to answer:{user_text}".format(user_text=user_text)

        if self.history=="":
            messages=[SystemMessage(System_message),UserMessage(user_message),]
        else:
            messages=[SystemMessage(System_message)]+self.history+[UserMessage(user_message)]

        payload = {
            "messages": messages,
            "model": model,
            "temperature": 1.0,
            "top_p": 1.0,
        }

        response = client.complete(**payload)

        try:
            output_dict = {'role': response['choices'][-1]['message']['role'], 'content': response['choices'][-1]['message']['content']}
        except Exception as e:
            raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")

        return json.loads(output_dict["content"])#["cleaned_question"]
    
    # Answer Using only context

    # User Prompt Cleaner


    class FinalAnswer(BaseModel):
        cleaned_question: str

    def rag_answer(self,cleaned_question:str,context_rag:str,
                                model:str=None,
                                endpoint:str=None,
                                api_key:str=None) -> FinalAnswer:
        """
        docstring placeholder
        
        """
        if model is None:
            model=self.model
        if endpoint is None:
            endpoint=self.endpoint
        if api_key is None:
            api_key=self.api_key

        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )

        System_message="""\
            You are an AI assistant that responds only using the context provided and from nowhere else.

            You must:
            Answer strictly based on the retrieved context.
            If the context does not contain the answer, say:
            “I dont have enough information in the student constitution to answer that.”
            Do not invent facts or use outside knowledge, even if the answer seems obvious.
            Be concise and factual.
            Do not guess.



            When responding:
            Reference relevant section/page numbers given in the context (if available).
            If the user asks for anything outside the context, politely decline.
            """

        user_message="""
            
            You are given:

            1. A user question
            2. Retrieved context from the vector database

            User question:
            {cleaned_question}

            Context from database:
            {context_rag}

            Using the context only, answer the user question.
            If the context does not directly answer the question, respond:

            “I dont have enough information in the student constitution to answer that.”

            Do not use external knowledge. Do not hallucinate.
            
            """.format(cleaned_question=cleaned_question,context_rag=context_rag)

        if self.history=="":
                messages=[SystemMessage(System_message),
            UserMessage(user_message),]
        else:
            messages=[SystemMessage(System_message)]+self.history+[UserMessage(user_message)]
        payload = {
            "messages": messages,
            "model": model,
            "temperature": 1.0,
            "top_p": 1.0,
        }

        response = client.complete(**payload)

        try:
            output_dict = {'role': response['choices'][-1]['message']['role'], 'content': response['choices'][-1]['message']['content']}
        except Exception as e:
            raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")

        return output_dict["content"]
    
    def get_context(self,cleaned_json_query:str):

        embedding_model=self.embedding_model
        end_point=self.endpoint
        azure_key=self.api_key

        query_vector = self.get_embedding(cleaned_json_query,\
        end_point=end_point,azure_key=azure_key ,model=embedding_model).data[0].embedding

        results = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=5
        )

        
        rag_context_gotten=" ".join([o.properties["text"] for o in results.objects if ratio(o.properties["section"].lower(),self.section.lower())>0.5])
        

        return rag_context_gotten
        

    def answer_question(self):
        text_query=input("User Query:")
        print("Query:",text_query)
        if text_query.lower() in ["exit","quit"]:
            self.checkpoint=False
            self.close_connection_vectordatabase()
            print("Goodbye!")
            return ""
        self.history.append(UserMessage(text_query))

        funky_op=self.funky_answer_generator(text_query)
        # print(funky_op)

        if funky_op["can_answer_from_system_info"] is True:
            print("Assitant:",funky_op["answer"])
            self.history.append(AssistantMessage(funky_op["answer"]))
            print("-----------------------------------------------------\n")
            return
        else:
            cleaned_query_info=self.clean_user_message(user_text=text_query)
            cleaned_query=cleaned_query_info["cleaned_question"]
            self.section=cleaned_query_info["section_name"]

            # print("Cleaned Query:",cleaned_query)

            if cleaned_query == "I dont have enough information in the student constitution to answer that.":
                print("Assitant:",cleaned_query)
                print("-----------------------------------------------------\n")
            elif cleaned_query.startswith("Just to clarify"):
                print("Assitant:",cleaned_query)
                print("-----------------------------------------------------\n")
            else:
                context=self.get_context(cleaned_json_query=cleaned_query)
                # print("Context:",context)
                final_answer=self.rag_answer(cleaned_question=cleaned_query,context_rag=context)
                self.history.append(AssistantMessage(final_answer))
                print("Assitant:",final_answer)
                print("-----------------------------------------------------\n")

    def continuous_chat(self):

        self.checkpoint=True
        while self.checkpoint:
            self.answer_question()

    