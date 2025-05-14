import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

#Step1:

text=""
pdf_reader=PdfReader("incorrect_facts.pdf")
for page in pdf_reader.pages:
    text += page.extract_text() + "\n"

#print(text)

#2 Chunk
text_splitter=CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=20,length_function=len)
chunks=text_splitter.split_text(text)



#3 Initialize Pinecone and create Index

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "rag-project"

'''pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
'''



index=pc.Index(index_name)


#Initialize embeddings and store chunks in pinecone
embeddings=SentenceTransformer('all-MiniLM-L6-v2')
for i, chunk in enumerate(chunks):
    chunk_embedding = embeddings.encode(chunk)
    index.upsert([(str(i), chunk_embedding, {"text": chunk})])

# Step 5 - Create llm object for chian

llm=ChatGroq(model='llama-3.1-8b-instant', temperature=0)

#Step 6 - User Asks Question

while True:
    query=input("Ask a question for type 'exit' to quit):")
    if query.lower() == "exit":
        break

    # Step 7 - Retrieve top 3 relevant chunks from Pinecone
    query_embedding = embeddings.encode(query)  # Make sure this is a list, not ndarray
    query_embedding_list = query_embedding.tolist()  # Convert to list if necessary

    # Query Pinecone for top 3 relevant results
    result = index.query(vector=query_embedding_list, top_k=1, include_metadata=True)

    # Step 8 - Combine top results into a single context string
    augmented_context = "\n\n".join([match.metadata["text"] for match in result.matches])


    #Step 9: Create Prompt template

    prompt=PromptTemplate(
        input_variables=["context","question"],
        template="You are a helpful assistant. Use the context provided to answer the question"
        "Context:{context}"
        "Question:{question}"
    )

    #Step 10-Set up Langchain llm with groq and prompt template
    chain=prompt | llm

    #step 11: generate response using augmented context and user question
    response =chain.invoke({"context":augmented_context, "question": query})
    print(f"Retrieved Text: {augmented_context}")
    print("Answer:", response.content)