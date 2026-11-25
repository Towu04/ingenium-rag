
import ragas 
import langchain

from ragas.testset import TestsetGenerator

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.testset import simple, reasoning, multi_context




# based on the chunks 
# make new dataset called synth_ground_truth that has generated questions and relevant chunks 

# envisioned data format: 
# question number / question / model answer / model chunks 
# rag answer / rag chunks 


# step 1: generate ground truth data 

# load context chunks 

# what are we using as an evaluator LLM? and as an embedder? 
generator = TestsetGenerator.from_langchain(
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k"), 
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small"),
)

# generate the testset 
# of size 10 
testset = generator.generate_with_langchain_docs(
    context_chunks, 
    testset_size = 10, 
    query_distribution = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}
)


# step 2: get grounf truth dataset into usable format 

ground_truth = testset.to_pandas()
print(ground_truth[["user_input", "reference"]].head())


