
import ragas 
from ragas import evaluate
from ragas.metrics import (
   faithfulness,
   answer_relevancy,
   answer_correctness
)

from generate_truth import ground_truth

from eval_metrics import recall_at_k, precision_at_k

# phase 1: evaluating the ragas synthetic dataset 

# step 1: run the rag 

# setup dataset to store RAG output 
rag_output = ground_truth

# for each question run the RAG and store the output 
for index, row in ground_truth.iterrows():

    # isolate the question
    # this requires the column name to be "question". if it's something else then replace it
    question = row["Question"]

    # run the RAG 
    # assume the model is called run_rag. if not replace it with actual function name  
    output_from_rag = run_rag(question)  
    # save the question, output, and what data was used to generate the output to the test_output object    
    rag_output.at[index, "rag_output"] = output_from_rag
    rag_output.at[index, "rag_relevant_chunks"] = output_from_rag["relevant_chunks"]  
    # assuming the output has a key called "relevant_chunks"


# step 2: evaluate context extraction 

# setup dataset to store metrics results 
synth_eval = rag_output

# line by line: extraction metrics 
for index, row in synth_eval.iterrows():
    # get the retrieved chunks from the RAG output
    retrieved_chunks = row["rag_retrieved_chunks"]
    # get the relevant chunks from the LLM output
    relevant_chunks = row["relevant_chunks"]

    # calculate recall@K and precision@K for K = 1,3,5
    for K in [1, 3, 5]:
        recall = recall_at_k(row, retrieved_chunks, relevant_chunks, K)
        precision = precision_at_k(row, retrieved_chunks, relevant_chunks, K)
        synth_eval.at[index, f"recall_at_{K}"] = recall
        synth_eval.at[index, f"precision_at_{K}"] = precision


# all at once: using ragas and an llm 

# might have to change format of evaluation dataset. we want questions, rag output, rag chunks 

result = evaluate(rag_output, 
                  metrics=[faithfulness, answer_relevancy, answer_correctness])




