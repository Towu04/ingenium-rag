from evaluation.metrics import citation_precision

# testing RAG 

# necessary libraries
import ragas
import pandas as pd 


# run file that has RAG code 
# assume file is called rag.py and is in same dir 
# assume rag is called run_rag
from rag.py import run_rag 


# step 1: load evaluation data 

# not yet known what format. csv? json? 

# if csv: load via pandas   
eval_dat = pd.read_csv("path_to_eval_file.csv")  

# if json: load via pandas
eval_dat = pd.read_json("path_to_eval_file.json", orient="records")


# step 2: run the RAG 

# prepare object to store RAG output 
test_output = []

# for each question run the RAG and store the output 
for index, row in eval_dat.iterrows():

    # isolate the question
    # this requires the column name to be "question". if it's something else then replace it
    question = row["question"]

    # run the RAG 
    # assume the model is called run_rag. if not replace it with actual function name  
    rag_output = run_rag(question)  
    # save the question, output, and what data was used to generate the output to the test_output object    
    test_output.append({"question_asked": question, 
                        "output_received": rag_output})    


# step 3: save this output to a file. let's do csv for now, ask if team prefers json 
# let's call the output file "rag_test_output.csv" for now.
output_df = pd.DataFrame(test_output)
output_df.to_csv("rag_test_output.csv", index = False)


# step 4: evaluate this output. proposal for now: use ragas 