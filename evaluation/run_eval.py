
# testing RAG 

# set a seed for reproducibility
import random
random.seed(42)

# necessary libraries
# import ragas
import pandas as pd 
import openpyxl
from dotenv import load_dotenv
import os
from pathlib import Path
from IPython.display import display


# run file that has RAG code 
# assume file is called rag.py and is in same dir 
# assume rag is called run_rag
## from rag.py import run_rag 


# step 1: load evaluation data 

# not yet known what format. csv? json? 

# if csv: load excel file via pandas   
eval_dat = pd.read_excel(
    "Evaluation/Emergent_Test_Questions.xlsx"
    ) 


# step 2: prepare the eval data 

# read the .env file to get the project names
load_dotenv("evaluation/eval.env")

# retreive client and projet names 
client_name = os.getenv("Client")   
project_names = os.getenv("Projects").split(",")

# keep columns in the test data file 
# column 5 should be a model answer given by an LLM 
# column 6 should be the relevant chunks selected by an LLM 

# client or project or neither (col 4)
eval_dat["type"] = None 

eval_dat["model_answer"] = None 
eval_dat["relevant_chunks"] = None
eval_dat["weights"] = 1




# step 2.1: fill in model answer with potential words to replace X 

for project_name in project_names:
     print(f"Project Name: {project_name}")

n = len(project_names)
print(f"Number of Project Names: {n}")

# print each project 
for project_name in project_names:
    print(f"Project Name: {project_name}")

# for each question with an X 
# make new rows where we replace X with 
# either the client name if "client" is mentioned 
# or each of the potential project names if "project" is mentioned
# delete the original row with the X in it 
for index, row in eval_dat.iterrows():
    question = row["Question"]

    if "X" in question and "client" in question.lower():

        # question type is "client" so put that in the type column
        eval_dat.at[index, "type"] = "client"

        # replace "X" with actual client name
        new_question = question.replace(
            "X", 
            client_name
            )
        # replace question with new question
        eval_dat.at[index, "Question"] = new_question

    elif "X" in question and "project" in question.lower():

        # question type is "project" so put that in the type column
        eval_dat.at[index, "type"] = "project"

        # replace with actual project names 
        # not all of them since there are so many: use random subset 
        random_project_names = random.sample(
            project_names, 
            5
        )

        for project_name in random_project_names:
            new_question = question.replace(
                "X", 
                project_name
                )
            # make new row 
            new_row = row.copy()
            # repace question with new question 
            new_row["Question"] = new_question
            # put weight of 1 / n 
            new_row["weights"] = 1 / n
            # append new row to eval_dat right under previous row 
            eval_dat = pd.concat(
                [eval_dat.iloc[:index+1], 
                 pd.DataFrame([new_row]), 
                 eval_dat.iloc[index+1:]]).reset_index(drop = True)
            
        # delete the original row with the X in it 
        eval_dat.drop(index, inplace = True)

    # or if question ID is 58 or 60 then add project names after the word "project": 
    elif "X" in question and (row["ID"] == 58 or row["ID"] == 60):

        # question type is "project" so put that in the type column
        eval_dat.at[index, "type"] = "project"
        
        # replace with actual project names 
        # not all of them since there are so many: use random subset 
        random_project_names = random.sample(
            project_names, 
            5
        )

        for project_name in random_project_names:
            new_question = question.replace(
                "project", 
                f"project {project_name}"
                )
            # make new row 
            new_row = row.copy()
            # repace question with new question 
            new_row["Question"] = new_question
            # put weight of 1 / n 
            new_row["weights"] = 1 / n
            # append new row to eval_dat right under the old row 
            eval_dat = pd.concat(
                [eval_dat.iloc[:index+1], 
                 pd.DataFrame([new_row]), 
                 eval_dat.iloc[index+1:]]).reset_index(drop = True)
        # delete the original row with the X in it 
        eval_dat.drop(index, inplace = True)

print("X in test questions edited successfully")


# set display option to show all columns
pd.set_option('display.max_columns', None)
display(eval_dat.head(10))


# step 2.2: generate model answers with an LLM using the documentation and the question

# step 2.3: generate reference of relevant chunks with an LLM using the documentation and the question


# step 3: run the RAG 

# prepare object to store RAG output 
eval_dat["rag_output"] = None
eval_dat["rag_relevant_chunks"] = None

# for each question run the RAG and store the output 
for index, row in eval_dat.iterrows():

    # isolate the question
    # this requires the column name to be "question". if it's something else then replace it
    question = row["Question"]

    # run the RAG 
    # assume the model is called run_rag. if not replace it with actual function name  
    rag_output = run_rag(question)  
    # save the question, output, and what data was used to generate the output to the test_output object    
    eval_dat.at[index, "rag_output"] = rag_output
    eval_dat.at[index, "rag_relevant_chunks"] = rag_output["relevant_chunks"]  # assuming the output has a key called "relevant_chunks"



# step 4: evaluate this output. proposal for now: use ragas 





