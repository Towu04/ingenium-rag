
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

import numpy as np



# step 1: load evaluation data 

# excel file provided with model questions 
eval_dat = pd.read_excel(
    "Evaluation/Emergent_Test_Questions.xlsx"
    ) 


# step 2: prepare the eval data 

# read the .env file to get the project names
load_dotenv("evaluation/eval.env")

# retreive client and projet names 
client_name = os.getenv("Client")   
project_names = os.getenv("Projects").split(",")

# add some new columns: 

# column 5 for the new modified question 
eval_dat["Mod_Question"] = None

# column 6 for client or project or neither 
eval_dat["Type"] = None 

# column 7 should be the weights for each question so that the sum of the weights is 1.0. if not, then we will set them to 1/n where n is the number of questions.
eval_dat["Weights"] = 1


# number of questions 
n = len(eval_dat)



# step 2.1: fill in model answer with potential words to replace X 

# for each question ID
for index in reversed(range(len(eval_dat))):
    row = eval_dat.iloc[index]
    question = row["Question"] # isolate question

    # make mod question the same 
    eval_dat.at[index, "Mod_Question"] = question

    # if it is a client question
    if "X" in question and "client" in question.lower():

        # question type is "client" so put that in the type column
        eval_dat.at[index, "Type"] = "client"

        # replace "X" with actual client name
        new_question = question.replace(
            "X", 
            client_name
            )
        # put modified question in the new column
        eval_dat.at[index, "Mod_Question"] = new_question

    # if the question is about some project X 
    elif "X" in question and "project" in question.lower():

        # replace with actual project names 
        # not all of them since there are so many: use random subset 
        random_project_names = random.sample(project_names, 5)

        for project_name in random_project_names:
            new_question = question.replace("X", project_name)
            # make new row 
            new_row = row.copy()
            # repace question with new question 
            new_row["Mod_Question"] = new_question
            # question type is "project" so put that in the type column
            new_row["Type"] = "project"
            # put weight of 1 / n 
            new_row["Weights"] = 1 / 5

            # append new row to eval_dat right under the old row
            eval_dat = pd.concat(
                [eval_dat.iloc[:index+1], 
                pd.DataFrame([new_row]), 
                eval_dat.iloc[index+1:]]).reset_index(drop = True)

        # delete the original row with the X in it 
        eval_dat.drop(index, inplace = True)    

    # or if question ID is 58 or 60 then add project names after the word "project":
    elif "X" in question and (row["ID"] == 58 or row["ID"] == 60): 

        # question type is "project" so put that in the type column
        eval_dat.at[index, "Type"] = "project"

        # replace with actual project names
        # not all of them since there are so many: use random subset
        random_project_names = random.sample(project_names, 5)

        for project_name in random_project_names:
            new_question = question.replace("X", project_name)
            # make new row 
            new_row = row.copy()
            # repace question with new question 
            new_row["Mod_Question"] = new_question
            # put weight of 1 / n 
            new_row["Weights"] = 1 / 5
            # append new row to eval_dat right under the old row 
            eval_dat = pd.concat(
                [eval_dat.iloc[:index+1], 
                 pd.DataFrame([new_row]), 
                 eval_dat.iloc[index+1:]]).reset_index(drop = True)

        # delete the original row with the X in it
        eval_dat.drop(index, inplace = True)
 


# step 2.2
# fill in potential contexts for other unspecific questions: 
# q 16, 24, 37, 69 

# import from env file 

q16 = os.getenv("Mod_Q16")   
q24 = os.getenv("Mod_Q24")
q37 = os.getenv("Mod_Q37")

# for the question whose ID is 16
index_16 = eval_dat[eval_dat['ID'] == 16].index
# replace modified question by env question 
eval_dat.loc[index_16, "Mod_Question"] = q16

# 24 
index_24 = eval_dat[eval_dat['ID'] == 24].index
eval_dat.loc[index_24, "Mod_Question"] = q24

# 37 
index_37 = eval_dat[eval_dat['ID'] == 37].index
eval_dat.loc[index_37, "Mod_Question"] = 37



# save file to excel
eval_dat.to_excel(
    "testfile.xlsx", 
    index = False
    )

print("test questions edited successfully")