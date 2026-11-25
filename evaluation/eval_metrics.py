
# for extraction 

# we have a given row from the eval_dat dataframe 

# metric 1: Recall@K 
# how many of the relevant chunks are in the top K chunks returned by the model?
# mathematically: no. chunks recalled that overlap with relevant chunks set / total no. of relevant chunks

def recall_at_k(row, retrieved_chunks, relevant_chunks, K): 

    # get IDs of top K retrieved chunks 
    retrieved_chunks = row["rag_retrieved_chunks"][:K]

    # get IDs of relevant chunks 
    relevant_chunks = row["relevant_chunks"]

    # calculate the number of relevant chunks that are in the top K retrieved chunks
    num_relevant_retrieved = len(set(retrieved_chunks).intersection(set(relevant_chunks)))
    # calculate recall at K
    recall_at_k = num_relevant_retrieved / len(set(relevant_chunks)) if set(relevant_chunks) else 0

    return recall_at_k

# test simple numerical example 
# make fake row 
fake_row = {
    "rag_retrieved_chunks": [1, 2, 3, 4, 5],
    "relevant_chunks": [2, 4, 6]
}
# run the function
recall_at_k(fake_row, fake_row["rag_retrieved_chunks"], fake_row["relevant_chunks"], 3)  
# should return 1/3 since only chunk 2 is relevant and in the top 3 retrieved chunks
print("Recall@3 for the fake row:", recall_at_k(fake_row, fake_row["rag_retrieved_chunks"], fake_row["relevant_chunks"], 3))

# metric 2: Precision@K
# how many of the top K chunks returned by the model are relevant?
# mathematically: no. chunks recalled that overlap with relevant chunks set / total no. of chunks recalled by the model

def precision_at_k(row, retrieved_chunks, relevant_chunks, K):

    # get IDs of top K retrieved chunks 
    retrieved_chunks = row["rag_retrieved_chunks"][:K]

    # get IDs of relevant chunks 
    relevant_chunks = row["relevant_chunks"]

    # calculate the number of relevant chunks that are in the top K retrieved chunks
    num_relevant_retrieved = len(set(retrieved_chunks).intersection(set(relevant_chunks)))
    # calculate precision at K
    precision_at_k = num_relevant_retrieved / len(set(retrieved_chunks)) if set(retrieved_chunks) else 0

    return precision_at_k

# test simple numerical example
# make fake row
fake_row = {
    "rag_retrieved_chunks": [1, 2, 3, 4, 5],
    "relevant_chunks": [2, 4, 6]
}
# run the function
precision_at_k(fake_row, fake_row["rag_retrieved_chunks"], fake_row["relevant_chunks"], 3)  
# should return 1/3 since only chunk 2 is relevant and in the top 3 retrieved chunks
print("Precision@3 for the fake row:", precision_at_k(fake_row, fake_row["rag_retrieved_chunks"], fake_row["relevant_chunks"], 3))  


# metric 3: MRR (Mean Reciprocal Rank)
# for each question, find the rank of the first relevant chunk in the list of chunks returned
# mathematically: 1 / rank of first relevant chunk  