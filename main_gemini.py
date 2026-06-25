import time
from modules.retrieval import vector_db
from modules.generation import llm_client
from modules.evaluation import faithfulness

def main():
    # 1. Initialize System (Ingestion & Vectorization are handled by the module)
    print("Initializing AI Tutor System...")
    segments, embeddings = vector_db.initialize_vector_db()
    
    if not segments or len(embeddings) == 0:
        print("System initialization failed. Please check your source.pdf. Exiting.")
        return

    print("\n  AI Tutor Online. Ready for queries. (Type 'q' to exit)\n")

    # 2. The Main Interaction Loop
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break
        
        # --- THE CLOSED RAG LOOP (NOVELTY 1) ---
        max_retries = 3
        current_top_k = 3 # Start by retrieving 3 chunks
        
        try:
            for attempt in range(max_retries):
                print(f"  [Attempt {attempt+1}] Retrieving {current_top_k} chunks...")
                
                # A. Retrieve Context
                context = vector_db.retrieve_context(user_input, segments, embeddings, top_k=current_top_k)
                
                # B. Generate Answer using Gemini
                draft_answer = llm_client.generate_draft_answer(user_input, context)
                
                # C. Evaluate Faithfulness
                faith_score = faithfulness.check_semantic_similarity(context, draft_answer)
                print(f"  [Attempt {attempt+1}] Faithfulness Score: {faith_score:.4f}")
                
                # D. The Quality Gate Logic
                if faith_score >= 0.75:
                    # The answer is faithful to the textbook. Show it to the user.
                    print(f"\nAI Tutor: {draft_answer}\n")
                    break 
                else:
                    # The answer failed the check. Trigger the dynamic resize.
                    print("  [Warning] Hallucination risk detected. Expanding context...")
                    current_top_k += 2 # Expand chunk retrieval size for the next loop
                    
            else:
                # If the loop exhausts all 3 attempts and still fails the faithfulness check
                print("\nAI Tutor: I'm sorry, I couldn't formulate a highly accurate answer based on the current textbook material. Could you rephrase your question?\n")
                
        except Exception as e:
            print(f"  Error processing query: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()