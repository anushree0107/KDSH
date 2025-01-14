import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def summarization_agent(large_text):
    prompt = ChatPromptTemplate(
        [
            ("system", """You are a Summarization Agent. Your task is to condense large pieces of text into 3-4 lines while preserving all critical information. Focus on extracting the main ideas, key facts, and overall context. Ensure the summary is clear, concise, and informative.
             Just give the summary, there should not be this kind of statements : Here is a summary of the paper in 3-4 lines
             """),
            ("human", "{input_text}")
        ]
    )

    input_query = prompt.format(input_text=large_text)

    llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    answer = llm.invoke(input_query)

    return answer.content

if __name__ == "__main__":
    large_text = """
    Deep learning is a subset of machine learning, which itself is a subset of artificial intelligence. It is based on artificial neural networks, which are inspired by the structure and function of the human brain. Deep learning models are designed to automatically learn and improve from large amounts of data by identifying patterns and structures without explicit programming.

    ### Key Characteristics of Deep Learning:
    1. **Hierarchical Feature Learning:** Deep learning models learn features in a hierarchical manner, where lower layers capture simple patterns and higher layers capture complex patterns.
    2. **End-to-End Learning:** These models can process raw data and make predictions directly, reducing the need for manual feature extraction.
    3. **Scalability:** Deep learning excels with large datasets, and its performance often improves as the data size increases.

    ### Common Architectures in Deep Learning:
    - **Convolutional Neural Networks (CNNs):** Primarily used for image-related tasks like classification, object detection, and segmentation.
    - **Recurrent Neural Networks (RNNs):** Suitable for sequential data like time series or natural language processing.
    - **Transformers:** Widely used in natural language processing and tasks like text generation and translation (e.g., GPT models).
    - **Generative Adversarial Networks (GANs):** Used for generating new data samples that resemble the training data.

    ### Applications of Deep Learning:
    1. **Computer Vision:** Object detection, image classification, and facial recognition.
    2. **Natural Language Processing:** Sentiment analysis, machine translation, and chatbots.
    3. **Healthcare:** Medical image analysis, drug discovery, and personalized treatment plans.
    4. **Autonomous Systems:** Self-driving cars and drones.
    5. **Finance:** Fraud detection, stock price prediction, and algorithmic trading.

    Deep learning continues to revolutionize multiple industries due to its ability to handle complex problems and massive datasets.


    """
    summary = summarization_agent(large_text)
    print(summary)
