from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st

model_name = 'deepset/roberta-base-squad2'

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def main():
    st.title("Context Based Question Answering Chatbot")
    question = st.text_input('Enter your question: ')
    context = st.text_input('Enter your context: ')
    if st.button('Get Answer'):
        QA_input = {
            'question': question,
            'context': context    
        }
        response = nlp(QA_input)
        st.write(response['answer'])

if __name__ == "__main__":
    main()
