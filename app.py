from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st

model_name = 'deepset/roberta-base-squad2'

nlp = pipeline('question-answering',model=model_name, tokenizer=model_name)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

##### Taking inputs 

question = input('what is your question > ')
context = input('context about question > ')


QA_input = {
    'question': question,
    'context': context    
}


###### Making Predicitions

response = nlp(QA_input)

print(response['answer'])


def main():
    # Create tabs for different sections
    selected_section = st.radio("Go to", ("Context", "Questions"))

    if selected_section == "Context":
        render_context_section()
    elif selected_section == "Questions":
        render_questions_section()

def render_context_section():
    st.title("Context Section")
    context = st.text_input('Enter your context : ')
    # TODO: Add content and functionality for the context section

def render_questions_section():
    st.title("Questions Section")
    question = st.text_input('Enter your question : ')
    # TODO: Add content and functionality for the questions section
    st.write(response['answer'])

if __name__ == "__main__":
    main()
