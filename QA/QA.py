from transformers import pipeline

answerer = pipeline('question-answering', model = 'distilbert-base-cased-distilled-squad', device=0)

def answer(context: str, caption1: str, caption2: str, transform_question=None):
    if transform_question:
        caption1 = transform_question(caption1)
        caption2 = transform_question(caption2)
    answer1 = answerer(context=context, question=caption1)['answer']
    answer2 = answerer(context=context, question=caption2)['answer']
    return (answer1, answer2)

def answer_no_context(context: str, caption1: str, caption2: str, transform_question=None):
    if transform_question:
        caption1 = transform_question(caption1)
        caption2 = transform_question(caption2)
    answer1 = answerer(context=context, question=caption1)['answer']
    answer2 = answerer(context=context, question=caption2)['answer']
    return (answer1, answer2)


# print(answer("I live in Boston and it takes me 4 hours to walk to school", 'Where do I live?', "How do I get to school?"))
