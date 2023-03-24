from os import remove
from langdetect import detect
from typing import List

def relevant(content: str, context: str) -> bool:
    '''
        Returns : True if the content is relevant to the context
        
        The function is used to check whether to add a paragraph from a post to the context
        when reverse image search for context.

        Parameters context: the context so far
        Parameters content: a paragraph to be added to the context
    '''
    # TODO: Implement semantic comparison or something here, 
    #       For now return True by default
    return len(content.split(' ')) > 11
    # return True

def preprocess_post(result: List, remove_irrelevant = True) -> List:
    '''
        Returns: A list of processed posts

        This function is used to process the parsed result,
        including the following steps:
            Language detection based on title

        Parameters result: The list of parsed posts
        Optional remove_irrelevant: Flag to remove posts using other languages
    '''

    for post in result:
        try:
            if post['title']:
                post['lang'] = detect(post['title'])
        except:
            post['lang'] = 'es'
            
    if remove_irrelevant:
        result = list(filter(lambda x: x['lang'] == 'en', result))

    return result

def preprocess_content(content: str) -> str:
    '''
        This function clean the content using the following steps:
            Replace ”, “, etc with corresponding symbol
    '''
    raise NotImplemented
    
# a = [{'title': 'This is Vietnamese post'}, {'title': 'Ô vui quá xá là vui'}]
# a = preprocess(a)
# print(a)