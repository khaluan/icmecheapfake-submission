from .request import *
from bs4 import BeautifulSoup
from .preprocess import relevant
import logging

crawler_log = logging.getLogger('Crawler log')
FORMAT = logging.Formatter('%(levelname)s - %(message)s - %(post_url)s')
LOG_FILE_HANDLER = logging.FileHandler('log/crawler.log', mode = 'w+')
LOG_FILE_HANDLER.setFormatter(FORMAT)
crawler_log.addHandler(LOG_FILE_HANDLER)


def require_Javascript(context):
    # print(context)
    return False

def crawl_context(post):
    content = get_content(post)
    context = parse_content(content, post)
    if require_Javascript(context):
        content = sel.get_content(post)
        # print(content.encode())
        context = parse_content(content, post)
        crawler_log.error('Crawling with Selenium', extra=post)
    else:
        crawler_log.error('Crawling with requests', extra=post)
        pass

    return context

def parse_content(response, post):
    soup = BeautifulSoup(response, 'html.parser')

    # Parse heading
    try:
        heading = soup.find('h1').text
    except:
        crawler_log.error('No heading', extra=post)
        heading = ''
    # print("Parsed heading")
    # Parse caption
    captions = soup.findAll('figcaption')
    if len(captions) == 0:
        caption = ''
    elif len(captions) == 1:
        caption = captions[0].text
    else:
        # TODO: Choose the correct caption
        crawler_log.warning('Multiple captions', extra=post)
        caption = ''
    # print("Parsed Caption")

    # Parse content
    content = soup.findAll('p')
    context = ''
    for par in content:
        sentences = par.text.split('\n')
        for s in sentences:
            if relevant(s.strip(), context):
                context += '\n' + s

    if not context:
        crawler_log.error('No content', extra=post)    
    # print("Parsed Content")

    return {
        'heading': heading.strip(),
        'caption': caption.strip(),
        'context': context.strip()
    }   

def crawl_context_raw(post):
    # return req.get_content(post), sel.get_content(post)
    return get_content(post), None

# post = {'post_url': 'https://twitter.com/therealendoreti'}
# print(crawl_context(post))