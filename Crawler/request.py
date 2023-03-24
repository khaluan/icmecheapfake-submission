from .util import request

def get_content(post):

    url = post['post_url']
    response = request(url)
    return response

# url = 'https://www.nytimes.com/2019/06/13/us/politics/julian-castro-fox-town-hall.html'
# post = {'post_url':url}
# print(crawl_context(post))