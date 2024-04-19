# from requests_html import HTMLSession
# from bs4 import BeautifulSoup
#
#
#
# session = HTMLSession()
# response = session.get(url)
#
# response.html.render()
#
# print(response)
from requests_html import AsyncHTMLSession
asession = AsyncHTMLSession()

async def get_pythonorg():
    url = 'https://www.etenders.gov.za/Home/opportunities?id=1'
    response = await asession.get(url)

    await  response.html.arender()


result = asession.run(get_pythonorg)




