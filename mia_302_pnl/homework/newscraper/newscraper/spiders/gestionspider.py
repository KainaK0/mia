import scrapy


class GestionspiderSpider(scrapy.Spider):
    name = "gestionspider"
    allowed_domains = ["gestion.pe"]
    start_urls = ["https://gestion.pe/archivo/"]


    def parse(self, response):
        articles = response.css('.story-item')

        for new in articles:
            yield{
                'title': new.css('h2 a::text').get(),
                'category': new.css('.story-item__section::text').get(),
                'summit': new.css('.story-item__section').attrib['href'],
                'date': new.css('.story-item__date-time::text').get(),
                'url': new.css('.story-item__section').attrib['href'],
            }

            # in_page_url = 'https://gestion.pe/' + new.css('.story-item__section').attrib['href']

            
            

            # 'summit': new.css('.entradilla-teaser-2col p::text').get(),


        next_page = response.css('.pagination-date a').attrib['href']

        if (next_page is not None) and (next_page != '/archivo/todas/2025-04-01/'):
            next_page_url = 'https://gestion.pe/' + next_page
            yield response.follow(next_page_url, callback = self.parse)