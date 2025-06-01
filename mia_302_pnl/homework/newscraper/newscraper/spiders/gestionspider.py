import scrapy


class GestionspiderSpider(scrapy.Spider):
    name = "gestionspider"
    allowed_domains = ["gestion.pe"]
    start_urls = ["https://gestion.pe/archivo/"]


    def parse(self, response):
        articles = response.css('.story-item__title')
        for new in articles:
            relative_url = new.attrib['href']

            if relative_url is not None:
                new_url = 'https://gestion.pe/' + relative_url

                yield response.follow(new_url, callback = self.parse_new_page)

        next_page = response.css('.pagination-date a').attrib['href']

        if (next_page is not None) and (next_page != '/archivo/todas/2025-04-01/'):
            next_page_url = 'https://gestion.pe/' + next_page
            yield response.follow(next_page_url, callback = self.parse)

            # yield{
            #     'title': new.css('h2 a::text').get(),
            #     'category': new.css('.story-item__section::text').get(),
            #     'summit': new.css('.story-item__section').attrib['href'],
            #     'date': new.css('.story-item__date-time::text').get(),
            #     'url': new.css('.story-item__section').attrib['href'],
            # }

    def parse_new_page(self,response):

        content = response.css('.story-contents__font-paragraph')
        description_list = []

        for paragraph in content:
            description_list.append(''.join(paragraph.css('::text').getall()))

        description = '\n'.join(description_list)

        tags = response.css('.st-tags__box h4 a::text').getall()

        tags = [tag for tag in tags if tag.strip() != ""]

        yield{
            'title': response.css('.sht__title::text').get(),
            'category' :  response.css('.sht__title__section a::text').get(),
            'summit' : response.css('.sht__summary::text').get(),
            'description': description,
            'date': response.css('.s-aut__time time::attr(datetime)').get(),
            'autor': response.css('.s-aut__n-row a::text').get(),
            'tags': str(tags),
            'url': response.url,
        }