import scrapy


class NewsspiderSpider(scrapy.Spider):
    name = "newsspider"
    allowed_domains = ["peru21.pe"]
    start_urls = ["https://peru21.pe/archivo/"]

    def parse(self, response):
        articles = response.css('article.node--type-article')

        for new in articles:
            yield{
                'title': new.css('a h2::text').get(),
                'category': new.css('.field__item a::text').get(),
                'summit': new.css('.entradilla-teaser-2col p::text').get(),
                'date': new.css('.field--name-field-fecha-actualizacion ::text').get(),
                'url': new.css('a').attrib['href'],
            }

        next_page = response.css('.pager li.pager__item--next a ::attr(href)').get()

        if (next_page is not None) and (next_page != '/archivo/10/'):
            next_page_url = 'https://peru21.pe' + next_page
            yield response.follow(next_page_url, callback = self.parse)