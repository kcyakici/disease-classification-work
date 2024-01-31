# Aslı Candan Yıldırım 260201061
# Kürşat Çağrı Yakıcı 290201098

import csv
from pymed import PubMed

# Create a PubMed object that GraphQL can use to query
# Note that the parameters are not required but kindly requested by PubMed Central
# https://www.ncbi.nlm.nih.gov/pmc/tools/developers/
pubmed = PubMed(tool="MyTool", email="my@email.address")

# Create a GraphQL query in plain text
query = 'rheumatism[Title]'

# Execute the query against the API
results = pubmed.query(query, max_results=1000)

with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(['Article Count', 'Article ID', 'Publication Date',
                       'Title', 'Keywords', 'Abstract'])

    total_article_count = 0
    max_article_count = 500

    for article in results:
        if article.abstract is None:
            continue

        article_id = article.pubmed_id
        # Line to work around a bug in PyMed, see: https://github.com/gijswobben/pymed/pull/30
        if '\n' in article_id:
            continue

        # Some articles don't have keywords
        if hasattr(article, 'keywords') and article.keywords:
            if None in article.keywords:
                article.keywords.remove(None)
            keywords = '", "'.join(article.keywords)
        else:
            keywords = "N/A"

        title = article.title
        publication_date = article.publication_date
        abstract = article.abstract

        total_article_count += 1
        csvwriter.writerow(
            [total_article_count, article_id, publication_date, title, keywords, abstract])

        if total_article_count >= max_article_count:
            break
